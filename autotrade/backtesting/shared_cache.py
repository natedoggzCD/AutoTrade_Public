"""Zero-copy sharing of cross-sectional cache DataFrames across worker processes.

The lab orchestrator builds large universe-wide rank/return matrices once in
the parent (see `build_default_alpha_context`). Without sharing, every
ProcessPool worker on Windows spawn gets its own private unpickled copy --
at full universe scale that's ~1-3 GB per worker, so 20 workers consume
~60 GB and trigger OOM during init.

This module hosts those matrices in `multiprocessing.shared_memory` blocks
so all workers reference the *same* memory pages. Per-job IPC stays tiny
(just the SharedFrameRef metadata: shm name + shape + dtype + index/columns).
Workers reconstruct DataFrame views that read from the shared bytes without
copying.

Lifecycle:
- Parent calls `share_dataframe()` for each cache entry; the returned
  `SharedFrameRef` is what gets pickled to workers. The returned
  `SharedMemory` handles are kept alive in the parent (we collect them in a
  list) so the OS doesn't reclaim the blocks while workers are using them.
- Workers call `attach_dataframe(ref)` to rehydrate. They must keep their
  attached `SharedMemory` handle alive for the lifetime of the view -- we
  stash it on the DataFrame via an attribute on the wrapping ndarray's base.
- On run completion the parent closes + unlinks every block.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SharedFrameRef:
    """Pickleable handle to a DataFrame whose values live in shared memory."""
    shm_name: str
    shape: Tuple[int, int]
    dtype_str: str
    index: pd.Index
    columns: pd.Index


def share_dataframe(
    df: pd.DataFrame,
) -> Tuple[SharedFrameRef, shared_memory.SharedMemory]:
    """Copy ``df``'s values into a new shared-memory block and return a ref.

    The caller must keep the returned ``SharedMemory`` object alive (e.g. by
    appending it to a list) until all workers are done; the OS reclaims the
    block when the last handle is closed.

    DataFrames with non-numeric columns are not supported -- the lab cs_cache
    is all float64 so this is fine in practice.
    """
    if df.empty:
        # Allocate a minimal placeholder so the ref still attaches cleanly.
        arr = np.zeros((0, 0), dtype=np.float64)
    else:
        arr = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    shm = shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
    if arr.nbytes > 0:
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
    ref = SharedFrameRef(
        shm_name=shm.name,
        shape=arr.shape,
        dtype_str=str(arr.dtype),
        index=df.index.copy(),
        columns=df.columns.copy(),
    )
    return ref, shm


# Worker-process registry of attached SharedMemory handles. Keeping these alive
# is required so the numpy views over them stay valid for the worker's lifetime.
_ATTACHED_HANDLES: List[shared_memory.SharedMemory] = []


def attach_dataframe(ref: SharedFrameRef) -> pd.DataFrame:
    """Attach to ``ref``'s shared-memory block and return a zero-copy view."""
    shm = shared_memory.SharedMemory(name=ref.shm_name)
    _ATTACHED_HANDLES.append(shm)  # keep alive for this worker process
    if ref.shape == (0, 0) or any(s == 0 for s in ref.shape):
        return pd.DataFrame(index=ref.index, columns=ref.columns, dtype=np.float64)
    dtype = np.dtype(ref.dtype_str)
    arr = np.ndarray(ref.shape, dtype=dtype, buffer=shm.buf)
    # copy=False is important: we want a view, not a duplicate.
    return pd.DataFrame(arr, index=ref.index, columns=ref.columns, copy=False)


def share_cache(
    cache: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[shared_memory.SharedMemory]]:
    """Replace DataFrame entries in ``cache`` with SharedFrameRef references.

    Returns ``(shared_cache, handles)``. Non-DataFrame entries are left
    untouched (they're typically tiny metadata).
    """
    out: Dict[str, Any] = {}
    handles: List[shared_memory.SharedMemory] = []
    for key, value in cache.items():
        if isinstance(value, pd.DataFrame):
            ref, shm = share_dataframe(value)
            out[key] = ref
            handles.append(shm)
        else:
            out[key] = value
    return out, handles


def rehydrate_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    """Attach to every SharedFrameRef in ``cache``, returning DataFrame views."""
    out: Dict[str, Any] = {}
    for key, value in cache.items():
        if isinstance(value, SharedFrameRef):
            out[key] = attach_dataframe(value)
        else:
            out[key] = value
    return out


def release_handles(handles: List[shared_memory.SharedMemory]) -> None:
    """Close + unlink every handle (call once at parent shutdown)."""
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
