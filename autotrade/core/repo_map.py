"""
Repo Map — Aider-inspired symbol index for codebase awareness.

Generates a compact, token-budgeted map of the most relevant files/symbols
in a codebase. Uses Python AST (no tree-sitter dependency) for definition
extraction and a simplified PageRank-like ranking on the file call graph.

Flow:
  1. Walk project → extract definitions (class, function) and references per file
  2. Build directed graph: edges from reference-file → definition-file
  3. Score files by connectivity (simplified PageRank)
  4. Render top-N files with folded code (signatures only, bodies elided)
  5. Fit to token budget via binary search

Usage:
    from autotrade.core.repo_map import RepoMap
    rm = RepoMap(project_root=Path("."), max_tokens=2048)
    map_text = rm.generate(focus_files=["autotrade/core/project_chat.py"])
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("RepoMap")

# =============================================================================
# CONSTANTS
# =============================================================================

# Directories/patterns to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", "dist", "build", "egg-info", ".tox", ".eggs",
    "backups", "logs", "charts", "data", "research",
}

# File extensions to index
INDEX_EXTENSIONS = {".py"}

# Max file size to parse (skip very large generated files)
MAX_FILE_SIZE = 200_000  # 200KB

# Chars per token estimate
CHARS_PER_TOKEN = 4

# Special files that always appear at top of map
SPECIAL_FILES = {
    "README.md", "pyproject.toml", "setup.py", "setup.cfg",
    "requirements.txt", "Makefile", "Dockerfile",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SymbolDef:
    """A symbol definition (class, function, method)."""
    name: str
    kind: str           # "class", "function", "method", "constant"
    file: str           # Relative path
    line: int
    signature: str      # Human-readable signature line
    indent: int = 0     # Indentation level


@dataclass
class SymbolRef:
    """A reference to a symbol name in a file."""
    name: str
    file: str           # Relative path
    line: int


@dataclass
class FileInfo:
    """Parsed information about a single file."""
    rel_path: str
    definitions: List[SymbolDef] = field(default_factory=list)
    references: Set[str] = field(default_factory=set)  # Just the names referenced
    size: int = 0
    lines: int = 0
    mtime: float = 0.0


# =============================================================================
# AST-BASED SYMBOL EXTRACTOR
# =============================================================================

class SymbolExtractor(ast.NodeVisitor):
    """Extract definitions and references from Python AST."""

    def __init__(self, file_rel_path: str, source_lines: List[str]):
        self.file = file_rel_path
        self.source_lines = source_lines
        self.definitions: List[SymbolDef] = []
        self.references: Set[str] = set()
        self._class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        sig = self._get_line(node.lineno)
        self.definitions.append(SymbolDef(
            name=node.name,
            kind="class",
            file=self.file,
            line=node.lineno,
            signature=sig,
            indent=node.col_offset,
        ))
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        kind = "method" if self._class_stack else "function"
        sig = self._get_line(node.lineno)
        self.definitions.append(SymbolDef(
            name=node.name,
            kind=kind,
            file=self.file,
            line=node.lineno,
            signature=sig,
            indent=node.col_offset,
        ))
        # Don't recurse into function bodies for definitions (only top-level)
        # But DO visit for references
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign):
        """Capture module-level constants (ALL_CAPS)."""
        if isinstance(node, ast.Assign) and not self._class_stack:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.definitions.append(SymbolDef(
                        name=target.id,
                        kind="constant",
                        file=self.file,
                        line=node.lineno,
                        signature=self._get_line(node.lineno),
                        indent=0,
                    ))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Collect symbol references (all identifier uses)."""
        self.references.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Collect attribute references like obj.method."""
        self.references.add(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Collect call references."""
        if isinstance(node.func, ast.Name):
            self.references.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.references.add(node.func.attr)
        self.generic_visit(node)

    def _get_line(self, lineno: int) -> str:
        """Get the source line (1-indexed), clean and truncated."""
        if 0 < lineno <= len(self.source_lines):
            line = self.source_lines[lineno - 1].rstrip()
            return line[:120]
        return ""


def extract_symbols(file_path: Path, rel_path: str) -> Optional[FileInfo]:
    """Parse a Python file and extract symbols."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        if len(source) > MAX_FILE_SIZE:
            return None

        lines = source.split("\n")
        tree = ast.parse(source, filename=str(file_path))

        extractor = SymbolExtractor(rel_path, lines)
        extractor.visit(tree)

        return FileInfo(
            rel_path=rel_path,
            definitions=extractor.definitions,
            references=extractor.references,
            size=len(source),
            lines=len(lines),
            mtime=file_path.stat().st_mtime,
        )
    except SyntaxError:
        # File has syntax errors — still index what we can with regex fallback
        return _regex_fallback(file_path, rel_path)
    except Exception as e:
        logger.debug(f"Failed to parse {rel_path}: {e}")
        return None


def _regex_fallback(file_path: Path, rel_path: str) -> Optional[FileInfo]:
    """Fallback: regex-based extraction when AST parsing fails."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        lines = source.split("\n")
        defs = []
        refs = set()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("class "):
                name = stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
                defs.append(SymbolDef(name=name, kind="class", file=rel_path,
                                      line=i, signature=line.rstrip()[:120]))
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                name = stripped.replace("async ", "").replace("def ", "").split("(")[0].strip()
                indent = len(line) - len(line.lstrip())
                kind = "method" if indent > 0 else "function"
                defs.append(SymbolDef(name=name, kind=kind, file=rel_path,
                                      line=i, signature=line.rstrip()[:120], indent=indent))

            # Simple reference extraction: identifiers that look like names
            for m in re.finditer(r'\b([A-Za-z_]\w+)\b', stripped):
                refs.add(m.group(1))

        return FileInfo(
            rel_path=rel_path,
            definitions=defs,
            references=refs,
            size=len(source),
            lines=len(lines),
            mtime=file_path.stat().st_mtime,
        )
    except Exception:
        return None


# =============================================================================
# FILE RANKING (simplified PageRank)
# =============================================================================

def rank_files(
    files: Dict[str, FileInfo],
    focus_files: Optional[List[str]] = None,
    focus_symbols: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Rank files by relevance using a simplified PageRank on the call graph.
    
    Builds a directed graph where:
      Node = relative file path
      Edge(A→B) = file A references a symbol defined in file B
    
    Inspired by Aider's repo-map ranking but without networkx dependency.
    Uses iterative power method instead.
    
    Args:
        files: Dict of rel_path → FileInfo
        focus_files: Files currently in context (get a boost)
        focus_symbols: Symbols mentioned by user (get a boost)
    
    Returns:
        List of (rel_path, score) sorted by score descending
    """
    focus_files = set(focus_files or [])
    focus_symbols = set(focus_symbols or [])

    # Build definition index: symbol_name → list of files that define it
    def_index: Dict[str, List[str]] = defaultdict(list)
    for rel_path, finfo in files.items():
        for d in finfo.definitions:
            def_index[d.name].append(rel_path)

    # Build adjacency: for each file, where do its references point?
    # adj[A] = {B: weight} means file A references symbols defined in file B
    adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for rel_path, finfo in files.items():
        for ref_name in finfo.references:
            if ref_name not in def_index:
                continue
            defining_files = def_index[ref_name]
            if len(defining_files) > 5:
                # Generic name (defined in many files) — low signal
                weight = 0.1
            else:
                weight = 1.0

            # Boost: symbol mentioned by user or looks important
            if ref_name in focus_symbols:
                weight *= 10.0
            if len(ref_name) >= 8 and ("_" in ref_name or any(c.isupper() for c in ref_name[1:])):
                weight *= 2.0  # snake_case or camelCase = real name
            if ref_name.startswith("_"):
                weight *= 0.1  # Private

            # Boost: referencer is a focus file
            if rel_path in focus_files:
                weight *= 10.0

            for def_file in defining_files:
                if def_file != rel_path:  # Skip self-references
                    adj[rel_path][def_file] += weight

    # Power iteration (simplified PageRank, 20 iterations)
    all_files = list(files.keys())
    n = len(all_files)
    if n == 0:
        return []

    file_idx = {f: i for i, f in enumerate(all_files)}
    damping = 0.85
    scores = [1.0 / n] * n

    # Personalization: focus files get higher initial score
    if focus_files:
        for f in focus_files:
            if f in file_idx:
                scores[file_idx[f]] = 100.0 / n

    for _ in range(20):
        new_scores = [(1 - damping) / n] * n
        for src_file, targets in adj.items():
            if src_file not in file_idx:
                continue
            src_idx = file_idx[src_file]
            total_weight = sum(targets.values())
            if total_weight == 0:
                continue
            for tgt_file, weight in targets.items():
                if tgt_file not in file_idx:
                    continue
                tgt_idx = file_idx[tgt_file]
                new_scores[tgt_idx] += damping * scores[src_idx] * (weight / total_weight)
        scores = new_scores

    # Sort by score
    ranked = [(all_files[i], scores[i]) for i in range(n)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =============================================================================
# MAP RENDERER
# =============================================================================

def render_file_map(finfo: FileInfo, max_defs: int = 30) -> str:
    """
    Render a file's definition signatures in a compact, folded format.
    
    Output looks like:
        autotrade/core/project_chat.py (3667 lines):
        │class ModelConfig:
        │class ProjectRAG:
        │    def __init__(self, project_root):
        │    def search(self, query, top_k=5):
        │class ProjectChat:
        │    def __init__(self):
        │    def _handle_agent_command(self, arg):
        ⋮
    """
    lines = [f"{finfo.rel_path} ({finfo.lines} lines):"]

    defs = finfo.definitions[:max_defs]
    for d in defs:
        indent_str = "  " * (d.indent // 4) if d.indent else ""
        # Clean up the signature line
        sig = d.signature.strip()
        # Keep just the def/class line, no body
        if ":" in sig:
            sig = sig.split("#")[0].rstrip()  # Remove inline comments
        lines.append(f"  {indent_str}{sig}")

    if len(finfo.definitions) > max_defs:
        lines.append(f"  ... (+{len(finfo.definitions) - max_defs} more definitions)")

    return "\n".join(lines)


# =============================================================================
# REPO MAP CLASS
# =============================================================================

class RepoMap:
    """
    Generates a compact, token-budgeted map of the most relevant files/symbols.
    
    Usage:
        rm = RepoMap(project_root=Path("."), max_tokens=2048)
        map_text = rm.generate(focus_files=["autotrade/core/project_chat.py"])
    """

    def __init__(
        self,
        project_root: Path,
        max_tokens: int = 2048,
        include_special: bool = True,
    ):
        self.root = project_root
        self.max_tokens = max_tokens
        self.include_special = include_special
        self._cache: Dict[str, FileInfo] = {}
        self._cache_time: float = 0

    def _scan_files(self) -> Dict[str, FileInfo]:
        """Walk project and extract symbols from all Python files."""
        # Use cache if less than 30 seconds old
        if self._cache and (time.time() - self._cache_time) < 30:
            return self._cache

        files: Dict[str, FileInfo] = {}
        t0 = time.time()

        for root_dir, dirs, filenames in os.walk(self.root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                if not any(fname.endswith(ext) for ext in INDEX_EXTENSIONS):
                    continue

                full_path = Path(root_dir) / fname
                try:
                    rel_path = str(full_path.relative_to(self.root)).replace("\\", "/")
                except ValueError:
                    continue

                finfo = extract_symbols(full_path, rel_path)
                if finfo:
                    files[rel_path] = finfo

        elapsed = time.time() - t0
        logger.debug(f"Scanned {len(files)} files in {elapsed:.2f}s")

        self._cache = files
        self._cache_time = time.time()
        return files

    def generate(
        self,
        focus_files: Optional[List[str]] = None,
        focus_symbols: Optional[List[str]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Generate the repo map, fitted to the token budget.
        
        Args:
            focus_files: Files currently relevant (get higher ranking)
            focus_symbols: Symbol names mentioned in the task
            task_description: Optional task text to auto-extract focus symbols
            
        Returns:
            A compact text map of the most relevant files and their signatures
        """
        files = self._scan_files()
        if not files:
            return "(no Python files found in project)"

        # Auto-extract focus symbols from task description
        if task_description and not focus_symbols:
            focus_symbols = self._extract_symbols_from_text(task_description, files)

        # Rank files
        ranked = rank_files(files, focus_files, focus_symbols)

        # Binary search: how many files fit in the token budget?
        max_chars = self.max_tokens * CHARS_PER_TOKEN
        header = "## Repo Map (most relevant files)\n\n"

        # Start with special files
        special_text = ""
        if self.include_special:
            for sf in SPECIAL_FILES:
                sf_path = self.root / sf
                if sf_path.exists():
                    special_text += f"{sf}\n"
            if special_text:
                special_text = "### Project files\n" + special_text + "\n"

        # Binary search for optimal number of files
        lo, hi = 1, min(len(ranked), 50)
        best_text = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            text = header + special_text + "### Code structure\n\n"

            for rel_path, _score in ranked[:mid]:
                finfo = files[rel_path]
                text += render_file_map(finfo) + "\n\n"

            if len(text) <= max_chars:
                best_text = text
                lo = mid + 1
            else:
                hi = mid - 1

        if not best_text:
            # Even 1 file is too much — truncate
            if ranked:
                rel_path, _ = ranked[0]
                best_text = header + render_file_map(files[rel_path], max_defs=10)

        return best_text.rstrip()

    def _extract_symbols_from_text(
        self, text: str, files: Dict[str, FileInfo]
    ) -> List[str]:
        """Extract likely symbol names from task text that match known definitions."""
        # Build set of all known symbol names
        all_defs = set()
        for finfo in files.values():
            for d in finfo.definitions:
                all_defs.add(d.name)

        # Find words in the text that match known definitions
        words = set(re.findall(r'\b([A-Za-z_]\w{2,})\b', text))
        matches = words & all_defs

        # Also match file paths mentioned in the text
        file_matches = []
        for word in re.findall(r'[\w/\\]+\.py', text):
            normalized = word.replace("\\", "/")
            for rel_path in files:
                if normalized in rel_path:
                    file_matches.append(rel_path)

        return list(matches)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase."""
        files = self._scan_files()
        total_defs = sum(len(f.definitions) for f in files.values())
        total_lines = sum(f.lines for f in files.values())
        return {
            "files": len(files),
            "definitions": total_defs,
            "total_lines": total_lines,
            "avg_defs_per_file": total_defs / max(len(files), 1),
        }
