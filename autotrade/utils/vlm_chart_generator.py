"""
VLM-Optimized Chart Generator (Option-2 Layout: No Volume)

4-Panel Layout: Price + MACD + RSI + Stochastic
Optimized for Vision Language Models (VLMs) like llama3.2-vision and qwen2.5vl.

Features:
- 1920x1200 @ 150dpi with tight layout
- 4-row GridSpec with ratios [68, 14, 9, 9]
- NO volume panel (replaced with Stochastic)
- MACD(12,26,9) with histogram
- RSI(14) with 30/50/70 lines
- Stochastic: FastK + SlowD only (simplified for VLM)
- S/R zones, swing markers, AVWAP(H) with proper anchor marker
- UNIFIED SIGNAL_STYLE: Colored dots with ID inside, black edge
- Single "Signals:" key strip with colored dots matching markers
- Leader lines for collision avoidance

Signal Style Map (single source of truth):
  1 = MACD (blue #1976D2)
  2 = AVWAP (purple #9C27B0)
  3 = BRK (teal #26A69A)
  4 = STOCH<20 (green #4CAF50)
  5 = RSI>50 (orange #FF9800)

Author: AutoTrade Self-Healing Agent
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import logging

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.patheffects import withStroke

logger = logging.getLogger(__name__)

# Chart output directory
CHARTS_DIR = Path(__file__).parent.parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


# ===== UNIFIED SIGNAL STYLE MAP (single source of truth) =====
# ALL signals rendered as BLACK circles with WHITE numbers for maximum VLM clarity
SIGNAL_STYLE = {
    1: {'name': 'MACD×', 'desc': 'MACD bullish crossover'},
    2: {'name': 'AVWAP', 'desc': 'Price reclaims AVWAP'},
    3: {'name': 'STOCH×', 'desc': 'Stochastic crossover'},
    4: {'name': 'RSI>50', 'desc': 'RSI crosses above 50'},
    5: {'name': 'SMA×', 'desc': 'Price crosses above SMA20'},
}

# Signal marker styling - BLACK circles with WHITE text for all
SIGNAL_MARKER_BG = '#1A1A1A'      # Black background
SIGNAL_MARKER_TEXT = '#FFFFFF'    # White text
SIGNAL_MARKER_EDGE = '#FFFFFF'    # White edge for contrast


class VLMChartConfig:
    """Configuration for VLM-optimized charts."""
    
    # Canvas - HARD REQUIREMENTS
    FIGSIZE = (19.2, 12)  # 1920x1200 at 100dpi
    DPI = 150
    
    # Panel ratios - Price/MACD/RSI/STOCH (NO VOLUME)
    PANEL_RATIOS = [68, 14, 9, 9]
    
    # Colors - High contrast
    BG_COLOR = '#FFFFFF'
    TEXT_COLOR = '#1A1A1A'
    GRID_COLOR = '#E8E8E8'
    
    # Candle colors
    UP_COLOR = '#26A69A'      # Teal green
    DOWN_COLOR = '#EF5350'    # Coral red
    
    # Indicator colors
    SMA20_COLOR = '#2196F3'   # Blue
    SMA50_COLOR = '#FF9800'   # Orange
    AVWAP_COLOR = '#9C27B0'   # Purple (thick)
    
    # MACD colors
    MACD_LINE_COLOR = '#2196F3'    # Blue
    MACD_SIGNAL_COLOR = '#FF9800'  # Orange
    MACD_HIST_UP = '#26A69A'
    MACD_HIST_DOWN = '#EF5350'
    
    # RSI colors
    RSI_COLOR = '#7C4DFF'
    
    # Stochastic colors (simplified: FastK + SlowD only)
    FAST_K_COLOR = '#2196F3'   # Blue - Fast%K
    SLOW_D_COLOR = '#FF9800'   # Orange - Slow%D
    
    # NOW marker colors
    NOW_LINE_COLOR = '#1976D2'     # Strong blue for NOW vertical line
    NOW_LINE_WIDTH = 2.5
    PX_LINE_COLOR = '#1976D2'      # Same blue for PX horizontal
    
    # S/R Zone colors
    SUPPORT_COLOR = '#4CAF50'
    RESISTANCE_COLOR = '#F44336'
    SUPPORT_ALPHA = 0.15
    RESISTANCE_ALPHA = 0.15
    
    # Marker colors
    SWING_HIGH_COLOR = '#D32F2F'
    SWING_LOW_COLOR = '#388E3C'
    
    # Signal dot styling
    SIGNAL_DOT_SIZE = 220       # Marker size for scatter (larger for ID inside)
    SIGNAL_DOT_EDGE_WIDTH = 1.2  # Black edge width
    SIGNAL_ID_FONT_SIZE = 9      # Font size for ID inside dot
    SIGNAL_ALPHA_ACTIVE = 1.0
    SIGNAL_ALPHA_STALE = 0.35
    
    # Collision avoidance
    SIGNAL_STACK_OFFSET_FACTOR = 0.5  # Multiplier of ATR for vertical stacking
    SIGNAL_LEADER_LINE_WIDTH = 0.8
    SIGNAL_LEADER_LINE_COLOR = '#666666'
    
    # Font sizes - VLM friendly (>= 12)
    TITLE_SIZE = 18
    LABEL_SIZE = 14
    TICK_SIZE = 12
    ANNOTATION_SIZE = 12
    LEGEND_SIZE = 11
    SR_LABEL_SIZE = 12
    
    # Line widths (>= 2 for all indicators)
    CANDLE_WIDTH = 0.6
    WICK_WIDTH = 1.2
    MA_LINE_WIDTH = 2.0
    AVWAP_LINE_WIDTH = 2.5
    MACD_LINE_WIDTH = 2.0
    RSI_LINE_WIDTH = 2.0
    STOCH_LINE_WIDTH = 2.0
    
    # Signal age gating
    MAX_SIGNAL_AGE = 5  # bars
    MAX_DISPLAYED_SIGNALS = 3  # Hard cap on displayed signals
    
    # Lookback
    DEFAULT_DAYS = 60


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators with EXACT formulas.
    """
    df = df.copy()
    
    # ===== Moving Averages =====
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    
    # ===== MACD (12, 26, 9) =====
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ===== RSI (14) =====
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ===== STOCHASTICS =====
    N = 14
    K_smooth = 3
    D_smooth = 3
    
    df['HighestHigh'] = df['High'].rolling(N).max()
    df['LowestLow'] = df['Low'].rolling(N).min()
    
    denom = df['HighestHigh'] - df['LowestLow']
    denom = denom.replace(0, np.nan)
    df['Fast_K'] = 100 * (df['Close'] - df['LowestLow']) / denom
    df['Slow_K'] = df['Fast_K'].rolling(K_smooth).mean()
    df['Slow_D'] = df['Slow_K'].rolling(D_smooth).mean()
    
    # ===== ATR (14) =====
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # ===== Swing Highs detection (pivot window L=3) =====
    L = 3
    swing_high_indices = []
    highs = df['High'].values
    
    for i in range(L, len(df) - L):
        window_start = max(0, i - L)
        window_end = min(len(df), i + L + 1)
        window = highs[window_start:window_end]
        
        if highs[i] == np.max(window):
            if np.sum(window == highs[i]) == 1:
                swing_high_indices.append(i)
    
    df['is_swing_high'] = False
    if swing_high_indices:
        df.iloc[swing_high_indices, df.columns.get_loc('is_swing_high')] = True
    
    # ===== Swing Lows detection =====
    swing_low_indices = []
    lows = df['Low'].values
    
    for i in range(L, len(df) - L):
        window_start = max(0, i - L)
        window_end = min(len(df), i + L + 1)
        window = lows[window_start:window_end]
        
        if lows[i] == np.min(window):
            if np.sum(window == lows[i]) == 1:
                swing_low_indices.append(i)
    
    df['is_swing_low'] = False
    if swing_low_indices:
        df.iloc[swing_low_indices, df.columns.get_loc('is_swing_low')] = True
    
    # ===== Anchored VWAP from most recent swing high =====
    df['AVWAP_high'] = np.nan
    
    avwap_anchor_idx = None
    avwap_anchor_price = None
    
    if swing_high_indices:
        t0 = swing_high_indices[-1]
        avwap_anchor_idx = t0
        avwap_anchor_price = df['High'].iloc[t0]
        
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        cum_pv = 0.0
        cum_v = 0.0
        
        for i in range(t0, len(df)):
            tp = df['TP'].iloc[i]
            vol = df['Volume'].iloc[i]
            
            if pd.notna(tp) and pd.notna(vol) and vol > 0:
                cum_pv += tp * vol
                cum_v += vol
                df.iloc[i, df.columns.get_loc('AVWAP_high')] = cum_pv / cum_v
    
    df.attrs['avwap_anchor_idx'] = avwap_anchor_idx
    df.attrs['avwap_anchor_price'] = avwap_anchor_price
    
    return df


def detect_events(df: pd.DataFrame, sr_data: Optional[Dict] = None) -> Dict[str, List]:
    """
    Detect chart events with unified signal system.
    Uses SIGNAL_STYLE for consistent ID mapping.
    """
    cfg = VLMChartConfig()
    events = {
        'swing_highs': [],
        'swing_lows': [],
        'support_zones': [],
        'resistance_zones': [],
        'signals': []
    }
    
    # Collect swing points
    for i, (idx, row) in enumerate(df.iterrows()):
        if row.get('is_swing_high', False):
            events['swing_highs'].append((idx, row['High'], i))
        if row.get('is_swing_low', False):
            events['swing_lows'].append((idx, row['Low'], i))
    
    events['swing_highs'] = events['swing_highs'][-5:]
    events['swing_lows'] = events['swing_lows'][-5:]
    
    # S/R zones
    if sr_data:
        for level in sr_data.get('support_levels', []):
            events['support_zones'].append(level)
        for level in sr_data.get('resistance_levels', []):
            events['resistance_zones'].append(level)
    else:
        if events['swing_lows']:
            recent_lows = [s[1] for s in events['swing_lows'][-3:]]
            if recent_lows:
                events['support_zones'].append(np.mean(recent_lows))
        
        if events['swing_highs']:
            recent_highs = [s[1] for s in events['swing_highs'][-3:]]
            if recent_highs:
                events['resistance_zones'].append(np.mean(recent_highs))
    
    # ===== Detect signals using iloc for proper alignment =====
    n = len(df)
    signal_candidates = []
    
    # Helper to add a signal at the EXACT bar where crossover occurs
    def add_signal(bar_i: int, sig_id: int, priority: int):
        """Add signal at exact iloc position bar_i."""
        if bar_i < 0 or bar_i >= n:
            return
        age = (n - 1) - bar_i
        if age > cfg.MAX_SIGNAL_AGE:
            return  # Too old
        signal_candidates.append({
            'bar_idx': bar_i,  # This is the x-coordinate for plotting
            'high': df['High'].iloc[bar_i],
            'low': df['Low'].iloc[bar_i],
            'atr': df['ATR'].iloc[bar_i] if pd.notna(df['ATR'].iloc[bar_i]) else 1.0,
            'signal_id': sig_id,
            'priority': priority,
            'age': age,
            'is_active': True
        })
    
    # Scan last 20 bars for crossovers using iloc (positions 0 to n-1)
    lookback = min(20, n - 1)
    start_i = n - lookback
    
    for i in range(start_i, n):
        if i < 1:
            continue  # Need previous bar
        
        # Get current and previous bar values
        macd_curr = df['MACD'].iloc[i]
        macd_prev = df['MACD'].iloc[i-1]
        sig_curr = df['MACD_Signal'].iloc[i]
        sig_prev = df['MACD_Signal'].iloc[i-1]
        
        rsi_curr = df['RSI'].iloc[i]
        rsi_prev = df['RSI'].iloc[i-1]
        
        fk_curr = df['Fast_K'].iloc[i]
        fk_prev = df['Fast_K'].iloc[i-1]
        sd_curr = df['Slow_D'].iloc[i]
        sd_prev = df['Slow_D'].iloc[i-1]
        
        close_curr = df['Close'].iloc[i]
        close_prev = df['Close'].iloc[i-1]
        sma20_curr = df['SMA20'].iloc[i]
        sma20_prev = df['SMA20'].iloc[i-1]
        avwap_curr = df['AVWAP_high'].iloc[i] if pd.notna(df['AVWAP_high'].iloc[i]) else None
        avwap_prev = df['AVWAP_high'].iloc[i-1] if pd.notna(df['AVWAP_high'].iloc[i-1]) else None
        
        # 1) MACD bullish crossover: MACD crosses above Signal
        if pd.notna(macd_curr) and pd.notna(macd_prev) and pd.notna(sig_curr) and pd.notna(sig_prev):
            if macd_prev <= sig_prev and macd_curr > sig_curr:
                add_signal(i, 1, 1)  # High priority
        
        # 2) AVWAP reclaim: Close crosses above AVWAP
        if avwap_curr is not None and avwap_prev is not None:
            if close_prev <= avwap_prev and close_curr > avwap_curr:
                add_signal(i, 2, 2)
        
        # 3) STOCH crossover: FastK crosses above SlowD (anywhere, not just oversold)
        if pd.notna(fk_curr) and pd.notna(fk_prev) and pd.notna(sd_curr) and pd.notna(sd_prev):
            if fk_prev <= sd_prev and fk_curr > sd_curr:
                add_signal(i, 3, 3)
        
        # 4) RSI crosses above 50
        if pd.notna(rsi_curr) and pd.notna(rsi_prev):
            if rsi_prev < 50 and rsi_curr >= 50:
                add_signal(i, 4, 4)
        
        # 5) SMA20 crossover: Close crosses above SMA20
        if pd.notna(sma20_curr) and pd.notna(sma20_prev):
            if close_prev <= sma20_prev and close_curr > sma20_curr:
                add_signal(i, 5, 5)
    
    # Sort by bar_idx (most recent first) then priority
    signal_candidates.sort(key=lambda x: (-x['bar_idx'], x['priority']))
    
    # Take up to MAX_DISPLAYED_SIGNALS
    events['signals'] = signal_candidates[:cfg.MAX_DISPLAYED_SIGNALS]
    
    return events


def _compute_signal_positions(signals: List[Dict], cfg: VLMChartConfig) -> List[Dict]:
    """
    Compute y-positions for signal dots with collision avoidance.
    Stacks dots vertically and adds leader line info if needed.
    """
    if not signals:
        return signals
    
    # Group signals by bar_idx for collision detection
    bar_groups = {}
    for sig in signals:
        bar_idx = sig['bar_idx']
        if bar_idx not in bar_groups:
            bar_groups[bar_idx] = []
        bar_groups[bar_idx].append(sig)
    
    # Compute positions with stacking
    for bar_idx, group in bar_groups.items():
        for rank, sig in enumerate(group):
            atr = sig['atr']
            
            # Base position above the high
            base_y = sig['high'] + 0.3 * atr
            
            # Stack offset for multiple signals on same bar
            stack_offset = rank * cfg.SIGNAL_STACK_OFFSET_FACTOR * atr
            sig['y_pos'] = base_y + stack_offset
            
            # Candle attachment point (top of candle)
            sig['candle_y'] = sig['high']
            
            # Need leader line if stacked (rank > 0) or offset is significant
            sig['needs_leader'] = rank > 0 or stack_offset > 0.2 * atr
    
    return signals


def _build_signal_legend(ax: plt.Axes, cfg: VLMChartConfig, price_max: float, price_range: float):
    """
    Build compact signal legend with colored dots using SIGNAL_STYLE.
    Format: ●1 MACD  ●2 AVWAP  ●3 BRK  ●4 STOCH<20  ●5 RSI>50
    """
    legend_y = price_max + price_range * 0.02
    
    # Build legend entries with colored dots
    legend_parts = []
    for sig_id in sorted(SIGNAL_STYLE.keys()):
        style = SIGNAL_STYLE[sig_id]
        legend_parts.append(f"●{sig_id} {style['name']}")
    
    legend_text = "Signals: " + "  ".join(legend_parts)
    
    # Draw the legend box
    ax.text(0.5, legend_y, legend_text, fontsize=cfg.LEGEND_SIZE,
           color=cfg.TEXT_COLOR, ha='left', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor=cfg.GRID_COLOR, alpha=0.95))
    
    # Overlay colored dots on top of the black dots in the text
    # This is a workaround since matplotlib text can't have inline colors
    # We'll draw small colored circles at approximate positions
    # For simplicity, we'll use a cleaner approach: draw the legend manually
    
    return legend_y


def _draw_signal_legend_colored(ax: plt.Axes, cfg: VLMChartConfig, x_start: float, y_pos: float):
    """
    Draw signal legend with exact same marker style as chart signals.
    Each entry: colored circle with black edge + white/black ID inside + NAME text
    Matches the signal dots on the chart exactly.
    """
    x = x_start
    spacing = 7.0  # Spacing between entries in data units
    
    # Background box
    total_width = spacing * len(SIGNAL_STYLE) + 5
    ax.add_patch(Rectangle((x - 0.5, y_pos - 0.4), total_width, 1.0,
                           facecolor='white', edgecolor=cfg.GRID_COLOR,
                           alpha=0.95, transform=ax.transData, zorder=8))
    
    # "Signals:" label
    ax.text(x, y_pos, "Signals:", fontsize=cfg.LEGEND_SIZE, fontweight='bold',
           color=cfg.TEXT_COLOR, ha='left', va='center', zorder=9)
    x += 5.0
    
    # Each signal entry - BLACK circle with WHITE number (same as chart markers)
    for sig_id in sorted(SIGNAL_STYLE.keys()):
        style = SIGNAL_STYLE[sig_id]
        
        # BLACK circle with WHITE edge (same as chart markers)
        ax.scatter(x, y_pos, s=cfg.SIGNAL_DOT_SIZE * 0.6, marker='o', 
                   facecolor=SIGNAL_MARKER_BG, edgecolor=SIGNAL_MARKER_EDGE, 
                   linewidths=cfg.SIGNAL_DOT_EDGE_WIDTH, zorder=9)
        
        # WHITE number INSIDE the circle (same as chart markers)
        ax.text(x, y_pos, str(sig_id), fontsize=cfg.SIGNAL_ID_FONT_SIZE - 1,
               fontweight='bold', color=SIGNAL_MARKER_TEXT,
               ha='center', va='center', zorder=10)
        
        # Signal name label (to the right of the marker)
        ax.text(x + 0.8, y_pos, style['name'], fontsize=cfg.LEGEND_SIZE - 1,
               color=cfg.TEXT_COLOR, ha='left', va='center', zorder=9)
        
        x += spacing


def plot_chart(
    df: pd.DataFrame,
    symbol: str,
    events: Dict,
    out_path: Path,
    sr_data: Optional[Dict] = None
) -> bool:
    """
    Create 4-panel chart: Price/MACD/RSI/STOCH.
    """
    cfg = VLMChartConfig()
    
    # Create figure
    fig = plt.figure(figsize=cfg.FIGSIZE, facecolor=cfg.BG_COLOR, constrained_layout=False)
    
    # GridSpec with ratios [68, 14, 9, 9]
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=cfg.PANEL_RATIOS,
        hspace=0.02,
        left=0.06, right=0.94, top=0.95, bottom=0.05
    )
    
    ax_price = fig.add_subplot(gs[0])
    ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_stoch = fig.add_subplot(gs[3], sharex=ax_price)
    
    all_axes = [ax_price, ax_macd, ax_rsi, ax_stoch]
    
    # Style all axes
    for ax in all_axes:
        ax.set_facecolor(cfg.BG_COLOR)
        ax.grid(True, alpha=0.25, color=cfg.GRID_COLOR, linewidth=0.5)
        ax.tick_params(labelsize=cfg.TICK_SIZE, colors=cfg.TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(cfg.GRID_COLOR)
    
    # Index mapping
    dates = df.index.tolist()
    date_to_x = {d: i for i, d in enumerate(dates)}
    x_last = len(df) - 1
    
    # ===== DRAW NOW VERTICAL LINE ACROSS ALL PANELS =====
    for ax in all_axes:
        ax.axvline(x_last, color=cfg.NOW_LINE_COLOR, linestyle='-',
                   linewidth=cfg.NOW_LINE_WIDTH, alpha=0.6, zorder=2)
    
    # ===== PRICE PANEL =====
    _plot_price_panel(ax_price, df, events, date_to_x, cfg, x_last)
    
    # ===== MACD PANEL =====
    _plot_macd_panel(ax_macd, df, date_to_x, cfg)
    
    # ===== RSI PANEL =====
    _plot_rsi_panel(ax_rsi, df, date_to_x, cfg)
    
    # ===== STOCHASTIC PANEL =====
    _plot_stoch_panel(ax_stoch, df, date_to_x, cfg)
    
    # Hide x-axis labels except bottom
    ax_price.tick_params(labelbottom=False)
    ax_macd.tick_params(labelbottom=False)
    ax_rsi.tick_params(labelbottom=False)
    
    # Date formatting on bottom panel ONLY
    ax_stoch.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_stoch.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_stoch.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=cfg.TICK_SIZE)
    
    # Title
    current_price = df['Close'].iloc[-1]
    change_pct = ((df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1) * 100 if len(df) >= 20 else 0
    
    ax_price.set_title(
        f"{symbol}  |  ${current_price:.2f}  |  {change_pct:+.1f}% (20d)  |  NOW",
        fontsize=cfg.TITLE_SIZE, fontweight='bold',
        color=cfg.TEXT_COLOR, loc='left', pad=10
    )
    
    # Save
    fig.savefig(out_path, dpi=cfg.DPI, facecolor=cfg.BG_COLOR, edgecolor='none')
    plt.close(fig)
    
    return True


def plot_intraday_chart(
    df: pd.DataFrame,
    symbol: str,
    events: Dict,
    out_path: Path,
    timeframe_label: str = "1H",
    sr_data: Optional[Dict] = None,
) -> bool:
    """
    Create intraday 4-panel chart with the same VLM visual system.

    Preserves the existing daily chart design while adapting title/context
    for intraday decision support.
    """
    cfg = VLMChartConfig()

    fig = plt.figure(figsize=cfg.FIGSIZE, facecolor=cfg.BG_COLOR, constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=cfg.PANEL_RATIOS,
        hspace=0.02,
        left=0.06,
        right=0.94,
        top=0.95,
        bottom=0.05,
    )

    ax_price = fig.add_subplot(gs[0])
    ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_stoch = fig.add_subplot(gs[3], sharex=ax_price)
    all_axes = [ax_price, ax_macd, ax_rsi, ax_stoch]

    for ax in all_axes:
        ax.set_facecolor(cfg.BG_COLOR)
        ax.grid(True, alpha=0.25, color=cfg.GRID_COLOR, linewidth=0.5)
        ax.tick_params(labelsize=cfg.TICK_SIZE, colors=cfg.TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(cfg.GRID_COLOR)

    dates = df.index.tolist()
    date_to_x = {d: i for i, d in enumerate(dates)}
    x_last = len(df) - 1

    for ax in all_axes:
        ax.axvline(
            x_last,
            color=cfg.NOW_LINE_COLOR,
            linestyle='-',
            linewidth=max(cfg.NOW_LINE_WIDTH, 3.0),
            alpha=0.8,
            zorder=2,
        )

    _plot_price_panel(ax_price, df, events, date_to_x, cfg, x_last)
    _plot_macd_panel(ax_macd, df, date_to_x, cfg)
    _plot_rsi_panel(ax_rsi, df, date_to_x, cfg)
    _plot_stoch_panel(ax_stoch, df, date_to_x, cfg)

    ax_price.tick_params(labelbottom=False)
    ax_macd.tick_params(labelbottom=False)
    ax_rsi.tick_params(labelbottom=False)
    ax_stoch.tick_params(axis='x', labelrotation=0)

    try:
        session_date = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d")
    except Exception:
        session_date = datetime.now().strftime("%Y-%m-%d")

    current_price = float(df['Close'].iloc[-1])
    try:
        open_ref = float(df['Open'].iloc[-1])
    except Exception:
        open_ref = current_price
    move_pct = ((current_price / open_ref) - 1.0) * 100.0 if open_ref > 0 else 0.0

    ax_price.set_title(
        f"{symbol} — Intraday ({timeframe_label}) — {session_date}  |  "
        f"${current_price:.2f}  |  {move_pct:+.1f}% (session)  |  NOW",
        fontsize=cfg.TITLE_SIZE,
        fontweight='bold',
        color=cfg.TEXT_COLOR,
        loc='left',
        pad=10,
    )

    fig.savefig(out_path, dpi=cfg.DPI, facecolor=cfg.BG_COLOR, edgecolor='none')
    plt.close(fig)
    return True


def _plot_price_panel(ax: plt.Axes, df: pd.DataFrame, events: Dict, date_to_x: Dict, cfg: VLMChartConfig, x_last: int):
    """Plot candlesticks, MAs, AVWAP, S/R zones, and unified signal dots with ID inside."""
    
    current_price = df['Close'].iloc[-1]
    price_max = df['High'].max()
    price_min = df['Low'].min()
    price_range = price_max - price_min
    
    # ===== PX HORIZONTAL LINE =====
    ax.axhline(y=current_price, color=cfg.PX_LINE_COLOR, 
               linestyle='--', linewidth=2.0, alpha=0.7, zorder=3)
    
    # ===== PX LABEL =====
    ax.annotate(f'PX: ${current_price:.2f}', xy=(len(df) + 0.5, current_price),
               fontsize=14, fontweight='bold', color='white', ha='left', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=cfg.PX_LINE_COLOR,
                        edgecolor='white', alpha=0.95))
    
    # ===== NOW LABEL =====
    ax.annotate('NOW', xy=(x_last, price_max * 0.995),
               fontsize=12, fontweight='bold', color='white', ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor=cfg.NOW_LINE_COLOR,
                        edgecolor='white', alpha=0.9))
    
    # Plot candlesticks
    for i, (idx, row) in enumerate(df.iterrows()):
        color = cfg.UP_COLOR if row['Close'] >= row['Open'] else cfg.DOWN_COLOR
        
        # Wick
        ax.plot([i, i], [row['Low'], row['High']], 
               color=color, linewidth=cfg.WICK_WIDTH, solid_capstyle='round')
        
        # Body
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        if body_height < 0.001:
            body_height = (row['High'] - row['Low']) * 0.1 or 0.01
        
        rect = Rectangle((i - cfg.CANDLE_WIDTH/2, body_bottom), cfg.CANDLE_WIDTH, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)
    
    # Plot SMAs
    valid_sma20 = df['SMA20'].dropna()
    valid_sma50 = df['SMA50'].dropna()
    
    if len(valid_sma20) > 0:
        x_sma20 = [date_to_x[d] for d in valid_sma20.index if d in date_to_x]
        ax.plot(x_sma20, valid_sma20.values, color=cfg.SMA20_COLOR,
               linewidth=cfg.MA_LINE_WIDTH, alpha=0.9)
    
    if len(valid_sma50) > 0:
        x_sma50 = [date_to_x[d] for d in valid_sma50.index if d in date_to_x]
        ax.plot(x_sma50, valid_sma50.values, color=cfg.SMA50_COLOR,
               linewidth=cfg.MA_LINE_WIDTH, alpha=0.9)
    
    # ===== AVWAP_high =====
    valid_avwap = df['AVWAP_high'].dropna()
    if len(valid_avwap) > 0:
        x_avwap = [date_to_x[d] for d in valid_avwap.index if d in date_to_x]
        ax.plot(x_avwap, valid_avwap.values, color=cfg.AVWAP_COLOR,
               linewidth=cfg.AVWAP_LINE_WIDTH, alpha=0.9)
    
    # ===== AVWAP ANCHOR MARKER =====
    avwap_anchor_idx = df.attrs.get('avwap_anchor_idx')
    avwap_anchor_price = df.attrs.get('avwap_anchor_price')
    if avwap_anchor_idx is not None and avwap_anchor_price is not None:
        ax.scatter(avwap_anchor_idx, avwap_anchor_price, marker='o', s=100,
                   color=cfg.AVWAP_COLOR, edgecolors='white', linewidths=1.5, zorder=6)
        ax.annotate('AVWAP', xy=(avwap_anchor_idx, avwap_anchor_price),
                   xytext=(avwap_anchor_idx + 1, avwap_anchor_price * 1.01),
                   fontsize=10, fontweight='bold', color=cfg.AVWAP_COLOR,
                   ha='left', va='bottom')
    
    # S/R Zones
    zone_height = df['ATR'].iloc[-1] if pd.notna(df['ATR'].iloc[-1]) else price_range * 0.02
    
    for sup_level in events.get('support_zones', []):
        rect = Rectangle((0, sup_level - zone_height/2), len(df), zone_height,
                         facecolor=cfg.SUPPORT_COLOR, alpha=cfg.SUPPORT_ALPHA,
                         edgecolor=cfg.SUPPORT_COLOR, linewidth=1)
        ax.add_patch(rect)
        ax.text(len(df) + 0.5, sup_level, f'S: ${sup_level:.2f}',
               fontsize=cfg.SR_LABEL_SIZE, color=cfg.SUPPORT_COLOR,
               fontweight='bold', va='center', ha='left')
    
    for res_level in events.get('resistance_zones', []):
        rect = Rectangle((0, res_level - zone_height/2), len(df), zone_height,
                         facecolor=cfg.RESISTANCE_COLOR, alpha=cfg.RESISTANCE_ALPHA,
                         edgecolor=cfg.RESISTANCE_COLOR, linewidth=1)
        ax.add_patch(rect)
        ax.text(len(df) + 0.5, res_level, f'R: ${res_level:.2f}',
               fontsize=cfg.SR_LABEL_SIZE, color=cfg.RESISTANCE_COLOR,
               fontweight='bold', va='center', ha='left')
    
    # Swing markers (small, non-signal markers)
    for date, price, bar_idx in events.get('swing_highs', []):
        if date in date_to_x:
            xi = date_to_x[date]
            ax.scatter(xi, price, marker='v', s=50, color=cfg.SWING_HIGH_COLOR,
                      edgecolors='white', linewidths=0.5, zorder=4, alpha=0.6)
    
    for date, price, bar_idx in events.get('swing_lows', []):
        if date in date_to_x:
            xi = date_to_x[date]
            ax.scatter(xi, price, marker='^', s=50, color=cfg.SWING_LOW_COLOR,
                      edgecolors='white', linewidths=0.5, zorder=4, alpha=0.6)
    
    # ===== SIGNAL DOTS: BLACK circles with WHITE numbers =====
    signals = events.get('signals', [])
    signals = _compute_signal_positions(signals, cfg)
    
    for sig in signals:
        bar_idx = sig['bar_idx']
        y_pos = sig['y_pos']
        sig_id = sig['signal_id']
        
        # Draw leader line if stacked (thin line from dot to candle top)
        if sig.get('needs_leader', False):
            ax.plot([bar_idx, bar_idx], [sig['candle_y'], y_pos],
                   color=cfg.SIGNAL_LEADER_LINE_COLOR, linewidth=cfg.SIGNAL_LEADER_LINE_WIDTH,
                   linestyle='-', alpha=0.6, zorder=6)
        
        # Draw BLACK circle with WHITE edge
        ax.scatter(bar_idx, y_pos, s=cfg.SIGNAL_DOT_SIZE, marker='o',
                   facecolor=SIGNAL_MARKER_BG, edgecolor=SIGNAL_MARKER_EDGE, 
                   linewidths=cfg.SIGNAL_DOT_EDGE_WIDTH, zorder=7)
        
        # Draw WHITE number inside
        ax.text(bar_idx, y_pos, str(sig_id), 
               fontsize=cfg.SIGNAL_ID_FONT_SIZE, fontweight='bold',
               color=SIGNAL_MARKER_TEXT, ha='center', va='center', zorder=8)
    
    # ===== SIGNAL LEGEND (black dots with numbers) =====
    legend_y = price_max + price_range * 0.025
    _draw_signal_legend_colored(ax, cfg, 0.5, legend_y)
    
    # Axis limits
    ax.set_xlim(-0.5, len(df) + 6)
    padding = price_range * 0.12  # Extra padding for signals and legend
    ax.set_ylim(price_min - padding * 0.5, price_max + padding)
    ax.set_ylabel('Price ($)', fontsize=cfg.LABEL_SIZE, color=cfg.TEXT_COLOR)


def _plot_macd_panel(ax: plt.Axes, df: pd.DataFrame, date_to_x: Dict, cfg: VLMChartConfig):
    """Plot MACD line, Signal line, and Histogram."""
    
    # Histogram bars
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(row['MACD_Hist']):
            color = cfg.MACD_HIST_UP if row['MACD_Hist'] >= 0 else cfg.MACD_HIST_DOWN
            ax.bar(i, row['MACD_Hist'], width=0.6, color=color, edgecolor='none', alpha=0.7)
    
    # MACD line
    valid_macd = df['MACD'].dropna()
    if len(valid_macd) > 0:
        x_macd = [date_to_x[d] for d in valid_macd.index if d in date_to_x]
        ax.plot(x_macd, valid_macd.values, color=cfg.MACD_LINE_COLOR,
               linewidth=cfg.MACD_LINE_WIDTH)
    
    # Signal line
    valid_signal = df['MACD_Signal'].dropna()
    if len(valid_signal) > 0:
        x_signal = [date_to_x[d] for d in valid_signal.index if d in date_to_x]
        ax.plot(x_signal, valid_signal.values, color=cfg.MACD_SIGNAL_COLOR,
               linewidth=cfg.MACD_LINE_WIDTH)
    
    # Zero line
    ax.axhline(y=0, color=cfg.TEXT_COLOR, linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_ylabel('MACD', fontsize=cfg.LABEL_SIZE, color=cfg.TEXT_COLOR)
    ax.set_xlim(-0.5, len(df) + 6)
    
    # In-panel label
    macd_max = df['MACD'].max() if pd.notna(df['MACD'].max()) else 1
    ax.text(1, macd_max * 0.9, 'MACD(12,26,9)', fontsize=cfg.LEGEND_SIZE,
           color=cfg.TEXT_COLOR, ha='left', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=cfg.GRID_COLOR, alpha=0.85))


def _plot_rsi_panel(ax: plt.Axes, df: pd.DataFrame, date_to_x: Dict, cfg: VLMChartConfig):
    """Plot RSI with 30/50/70 lines."""
    
    valid_rsi = df['RSI'].dropna()
    if len(valid_rsi) > 0:
        x_rsi = [date_to_x[d] for d in valid_rsi.index if d in date_to_x]
        ax.plot(x_rsi, valid_rsi.values, color=cfg.RSI_COLOR,
               linewidth=cfg.RSI_LINE_WIDTH)
        
        # Fill oversold/overbought
        ax.fill_between(x_rsi, valid_rsi.values, 30,
                       where=(valid_rsi.values < 30),
                       color=cfg.SUPPORT_COLOR, alpha=0.3)
        ax.fill_between(x_rsi, valid_rsi.values, 70,
                       where=(valid_rsi.values > 70),
                       color=cfg.RESISTANCE_COLOR, alpha=0.3)
    
    # Reference lines
    ax.axhline(y=30, color=cfg.SUPPORT_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=50, color=cfg.TEXT_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=70, color=cfg.RESISTANCE_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels on right
    ax.text(len(df) + 0.5, 30, '30', fontsize=cfg.TICK_SIZE, color=cfg.SUPPORT_COLOR, va='center')
    ax.text(len(df) + 0.5, 50, '50', fontsize=cfg.TICK_SIZE, color=cfg.TEXT_COLOR, va='center')
    ax.text(len(df) + 0.5, 70, '70', fontsize=cfg.TICK_SIZE, color=cfg.RESISTANCE_COLOR, va='center')
    
    ax.set_ylabel('RSI', fontsize=cfg.LABEL_SIZE, color=cfg.TEXT_COLOR)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(df) + 6)
    
    # In-panel label
    ax.text(1, 90, 'RSI(14)', fontsize=cfg.LEGEND_SIZE,
           color=cfg.TEXT_COLOR, ha='left', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=cfg.GRID_COLOR, alpha=0.85))


def _plot_stoch_panel(ax: plt.Axes, df: pd.DataFrame, date_to_x: Dict, cfg: VLMChartConfig):
    """Plot simplified Stochastic: FastK + SlowD only."""
    
    # Fast %K
    valid_fk = df['Fast_K'].dropna()
    if len(valid_fk) > 0:
        x_fk = [date_to_x[d] for d in valid_fk.index if d in date_to_x]
        ax.plot(x_fk, valid_fk.values, color=cfg.FAST_K_COLOR,
               linewidth=cfg.STOCH_LINE_WIDTH, alpha=0.9)
    
    # Slow %D
    valid_sd = df['Slow_D'].dropna()
    if len(valid_sd) > 0:
        x_sd = [date_to_x[d] for d in valid_sd.index if d in date_to_x]
        ax.plot(x_sd, valid_sd.values, color=cfg.SLOW_D_COLOR,
               linewidth=cfg.STOCH_LINE_WIDTH, alpha=0.9)
    
    # Reference lines
    ax.axhline(y=20, color=cfg.SUPPORT_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=80, color=cfg.RESISTANCE_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    
    # Fill zones
    ax.axhspan(0, 20, color=cfg.SUPPORT_COLOR, alpha=0.1)
    ax.axhspan(80, 100, color=cfg.RESISTANCE_COLOR, alpha=0.1)
    
    # Labels on right
    ax.text(len(df) + 0.5, 20, '20', fontsize=cfg.TICK_SIZE, color=cfg.SUPPORT_COLOR, va='center')
    ax.text(len(df) + 0.5, 80, '80', fontsize=cfg.TICK_SIZE, color=cfg.RESISTANCE_COLOR, va='center')
    
    # In-panel label
    ax.text(1, 90, 'STOCH: FastK & SlowD', fontsize=cfg.LEGEND_SIZE,
           color=cfg.TEXT_COLOR, ha='left', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=cfg.GRID_COLOR, alpha=0.85))
    
    ax.set_ylabel('STOCH', fontsize=cfg.LABEL_SIZE, color=cfg.TEXT_COLOR)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(df) + 6)


class VLMChartGenerator:
    """Main class for generating VLM-optimized charts."""
    
    def __init__(self, config: VLMChartConfig = None):
        self.config = config or VLMChartConfig()
    
    def generate_chart(
        self,
        symbol: str,
        days: int = VLMChartConfig.DEFAULT_DAYS,
        sr_data: Optional[Dict] = None,
        force_regenerate: bool = False,
        output_prefix: str = ""
    ) -> Optional[Path]:
        """Generate a VLM-optimized 4-panel chart."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None
        
        today = datetime.now().strftime('%Y%m%d')
        prefix = f"{output_prefix}_" if output_prefix else ""
        chart_path = CHARTS_DIR / f"{prefix}{symbol}_{today}_VLM.png"
        
        if chart_path.exists() and not force_regenerate:
            logger.debug(f"Using cached: {chart_path}")
            return chart_path
        
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f'{days + 60}d')
            
            if len(hist) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(hist)} bars")
                return None
            
            hist = hist.tail(days)
            df = compute_indicators(hist)
            events = detect_events(df, sr_data)
            success = plot_chart(df, symbol, events, chart_path, sr_data)
            
            if success:
                logger.info(f"Generated: {chart_path}")
                return chart_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate chart for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_intraday_chart(
        self,
        symbol: str,
        interval: str = "1h",
        period: str = "5d",
        sr_data: Optional[Dict] = None,
        force_regenerate: bool = False,
        output_prefix: str = "",
    ) -> Optional[Path]:
        """
        Generate an intraday VLM chart using the same 4-panel design system.

        Existing daily chart path remains unchanged; this is additive.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None

        today = datetime.now().strftime('%Y%m%d')
        interval_tag = interval.replace("m", "M").replace("h", "H")
        prefix = f"{output_prefix}_" if output_prefix else ""
        chart_path = CHARTS_DIR / f"{prefix}{symbol}_{today}_{interval_tag}_INTRADAY_VLM.png"

        if chart_path.exists() and not force_regenerate:
            logger.debug(f"Using cached intraday chart: {chart_path}")
            return chart_path

        try:
            raw = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if raw is None or raw.empty:
                stock = yf.Ticker(symbol)
                raw = stock.history(period=period, interval=interval)

            if raw is None or raw.empty:
                logger.warning(f"No intraday data for {symbol} ({period}/{interval})")
                return None

            # Flatten possible multi-index columns from yfinance download.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [str(c[0]) for c in raw.columns]
            raw = raw.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            for required in ("Open", "High", "Low", "Close", "Volume"):
                if required not in raw.columns:
                    logger.warning(f"Missing required intraday column '{required}' for {symbol}")
                    return None

            raw = raw.tail(140).copy()
            interval_l = str(interval or "").lower()
            min_bars = 24 if interval_l in {"1h", "60m", "90m"} else 40
            if len(raw) < min_bars:
                logger.warning(
                    f"Insufficient intraday bars for {symbol}: "
                    f"{len(raw)} (min={min_bars}, interval={interval})"
                )
                return None

            df = compute_indicators(raw)
            events = detect_events(df, sr_data)

            timeframe_label = "1H" if interval.lower() == "1h" else interval.upper()
            success = plot_intraday_chart(
                df=df,
                symbol=symbol,
                events=events,
                out_path=chart_path,
                timeframe_label=timeframe_label,
                sr_data=sr_data,
            )
            if success:
                logger.info(f"Generated intraday chart: {chart_path}")
                return chart_path
            return None
        except Exception as e:
            logger.error(f"Failed to generate intraday chart for {symbol}: {e}")
            return None


_vlm_generator: Optional[VLMChartGenerator] = None


def get_vlm_generator() -> VLMChartGenerator:
    """Get singleton generator."""
    global _vlm_generator
    if _vlm_generator is None:
        _vlm_generator = VLMChartGenerator()
    return _vlm_generator


def generate_vlm_chart(
    symbol: str,
    days: int = 60,
    sr_data: Optional[Dict] = None,
    force: bool = False,
    prefix: str = ""
) -> Optional[Path]:
    """Convenience function."""
    return get_vlm_generator().generate_chart(
        symbol, days=days, sr_data=sr_data,
        force_regenerate=force, output_prefix=prefix
    )


def generate_vlm_intraday_chart(
    symbol: str,
    interval: str = "1h",
    period: str = "5d",
    sr_data: Optional[Dict] = None,
    force: bool = False,
    prefix: str = "",
) -> Optional[Path]:
    """Convenience wrapper for intraday VLM chart generation."""
    return get_vlm_generator().generate_intraday_chart(
        symbol=symbol,
        interval=interval,
        period=period,
        sr_data=sr_data,
        force_regenerate=force,
        output_prefix=prefix,
    )


if __name__ == "__main__":
    import sys
    import random
    
    TEST_SYMBOLS = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--random':
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            symbols = random.sample(TEST_SYMBOLS, min(count, len(TEST_SYMBOLS)))
        else:
            symbols = [s.upper() for s in sys.argv[1:]]
    else:
        symbols = ['AAPL']
    
    print(f"Generating VLM charts (unified SIGNAL_STYLE): {symbols}")
    print("Signal dots: colored circles with ID inside, black edge, leader lines for stacking")
    
    generator = VLMChartGenerator()
    
    for symbol in symbols:
        print(f"\n  {symbol}...", end=" ")
        path = generator.generate_chart(symbol, force_regenerate=True, output_prefix="VL_TEST")
        if path:
            print(f"✓ {path.name}")
        else:
            print("✗ Failed")
    
    print(f"\nDone! Charts in: {CHARTS_DIR}")
