# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Renderer for the goodput timeline."""

import bisect
import collections
import datetime
import html as html_lib
from typing import Any, Callable, NamedTuple, Optional, Sequence, TypeVar, Union

from cloud_goodput.ml_goodput_measurement.timeline import builder
from cloud_goodput.ml_goodput_measurement.timeline import event_parser
from cloud_goodput.ml_goodput_measurement.timeline import events
from cloud_goodput.ml_goodput_measurement.timeline import metrics
from cloud_goodput.ml_goodput_measurement.timeline import step_times
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_FALLBACK_COLOR = '#95a5a6'

_OVERVIEW_KIND_LABELS: dict[events.EventType, str] = {
    events.EventType.STEP_BLOCK: 'Stepping',
    events.EventType.WASTED_PROGRESS: 'Stepping',
    events.EventType.STARTUP: 'Startup / Recovery',
    events.EventType.INTERRUPTION: 'Interruption',
}
_OVERVIEW_LEGEND_LABELS: dict[events.EventType, str] = {
    events.EventType.STEP_BLOCK: (
        'Stepping (Productive Steps + JIT compile overhead)'
    ),
    events.EventType.PRODUCTIVE_STEPS: 'Productive Steps (excl. JIT compile)',
    events.EventType.WASTED_PROGRESS: (
        'Wasted Progress (Lost + JIT compile during wasted progress)'
    ),
    events.EventType.LOST_PROGRESS: (
        'Lost Progress (Repeated Steps, excl. JIT compile)'
    ),
    events.EventType.STARTUP: 'Startup / Recovery',
    events.EventType.INTERRUPTION: 'Interruption',
    events.EventType.TPU_INIT: 'Device Init (TPU)',
    events.EventType.TRAINING_PREP: 'Training Prep',
    events.EventType.CHECKPOINT_RESTORE: 'Checkpoint / Snapshot Restore',
    events.EventType.SNAPSHOT_SAVE: 'Snapshot Save',
    events.EventType.AOT_COMPILE: 'AOT Compilation',
    events.EventType.DATA_LOADING: 'Data Loading',
    events.EventType.JIT_COMPILE: 'JIT Compile Overhead',
    events.EventType.CHECKPOINT_SAVE: 'Checkpoint Save',
    events.EventType.ELASTIC_REINIT: 'Elastic Reinitialization',
    events.EventType.ELASTIC_SCALE_UP: 'Elastic Scale Up',
    events.EventType.ELASTIC_SLICE_DOWN: 'Elastic Slice Down',
    events.EventType.CUSTOM: 'Custom Badput',
}
_OVERVIEW_STARTUP_SUBEVENTS = {
    events.EventType.ELASTIC_REINIT,
    events.EventType.ELASTIC_SCALE_UP,
    events.EventType.ELASTIC_SLICE_DOWN,
    events.EventType.TRAINING_PREP,
    events.EventType.TPU_INIT,
    events.EventType.CHECKPOINT_RESTORE,
    events.EventType.SNAPSHOT_SAVE,
    events.EventType.AOT_COMPILE,
    events.EventType.DATA_LOADING,
    events.EventType.CUSTOM,
}
_OVERVIEW_STEPPING_SUBEVENTS = {
    events.EventType.PRODUCTIVE_STEPS,
    events.EventType.LOST_PROGRESS,
    events.EventType.JIT_COMPILE,
    events.EventType.CHECKPOINT_SAVE,
    events.EventType.CUSTOM,
}
# Z-order priority from bottom (0 = holding containers) to top (small nested
# sub-events). Productive Steps and Lost Progress are the parts of their
# containers outside the JIT Compile Overhead, so they sit just below it.
# Checkpoint Save can happen while steps run, so it is drawn above all three.
_Z_ORDER_PRIORITY: dict[events.EventType, int] = {
    events.EventType.INTERRUPTION: 0,
    events.EventType.STARTUP: 1,
    events.EventType.STEP_BLOCK: 2,
    events.EventType.WASTED_PROGRESS: 3,
    events.EventType.ELASTIC_REINIT: 4,
    events.EventType.ELASTIC_SCALE_UP: 5,
    events.EventType.ELASTIC_SLICE_DOWN: 6,
    events.EventType.TRAINING_PREP: 7,
    events.EventType.TPU_INIT: 8,
    events.EventType.CHECKPOINT_RESTORE: 9,
    events.EventType.SNAPSHOT_SAVE: 9,
    events.EventType.AOT_COMPILE: 10,
    events.EventType.PRODUCTIVE_STEPS: 11,
    events.EventType.LOST_PROGRESS: 12,
    events.EventType.JIT_COMPILE: 13,
    events.EventType.DATA_LOADING: 14,
    events.EventType.CHECKPOINT_SAVE: 15,
    events.EventType.CUSTOM: 16,
}
_PHASE_COLORS: dict[str, str] = {
    'fault_detection_teardown': '#dc2626',
    'slice_wait_reschedule': '#ef4444',
    'pathways_cluster_init': '#06b6d4',
    'tpu_mesh_model_init': '#f97316',
    'data_iterator_init': '#6366f1',
    'checkpoint_restore': '#14b8a6',
    'aot_xla_compilation': '#8b5cf6',
    'first_batch_data_loading': '#ec4899',
    'jit_compile_overhead': '#d946ef',
    'lost_progress': '#f59e0b',
}
_MID_TIER_SUBEVENTS = {
    events.EventType.ELASTIC_REINIT,
    events.EventType.TRAINING_PREP,
    events.EventType.ELASTIC_SCALE_UP,
    events.EventType.ELASTIC_SLICE_DOWN,
}
_OVERVIEW_ROW_ORDER = ['Stepping', 'Startup / Recovery', 'Interruption']

# Overview figure geometry in pixels. The Gantt rows, legend, range selector,
# range slider and title keep the sizes and offsets they had before the panels
# below the rows were added (a 465px figure with 150px top and 45px bottom).
# The top margin fits two rows of legend; _FIT_LEGEND_JS makes room for more.
_OVERVIEW_TOP_PX = 150
_OVERVIEW_BOTTOM_PX = 45
_OVERVIEW_GANTT_PX = 270
_OVERVIEW_STEP_TIME_PX = 150
_OVERVIEW_SLICES_PX = 115
_OVERVIEW_PANEL_GAP_PX = 36
_OVERVIEW_LEGEND_GAP_PX = 65
_OVERVIEW_RANGESELECTOR_GAP_PX = 8
_OVERVIEW_RANGESLIDER_PX = 19
_OVERVIEW_TITLE_TOP_PX = 14
# Per-run timeline geometry in pixels: a 100px plot, with the "Event Type"
# legend 15px above it and the title 12px below the top of the figure.
_RUN_TOP_PX = 95
_RUN_BOTTOM_PX = 45
_RUN_PLOT_PX = 100
_RUN_LEGEND_GAP_PX = 15
_RUN_TITLE_TOP_PX = 12
# When the "Event Type" legend of the overview or of a run's timeline wraps
# into more rows than the top margin fits, _FIT_LEGEND_JS makes the margin and
# the figure taller, so that the legend stays at least this far below the
# title.
_FIT_LEGEND_TITLE_GAP_PX = 8
# Step-time panel: steps slower than this multiple of their expected step time
# (the baseline, or with fewer active elastic slices the step time measured
# with them; see step_times.Baseline) are slow (the builder uses the same
# threshold for JIT compile overhead), and the y-axis stops at
# max(2.5 x baseline, 1.1 x p95) so a few very slow steps do not flatten the
# rest.
_SLOW_STEP_FACTOR = 1.15
_STEP_TIME_RANGE_FACTOR = 2.5
_STEP_TIME_RANGE_PERCENTILE = 0.95
_STEP_LINE_COLOR = '#0284c7'
# Slow steps during a checkpoint save or a JIT compile overhead segment get the
# Gantt color of that sub-event; only the other slow steps are red.
_SLOW_STEP_COLOR = '#e11d48'
_CKPT_STEP_COLOR = events.EVENT_STYLE_MAP[
    events.EventType.CHECKPOINT_SAVE
].color
_JIT_STEP_COLOR = events.EVENT_STYLE_MAP[events.EventType.JIT_COMPILE].color
_BASELINE_COLOR = '#334155'
_BASELINE_BAND_COLOR = 'rgba(16, 185, 129, 0.10)'
# The overview's plot background; the step-time panel paints it over the
# normal range band where the expected step time is below the baseline.
_OVERVIEW_PLOT_BG_COLOR = '#f8fafc'
_SLICE_COLORS = {
    'active': '#059669',
    'available': '#2563eb',
    'total': '#64748b',
}

_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      padding: 0 1.5rem 3.5rem;
      color: #0f172a;
      background-color: #f8fafc;
      line-height: 1.45;
    }}
    .page-container {{ max-width: 1640px; margin: 0 auto; }}
    .page-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 1.25rem 0 0.85rem;
    }}
    h1 {{
      font-size: 1.35rem; font-weight: 700; color: #0f172a;
      letter-spacing: -0.02em;
    }}
    .subtitle-badge {{
      font-size: 0.78rem; font-weight: 600; color: #475569;
      background: #e2e8f0; padding: 0.25rem 0.65rem; border-radius: 9999px;
    }}
    nav {{
      position: sticky; top: 0;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 0.5rem 0.75rem;
      margin-bottom: 1.25rem;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
      z-index: 100; display: flex; flex-wrap: wrap; gap: 0.35rem;
      align-items: center;
      max-height: 140px; overflow-y: auto;
    }}
    nav .label {{
      font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; color: #64748b; margin-right: 0.35rem;
      white-space: nowrap;
    }}
    nav a {{
      font-size: 0.75rem; font-weight: 500; color: #334155;
      background: #f1f5f9; border: 1px solid #e2e8f0;
      text-decoration: none; padding: 0.22rem 0.6rem;
      border-radius: 6px; white-space: nowrap;
      transition: all 0.15s ease;
    }}
    nav a:hover {{
      background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe;
    }}
    nav a.ov {{
      font-weight: 600; color: #1d4ed8; background: #eff6ff;
      border: 1px solid #93c5fd;
    }}
    section.section-card {{
      margin-top: 1.5rem; scroll-margin-top: 4.5rem;
      background: #ffffff; border: 1px solid #e2e8f0;
      border-radius: 12px; padding: 1.15rem 1.35rem;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
    }}
    .section-header {{
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 0.5rem;
      padding-bottom: 0.65rem; margin-bottom: 0.5rem;
      border-bottom: 1px solid #f1f5f9;
    }}
    h2 {{
      font-size: 1.02rem; font-weight: 700; color: #0f172a;
      display: flex; align-items: center; gap: 0.5rem;
    }}
    .run-meta-badge {{
      font-size: 0.76rem; font-weight: 500; color: #475569;
      background: #f8fafc; border: 1px solid #e2e8f0;
      padding: 0.2rem 0.6rem; border-radius: 6px;
    }}
    a.back {{
      font-size: 0.75rem; font-weight: 500; color: #64748b;
      background: #f8fafc; border: 1px solid #e2e8f0;
      padding: 0.2rem 0.55rem; border-radius: 6px;
      text-decoration: none; transition: all 0.15s ease;
    }}
    a.back:hover {{ color: #1d4ed8; background: #eff6ff; border-color: #bfdbfe; }}
    .chart-divider {{
      border-top: 1px dashed #e2e8f0; margin: 0.6rem 0;
    }}
  </style>
</head>
<body>
  <div class="page-container">
    <div class="page-header">
      <h1>Goodput Events Timeline &mdash; {workload_id}</h1>
      <span class="subtitle-badge">Interactive Workload &amp; MTTR Diagnostics</span>
    </div>
    {body}
  </div>
</body>
</html>
"""

# Passed as `post_script` to `Figure.to_html` for a figure whose layout.meta
# has `fit_legend` (see build_overview_figure and build_figure). The horizontal
# legend sits above the plot and wraps into more rows as the page gets
# narrower, growing upward into the title. After every redraw, including the
# ones on window resizes, this moves the top of the plot down (or back up, but
# not above `min_top_px`) by making the top margin and the figure taller by the
# same amount, so the plot keeps its height, and keeps the title `title_top_px`
# below the top.
_FIT_LEGEND_JS = """\
(function() {
  var gd = document.getElementById('{plot_id}');
  var fit = gd.layout.meta && gd.layout.meta.fit_legend;
  if (!fit) {
    return;
  }
  function fitLegend() {
    var title = gd.querySelector('.gtitle');
    var legend = gd.querySelector('.legend');
    if (!title || !legend) {
      return;
    }
    var shift = Math.ceil(
        title.getBoundingClientRect().bottom + fit.title_gap_px -
        legend.getBoundingClientRect().top);
    var top = gd.layout.margin.t;
    var newTop = Math.max(fit.min_top_px, top + shift);
    // Ignore a 1px shrink, which could flip back and forth with rounding.
    if (newTop === top || shift === -1) {
      return;
    }
    var height = gd.layout.height + newTop - top;
    // to_html gives the div the initial height; it must grow with the figure,
    // or the figure would spill over what comes after it on the page.
    gd.style.height = height + 'px';
    Plotly.relayout(gd, {
      'margin.t': newTop,
      'height': height,
      'title.y': 1 - fit.title_top_px / height,
    });
  }
  gd.on('plotly_afterplot', fitLegend);
  fitLegend();
})();
"""


def _to_dt(unix: float) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(unix, tz=datetime.timezone.utc)


def _fmt_dt(unix: float) -> str:
  return _to_dt(unix).strftime('%Y-%m-%d %H:%M:%S') + ' UTC'


def _fmt_duration(seconds: float) -> str:
  if seconds < 60:
    return f'{seconds:.1f}s'
  if seconds < 3600:
    return f'{seconds / 60:.1f}m'
  return f'{seconds / 3600:.2f}h'


def _run_label(run_idx: int) -> str:
  return 'Startup' if run_idx == 0 else f'Recovery #{run_idx}'


def _style_tier_for_kind(
    kind: Optional[events.EventType],
) -> tuple[float, int, str, float]:
  """Returns (bar_width, cornerradius, border_color, border_width)."""
  if kind is None or kind in _OVERVIEW_KIND_LABELS:
    if kind == events.EventType.WASTED_PROGRESS:
      return (0.68, 6, 'rgba(217, 119, 6, 0.4)', 1.2)
    if kind == events.EventType.STEP_BLOCK:
      return (0.68, 6, 'rgba(5, 150, 105, 0.4)', 1.2)
    return (0.68, 6, 'rgba(15, 23, 42, 0.14)', 1.0)
  if kind in _MID_TIER_SUBEVENTS:
    return (0.48, 4, '#ffffff', 1.4)
  if kind == events.EventType.CHECKPOINT_SAVE:
    return (0.48, 3, '#ffffff', 1.6)
  return (0.42, 3, '#ffffff', 1.4)


def _fix_legend(
    fig: go.Figure,
    kind_label_map: Optional[dict[events.EventType, str]] = None,
) -> None:
  """Deduplicates figure legend and applies nested bar styling."""
  seen: set[events.EventType] = set()
  for trace in fig.data:
    trace_color = getattr(trace.marker, 'color', None)
    matched = next(
        (
            k
            for k, s in events.EVENT_STYLE_MAP.items()
            if s.color == trace_color
        ),
        None,
    )
    bar_w, radius, b_color, b_width = _style_tier_for_kind(matched)
    trace.width = bar_w
    trace.marker.cornerradius = radius
    trace.marker.line = dict(color=b_color, width=b_width)

    label = (
        kind_label_map[matched]
        if (kind_label_map and matched and matched in kind_label_map)
        else (events.EVENT_STYLE_MAP[matched].row_label if matched else None)
    )
    if matched and matched not in seen and label:
      trace.name = label
      trace.showlegend = True
      seen.add(matched)
    else:
      trace.showlegend = False


def _add_topmost_hover_layer(fig: go.Figure, df: pd.DataFrame) -> None:
  """Makes the hover tooltip match the bar that is visually on top.

  With overlaid bars, plotly picks the hovered bar with a heuristic that
  favours bars whose end is close to the cursor, so a wide container bar
  (e.g. Wasted Progress) can win over the sub-event drawn on top of it (e.g.
  Checkpoint Save or Lost Progress). To make hover deterministic, all drawn
  bars are made non-hoverable and a transparent hover-only layer is added.
  The layer is cut at every bar boundary, so at any time point all candidate
  hover bars share the same extent and plotly only discriminates on bar
  thickness: the thinnest bar under the cursor wins. For each elementary
  interval it holds the topmost bar plus every lower bar that is thicker than
  all bars above it (reachable only by hovering its visible outer edge).

  Args:
    fig: Figure built by px.timeline from `df`, after `_fix_legend`.
    df: Records with Task, Start, Finish, Color, Hover and ZOrder columns, in
      draw order.
  """
  color_to_kind = {s.color: k for k, s in events.EVENT_STYLE_MAP.items()}
  fig.update_traces(hoverinfo='skip', hovertemplate=None)

  ys, bases, durs_ms, widths, hovers = [], [], [], [], []
  for task, grp in df.groupby('Task', sort=False):
    bars = [
        (
            r.Start,
            r.Finish,
            r.ZOrder,
            draw_idx,
            _style_tier_for_kind(color_to_kind.get(r.Color))[0],
            r.Hover,
        )
        for draw_idx, r in enumerate(grp.itertuples(index=False))
        if r.Finish > r.Start
    ]
    bounds = sorted({b[0] for b in bars} | {b[1] for b in bars})
    for iv_start, iv_end in zip(bounds, bounds[1:]):
      covering = [b for b in bars if b[0] <= iv_start and b[1] >= iv_end]
      # Topmost first: higher z-order, then later in draw order.
      covering.sort(key=lambda b: (b[2], b[3]), reverse=True)
      max_width = 0.0
      for _, _, _, _, width, hover in covering:
        if width > max_width + 1e-9:
          ys.append(task)
          bases.append(iv_start)
          durs_ms.append((iv_end - iv_start).total_seconds() * 1000.0)
          widths.append(width)
          hovers.append(hover)
          max_width = width

  if not ys:
    return
  fig.add_trace(
      go.Bar(
          y=ys,
          base=bases,
          x=durs_ms,
          width=widths,
          orientation='h',
          marker=dict(color='rgba(0,0,0,0)', line=dict(width=0)),
          customdata=hovers,
          hovertemplate='%{customdata}<extra></extra>',
          showlegend=False,
          name='',
      )
  )


def _hover(
    seg: builder.Segment, extra_lines: str = '', title_override: str = ''
) -> str:
  header = title_override or seg.label
  return (
      f'<b>{header}</b><br><span style="color:#cbd5e1">Start:</span>   '
      f' {_fmt_dt(seg.start)}<br><span style="color:#cbd5e1">End:</span>     '
      f' {_fmt_dt(seg.end)}<br><span style="color:#cbd5e1">Duration:</span>'
      f' <b>{_fmt_duration(seg.duration_s)}</b> ({seg.duration_s:.1f}s)'
      + extra_lines
  )


def _get_sub_event_details(
    overview_seg: builder.Segment,
    all_segments: list[builder.Segment],
    rec_stats: Optional[metrics.RecoverySummaryStats] = None,
) -> str:
  """Builds segment-specific hover details for Workload Overview bars."""
  lines = []
  if 'step_range' in overview_seg.meta:
    s0, s1 = overview_seg.meta['step_range']
    count = int(overview_seg.meta.get('count') or (s1 - s0 + 1))
    lines.append(f'Steps: {s0}\u2013{s1} ({count} steps)')
    if count > 0 and overview_seg.duration_s > 0:
      lines.append(
          f'Avg step time: {overview_seg.duration_s / count:.2f}s/step'
      )

  if overview_seg.kind == events.EventType.INTERRUPTION:
    lines.append('<b>Detection, Teardown, Infra Recovery</b>')
    if (
        'last_step' in overview_seg.meta
        and overview_seg.meta['last_step'] is not None
    ):
      lines.append(
          f'Last step before disruption: {overview_seg.meta["last_step"]}'
      )
    if rec_stats and rec_stats.incidents:
      target_inc = next(
          (
              inc
              for inc in rec_stats.incidents
              if inc.incident_idx == overview_seg.run_idx + 1
          ),
          None,
      )
      if target_inc is not None and target_inc.rolled_back_steps > 0:
        lines.append(
            'Lost progress on restart:'
            f' {target_inc.rolled_back_steps} repeated steps'
            f' ({_fmt_duration(target_inc.lost_progress_s)})'
        )
    return ('<br>' + '<br>'.join(lines)) if lines else ''

  if overview_seg.kind == events.EventType.STARTUP:
    sub_segs = []
    for s in all_segments:
      if s.kind in _OVERVIEW_KIND_LABELS:
        continue
      if s.run_idx != overview_seg.run_idx:
        continue
      if s.kind in _OVERVIEW_STARTUP_SUBEVENTS:
        if max(s.start, overview_seg.start) < min(s.end, overview_seg.end):
          sub_segs.append(s)

    if sub_segs:
      grouped = collections.defaultdict(float)
      for s in sub_segs:
        grouped[s.label] += s.duration_s

      lines.append('<br><b>Recorded Sub-events:</b>')
      for label, dur in sorted(
          grouped.items(), key=lambda x: x[1], reverse=True
      ):
        lines.append(f'- {label}: {_fmt_duration(dur)}')

  return ('<br>' + '<br>'.join(lines)) if lines else ''


def _stepping_runs(
    steps: list[tuple[int, float]],
    segments: list[builder.Segment],
) -> list[tuple[int, list[step_times.StepDelta]]]:
  """Returns the step times of every run as (run_idx, deltas), in time order.

  Only steps inside a run's stepping window count, so the gap from the last
  step before an interruption to the first step after it is not a step time.
  Runs without two consecutive steps are left out.

  Args:
    steps: (step number, start time) pairs, by time.
    segments: The workload timeline, see builder.build_timeline.
  """
  runs = []
  for run_idx, (lo, hi) in sorted(
      metrics.stepping_windows(segments).items(), key=lambda item: item[1]
  ):
    # The window starts at the run's first step and ends after its last one.
    deltas = step_times.consecutive_step_deltas(
        [p for p in steps if lo <= p[1] < hi]
    )
    if deltas:
      runs.append((run_idx, deltas))
  return runs


def _panel_legend(fig: go.Figure, text: str, yaxis: str) -> None:
  """Adds a small legend right above the top-right corner of a panel.

  It sits in the gap above the panel (_OVERVIEW_PANEL_GAP_PX), so it does not
  cover the panel's points.

  Args:
    fig: The overview figure.
    text: The legend, one line of HTML.
    yaxis: The panel's y-axis id, e.g. 'y2'.
  """
  fig.add_annotation(
      text=text,
      xref='x domain',
      x=1,
      xanchor='right',
      yref=f'{yaxis} domain',
      y=1,
      yanchor='bottom',
      yshift=2,
      showarrow=False,
      align='right',
      font=dict(size=10.5, color='#334155'),
      bgcolor='rgba(255, 255, 255, 0.88)',
      bordercolor='#e2e8f0',
      borderwidth=1,
      borderpad=3,
  )


def _panel_yaxis(title: str, y_max: float) -> dict[str, Any]:
  """Layout shared by the y-axes of the panels below the Gantt rows."""
  return dict(
      title=dict(text=title, font=dict(size=11, color='#475569'), standoff=8),
      range=[0, y_max],
      fixedrange=True,
      anchor='x',
      tickfont=dict(size=10.5, color='#475569'),
      showgrid=True,
      gridcolor='#e2e8f0',
      zeroline=False,
      linecolor='#cbd5e1',
  )


def _add_step_time_panel(
    fig: go.Figure,
    runs: list[tuple[int, list[step_times.StepDelta]]],
    baseline: step_times.Baseline,
    yaxis: str,
    checkpoint_saves: Sequence[tuple[float, float]] = (),
    jit_compiles: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
  """Adds the step time vs. baseline panel; returns its y-axis layout.

  One point per step (x = step start, y = time to the next step), with a gap
  between runs. A step is slow when it takes more than _SLOW_STEP_FACTOR times
  its expected time (see step_times.Baseline): the baseline, or for a step
  with fewer active elastic slices the step time measured with them, drawn as
  a dotted line with the normal range band raised or lowered to match. A step
  with fewer active slices but no expected time (too few steps ran with them)
  is never slow. Slow steps during a checkpoint save or a JIT compile overhead
  segment get its Gantt color and a diamond or a square; the other slow steps
  are red. Steps above the chart range are drawn at the top as triangles.

  Args:
    fig: The overview figure.
    runs: The step times of every run, see _stepping_runs; not empty.
    baseline: The workload's baseline; its step time is > 0.
    yaxis: The trace y-axis id, e.g. 'y2'.
    checkpoint_saves: (start, end) of every checkpoint save.
    jit_compiles: (start, end) of every JIT compile overhead segment.

  Returns:
    The layout of the panel's y-axis.
  """
  baseline_s = baseline.step_time_s
  all_deltas = [d for _, deltas in runs for d in deltas]
  slices = [baseline.slices(d.start, d.end) for d in all_deltas]
  expected_times = [baseline.expected_s(d.start, d.end) for d in all_deltas]
  ckpts = step_times.overlapping(checkpoint_saves, all_deltas)
  jits = step_times.overlapping(jit_compiles, all_deltas)
  times = sorted(d.delta_s for d in all_deltas)
  p95 = times[int(_STEP_TIME_RANGE_PERCENTILE * (len(times) - 1))]
  slow_from = _SLOW_STEP_FACTOR * baseline_s
  max_expected = max((t for t in expected_times if t), default=0.0)
  y_max = max(
      _STEP_TIME_RANGE_FACTOR * baseline_s,
      1.1 * p95,
      # Keeps the raised band of the steps with fewer active slices in range.
      1.1 * _SLOW_STEP_FACTOR * max_expected,
  )

  xs, ys, colors, symbols, sizes, hovers = [], [], [], [], [], []
  # Consecutive steps of a run with the same fewer active slices and an
  # expected time, as [start, end, expected time].
  stretches: list[list[float]] = []
  slow_kinds = set()
  i = 0
  for run_idx, deltas in runs:
    if xs:
      # Breaks the line between runs.
      xs.append(None)
      ys.append(None)
      colors.append(_STEP_LINE_COLOR)
      symbols.append('circle')
      sizes.append(0)
      hovers.append('')
    prev_slices = None
    for d in deltas:
      counts, expected = slices[i], expected_times[i]
      ckpt, jit = ckpts[i], jits[i]
      i += 1
      fewer = counts is not None and counts[0] < counts[1]
      if not fewer:
        vs = f'{d.delta_s / baseline_s:.2f}\u00d7 baseline ({baseline_s:.2f}s)'
      elif expected is None:
        vs = (
            f'{counts[0]} of {counts[1]} slices active: no expected time'
            f' (fewer than {step_times.MIN_STEPS_FOR_EXPECTED} steps with them)'
        )
      else:
        if counts == prev_slices:
          stretches[-1][1] = d.end
        else:
          stretches.append([d.start, d.end, expected])
        vs = (
            f'{d.delta_s / expected:.2f}\u00d7 expected ({expected:.2f}s: lower'
            f' median of the steps with {counts[0]} of {counts[1]} slices'
            ' active)'
        )
      prev_slices = counts if fewer and expected is not None else None

      slow = expected is not None and d.delta_s > _SLOW_STEP_FACTOR * expected
      color, symbol, why = _STEP_LINE_COLOR, 'circle', ''
      if slow:
        reasons = []
        if ckpt:
          reasons.append(
              f'checkpoint save ({_fmt_duration(ckpt[1] - ckpt[0])}) during'
              ' the step'
          )
        if jit:
          reasons.append('JIT compile overhead after the (re)start')
        if ckpt:
          kind, color, symbol = 'ckpt', _CKPT_STEP_COLOR, 'diamond'
        elif jit:
          kind, color, symbol = 'jit', _JIT_STEP_COLOR, 'square'
        else:
          kind, color = 'other', _SLOW_STEP_COLOR
          reasons.append('no checkpoint save or JIT compile in the step')
        slow_kinds.add(kind)
        why = f'<br><i>Slow: {"; ".join(reasons)}</i>'
      clipped = d.delta_s > y_max
      note = '<br><i>Off the chart; drawn at the top</i>' if clipped else ''
      xs.append(_to_dt(d.start))
      ys.append(min(d.delta_s, y_max))
      colors.append(color)
      symbols.append('triangle-up' if clipped else symbol)
      sizes.append(7 if clipped else (5.5 if slow else 3.5))
      hovers.append(
          f'<b>Step {d.step} \u2192 {d.step + 1}</b><br>'
          '<span style="color:#cbd5e1">Step time:</span>'
          f' <b>{d.delta_s:.2f}s</b> \u00b7 {vs}{why}{note}'
          f'<br><span style="color:#cbd5e1">Start:</span> {_fmt_dt(d.start)}'
          f'<br>Run {run_idx + 1} \u2014 {_run_label(run_idx)}'
      )

  # Normal range band, across the whole panel width.
  fig.add_shape(
      type='rect',
      xref='x domain',
      x0=0,
      x1=1,
      yref=yaxis,
      y0=0,
      y1=slow_from,
      fillcolor=_BASELINE_BAND_COLOR,
      line=dict(width=0),
      layer='below',
  )
  # While fewer slices are active, the band goes up or down to the expected
  # step time; lowering it paints the plot background over the band.
  for start, end, expected in stretches:
    fig.add_shape(
        type='rect',
        xref='x',
        x0=_to_dt(start),
        x1=_to_dt(end),
        yref=yaxis,
        y0=slow_from,
        y1=_SLOW_STEP_FACTOR * expected,
        fillcolor=(
            _BASELINE_BAND_COLOR
            if expected > baseline_s
            else _OVERVIEW_PLOT_BG_COLOR
        ),
        line=dict(width=0),
        layer='below',
    )
  # Baseline line, across the whole panel width.
  fig.add_shape(
      type='line',
      xref='x domain',
      x0=0,
      x1=1,
      yref=yaxis,
      y0=baseline_s,
      y1=baseline_s,
      line=dict(color=_BASELINE_COLOR, width=1.2, dash='dash'),
      layer='below',
  )
  if stretches:
    # The expected step time while fewer slices are active.
    ex, ey = [], []
    for start, end, expected in stretches:
      ex += [_to_dt(start), _to_dt(end), None]
      ey += [expected, expected, None]
    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode='lines',
            line=dict(color=_BASELINE_COLOR, width=1.2, dash='dot'),
            hoverinfo='skip',
            yaxis=yaxis,
            name='Expected step time',
            showlegend=False,
        )
    )
  fig.add_trace(
      go.Scatter(
          x=xs,
          y=ys,
          mode='lines+markers',
          line=dict(color=_STEP_LINE_COLOR, width=1.1),
          marker=dict(
              color=colors, symbol=symbols, size=sizes, line=dict(width=0)
          ),
          customdata=hovers,
          hovertemplate='%{customdata}<extra></extra>',
          yaxis=yaxis,
          name='Step time',
          showlegend=False,
      )
  )

  vs_label = 'expected' if stretches else 'baseline'
  threshold = f'> {_SLOW_STEP_FACTOR:.2f}\u00d7 {vs_label}'
  kinds = [
      f'<span style="color:{color}">{glyph} {label}</span>'
      for kind, color, glyph, label in (
          ('ckpt', _CKPT_STEP_COLOR, '\u25c6', 'checkpoint save'),
          ('jit', _JIT_STEP_COLOR, '\u25a0', 'JIT compile'),
      )
      if kind in slow_kinds
  ]
  if kinds:
    kinds.append(f'<span style="color:{_SLOW_STEP_COLOR}">\u25cf other</span>')
    slow_legend = f'Slow ({threshold}): ' + '\u00a0\u00a0'.join(kinds)
  else:
    slow_legend = (
        f'<span style="color:{_SLOW_STEP_COLOR}">\u25cf {threshold}</span>'
    )
  legend = [
      f'<span style="color:{_STEP_LINE_COLOR}">\u25cf Step time</span>',
      slow_legend,
      f'<span style="color:{_BASELINE_COLOR}">\u254c Baseline'
      f' {baseline_s:.2f}s</span>',
  ]
  if stretches:
    legend.append(
        f'<span style="color:{_BASELINE_COLOR}">\u2504 Expected with fewer'
        ' slices</span>'
    )
  _panel_legend(fig, '\u00a0\u00a0\u00a0'.join(legend), yaxis)
  return dict(_panel_yaxis('Step time (s)', y_max), ticksuffix='s')


def _add_slices_panel(
    fig: go.Figure,
    slice_counts: list[tuple[float, int, int, int]],
    timeline_end: float,
    yaxis: str,
) -> dict[str, Any]:
  """Adds the elastic slice counts panel; returns its y-axis layout.

  Step lines of the total, available and active slice counts; each count
  holds until the next report, the last one until the end of the timeline.
  Hovering an interval shows its counts and when it started and ended.

  Args:
    fig: The overview figure.
    slice_counts: (timestamp, active, available, total) reports, by time.
    timeline_end: The end of the timeline, Unix seconds.
    yaxis: The trace y-axis id, e.g. 'y3'.

  Returns:
    The layout of the panel's y-axis.
  """
  bounds = [c[0] for c in slice_counts]
  bounds.append(max(timeline_end, bounds[-1]))
  xs = [_to_dt(t) for t in bounds]
  max_count = max(max(c[1:]) for c in slice_counts)
  y_max = max(max_count, 1) * 1.15

  # (name, column of slice_counts, color, line style, fill); active is drawn
  # last, on top of the others.
  for name, col, color, line, fill in (
      ('Total slices', 3, 'total', dict(width=1.4, dash='dot'), None),
      ('Available slices', 2, 'available', dict(width=1.6), None),
      ('Active slices', 1, 'active', dict(width=2), 'rgba(5, 150, 105, 0.12)'),
  ):
    ys = [c[col] for c in slice_counts]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys + ys[-1:],  # The last count holds until the timeline end.
            name=name,
            mode='lines',
            line_shape='hv',
            line=dict(color=_SLICE_COLORS[color], **line),
            fill='tozeroy' if fill else None,
            fillcolor=fill,
            yaxis=yaxis,
            hoverinfo='skip',
            showlegend=False,
        )
    )

  # Hover layer: a transparent bar per interval between two reports, so the
  # counts show wherever the cursor is in the panel.
  starts, widths_ms, hovers = [], [], []
  for (t0, active, avail, total), t1 in zip(slice_counts, bounds[1:]):
    if t1 <= t0:
      continue
    share = f' ({active / total * 100:.0f}% active)' if total > 0 else ''
    latest = ' \u00b7 latest report' if t1 == bounds[-1] else ''
    starts.append(_to_dt(t0))
    widths_ms.append((t1 - t0) * 1000.0)
    hovers.append(
        '<b>Elastic slices</b><br>'
        f'<span style="color:#cbd5e1">Active:</span> <b>{active}</b> \u00b7'
        f' <span style="color:#cbd5e1">Available:</span> {avail} \u00b7'
        f' <span style="color:#cbd5e1">Total:</span> {total}{share}'
        f'<br><span style="color:#cbd5e1">From:</span> {_fmt_dt(t0)}'
        f'<br><span style="color:#cbd5e1">Until:</span> {_fmt_dt(t1)}'
        f' ({_fmt_duration(t1 - t0)}){latest}'
    )
  if starts:
    fig.add_trace(
        go.Bar(
            x=starts,
            y=[y_max] * len(starts),
            width=widths_ms,
            offset=0,
            marker=dict(color='rgba(0,0,0,0)', line=dict(width=0)),
            customdata=hovers,
            hovertemplate='%{customdata}<extra></extra>',
            yaxis=yaxis,
            showlegend=False,
            name='',
        )
    )
  _panel_legend(
      fig,
      '\u00a0\u00a0\u00a0'.join(
          f'<span style="color:{_SLICE_COLORS[key]}">{glyph} {label}</span>'
          for key, glyph, label in (
              ('active', '\u25ac', 'Active'),
              ('available', '\u25ac', 'Available'),
              ('total', '\u2504', 'Total'),
          )
      ),
      yaxis,
  )
  layout = _panel_yaxis('Slices', y_max)
  if max_count <= 8:
    layout.update(tick0=0, dtick=1)
  return layout


class _Panel(NamedTuple):
  """A panel drawn below the Gantt rows of the overview figure."""

  name: str  # Listed in layout.meta['panels'].
  yaxis: str  # The y-axis id of its traces, e.g. 'y2'.
  height_px: int
  yaxis_layout: dict[str, Any]


def build_overview_figure(
    segments: list[builder.Segment],
    workload_id: str,
    entries: Optional[Union[list[dict[str, Any]], pd.DataFrame]] = None,
    rec_stats: Optional[metrics.RecoverySummaryStats] = None,
    parsed: Optional[event_parser.ParsedLogs] = None,
) -> go.Figure:
  """Builds the overview figure: the run timeline plus workload-level panels.

  Below the Gantt rows (Stepping, Startup / Recovery, Interruption), on the
  same x-axis, come the step time vs. baseline panel (when two consecutive
  steps exist) and the elastic slice counts panel (when the log has
  elastic_slice_counts, which McJAX and non-elastic workloads do not log).
  `layout.meta['panels']` lists the drawn panels.

  Args:
    segments: The workload timeline, see builder.build_timeline.
    workload_id: Shown in the title.
    entries: Goodput log payloads, or a DataFrame of LogRecords; the source of
      `rec_stats` and `parsed` when those are not given.
    rec_stats: Recovery stats for the hover texts.
    parsed: Parsed log entries for the panels.

  Returns:
    The figure.
  """
  overview_segs = [s for s in segments if s.kind in _OVERVIEW_KIND_LABELS]
  if not overview_segs:
    fig = go.Figure()
    fig.update_layout(title=f'No data for {workload_id}')
    return fig

  if rec_stats is None and entries is not None:
    rec_stats = metrics.extract_recovery_incidents(entries, segments)
  if parsed is None and entries is not None:
    parsed = event_parser.parse_entries(metrics.to_raw_entries(entries))

  records = []
  startup_windows = [
      (s.run_idx, s.start, s.end, s)
      for s in overview_segs
      if s.kind == events.EventType.STARTUP
  ]
  stepping_windows = [
      (s.run_idx, s.start, s.end, s)
      for s in overview_segs
      if s.kind
      in (events.EventType.STEP_BLOCK, events.EventType.WASTED_PROGRESS)
  ]

  # 1. Holding base segments first (bottom of the overlay z-order)
  for seg in sorted(
      overview_segs, key=lambda s: _Z_ORDER_PRIORITY.get(s.kind, 0)
  ):
    style = events.EVENT_STYLE_MAP.get(seg.kind)
    sub_details = _get_sub_event_details(seg, segments, rec_stats=rec_stats)
    run_desc = f'<br>Run: {seg.run_idx + 1} \u2014 {_run_label(seg.run_idx)}'
    records.append({
        'Task': _OVERVIEW_KIND_LABELS[seg.kind],
        'Start': _to_dt(seg.start),
        'Finish': _to_dt(seg.end),
        'Color': style.color if style else _FALLBACK_COLOR,
        'Hover': _hover(seg, f'{run_desc}{sub_details}'),
        'ZOrder': _Z_ORDER_PRIORITY.get(seg.kind, 0),
    })

  # 2. Sub-events overlaid onto 'Startup / Recovery' and 'Stepping' rows
  sub_candidates = [s for s in segments if s.kind not in _OVERVIEW_KIND_LABELS]
  sub_candidates.sort(
      key=lambda s: (_Z_ORDER_PRIORITY.get(s.kind, 50), -s.duration_s)
  )

  def _find_window(s, windows, require_overlap):
    return next(
        (
            parent_seg
            for r_idx, st, ed, parent_seg in windows
            if r_idx == s.run_idx
            and (not require_overlap or max(s.start, st) < min(s.end, ed))
        ),
        None,
    )

  for s in sub_candidates:
    style = events.EVENT_STYLE_MAP.get(s.kind)
    run_desc = f'<br>Run: {s.run_idx + 1} \u2014 {_run_label(s.run_idx)}'

    # Prefer the row whose window the sub-event overlaps in time (custom
    # events can occur during both startup and stepping).
    matching_startup = (
        _find_window(s, startup_windows, require_overlap=True)
        if s.kind in _OVERVIEW_STARTUP_SUBEVENTS
        else None
    )
    matching_step = (
        _find_window(s, stepping_windows, require_overlap=True)
        if s.kind in _OVERVIEW_STEPPING_SUBEVENTS and matching_startup is None
        else None
    )
    if matching_startup is None and matching_step is None:
      # Skip fallback for DATA_LOADING so that per-step data loading during
      # stepping does not land on the 'Startup / Recovery' row.
      if (
          s.kind != events.EventType.DATA_LOADING
          and s.kind in _OVERVIEW_STARTUP_SUBEVENTS
      ):
        matching_startup = _find_window(
            s, startup_windows, require_overlap=False
        )
      if matching_startup is None and s.kind in _OVERVIEW_STEPPING_SUBEVENTS:
        matching_step = _find_window(s, stepping_windows, require_overlap=False)

    if matching_startup is not None:
      pct_of_startup = (
          (s.duration_s / matching_startup.duration_s * 100.0)
          if matching_startup.duration_s > 0
          else 0.0
      )
      sub_extra = (
          f'{run_desc}<br>Share of {_run_label(s.run_idx)}:'
          f' {pct_of_startup:.1f}% ({_fmt_duration(s.duration_s)} /'
          f' {_fmt_duration(matching_startup.duration_s)})'
      )
      records.append({
          'Task': 'Startup / Recovery',
          'Start': _to_dt(s.start),
          'Finish': _to_dt(s.end),
          'Color': style.color if style else _FALLBACK_COLOR,
          'Hover': _hover(s, sub_extra),
          'ZOrder': _Z_ORDER_PRIORITY.get(s.kind, 50),
      })
    elif matching_step is not None:
      sub_extra = run_desc
      # Productive Steps and Lost Progress are parts of their container, so
      # their hover does not repeat it.
      if s.kind not in (
          events.EventType.PRODUCTIVE_STEPS,
          events.EventType.LOST_PROGRESS,
      ):
        sub_extra += f'<br>Occurs during: {matching_step.label}'
      records.append({
          'Task': 'Stepping',
          'Start': _to_dt(s.start),
          'Finish': _to_dt(s.end),
          'Color': style.color if style else _FALLBACK_COLOR,
          'Hover': _hover(s, sub_extra),
          'ZOrder': _Z_ORDER_PRIORITY.get(s.kind, 50),
      })

  df = pd.DataFrame(records)
  df = df.sort_values('ZOrder', kind='mergesort')
  row_order = [r for r in _OVERVIEW_ROW_ORDER if r in df['Task'].values]

  fig = px.timeline(
      df,
      x_start='Start',
      x_end='Finish',
      y='Task',
      color='Color',
      color_discrete_map={c: c for c in df['Color'].unique()},
      category_orders={'Task': row_order},
  )
  _fix_legend(fig, kind_label_map=_OVERVIEW_LEGEND_LABELS)
  _add_topmost_hover_layer(fig, df)

  # Panels below the Gantt rows, top to bottom, each on its own y-axis of the
  # shared x-axis. They go after the hover layer above.
  panels: list[_Panel] = []
  if parsed is not None:
    step_runs = _stepping_runs(parsed.steps, segments)
    if step_runs:
      # With elastic slice counts, steps with fewer active slices are compared
      # with their expected time, not with the full-capacity baseline.
      baseline = step_times.Baseline(parsed.steps, parsed.elastic_slice_counts)
      layout = _add_step_time_panel(
          fig,
          step_runs,
          baseline,
          'y2',
          checkpoint_saves=parsed.checkpoint_save,
          jit_compiles=[
              (s.start, s.end)
              for s in segments
              if s.kind == events.EventType.JIT_COMPILE
          ],
      )
      panels.append(_Panel('step_time', 'y2', _OVERVIEW_STEP_TIME_PX, layout))
    if parsed.elastic_slice_counts:
      timeline_end = max(s.end for s in overview_segs)
      layout = _add_slices_panel(
          fig, parsed.elastic_slice_counts, timeline_end, 'y3'
      )
      panels.append(_Panel('slices', 'y3', _OVERVIEW_SLICES_PX, layout))

  heights = [_OVERVIEW_GANTT_PX] + [p.height_px for p in panels]
  plot_h = sum(heights) + _OVERVIEW_PANEL_GAP_PX * len(panels)
  domains = []
  top_px = plot_h
  for h in heights:
    domains.append([(top_px - h) / plot_h, top_px / plot_h])
    top_px -= h + _OVERVIEW_PANEL_GAP_PX
  height = _OVERVIEW_TOP_PX + plot_h + _OVERVIEW_BOTTOM_PX
  # The x-axis ticks go under the bottom panel.
  bottom_yaxis = panels[-1].yaxis if panels else 'y'

  fig.update_xaxes(
      title=dict(text='Time (UTC)', font=dict(size=12, color='#475569')),
      tickfont=dict(size=11, color='#475569'),
      gridcolor='#e2e8f0',
      griddash='dot',
      linecolor='#cbd5e1',
      anchor=bottom_yaxis,
      rangeslider=dict(
          visible=True,
          thickness=_OVERVIEW_RANGESLIDER_PX / plot_h,
          bgcolor='#f1f5f9',
          bordercolor='#cbd5e1',
          borderwidth=1,
      ),
      rangeselector=dict(
          x=0,
          xanchor='left',
          y=1 + _OVERVIEW_RANGESELECTOR_GAP_PX / plot_h,
          yanchor='bottom',
          bgcolor='#f1f5f9',
          activecolor='#dbeafe',
          bordercolor='#cbd5e1',
          borderwidth=1,
          font=dict(size=11, color='#334155'),
          buttons=[
              dict(count=1, label='1h', step='hour', stepmode='backward'),
              dict(count=6, label='6h', step='hour', stepmode='backward'),
              dict(count=12, label='12h', step='hour', stepmode='backward'),
              dict(count=1, label='1d', step='day', stepmode='backward'),
              dict(step='all', label='All'),
          ],
      ),
  )
  yaxes = {
      'yaxis': dict(
          domain=domains[0],
          anchor='x',
          title=dict(text=''),
          autorange='reversed',
          fixedrange=True,
          tickfont=dict(size=12, color='#1e293b'),
          showgrid=True,
          gridcolor='#e2e8f0',
          linecolor='#cbd5e1',
      )
  }
  for p, domain in zip(panels, domains[1:]):
    yaxes[p.yaxis.replace('y', 'yaxis')] = dict(p.yaxis_layout, domain=domain)
  fig.update_layout(
      font=dict(
          family=(
              'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,'
              ' sans-serif'
          ),
          color='#0f172a',
      ),
      title=dict(
          text=f'<b>Overview \u2014 {workload_id}</b>',
          y=1 - _OVERVIEW_TITLE_TOP_PX / height,
          x=0.0,
          xanchor='left',
          yanchor='top',
          font=dict(size=14, color='#0f172a'),
      ),
      barmode='overlay',
      legend_title=dict(
          text='<b>Event Type</b>', font=dict(size=11.5, color='#334155')
      ),
      legend=dict(
          orientation='h',
          yanchor='bottom',
          y=1 + _OVERVIEW_LEGEND_GAP_PX / plot_h,
          xanchor='left',
          x=0,
          bgcolor='rgba(248, 250, 252, 0.92)',
          bordercolor='#e2e8f0',
          borderwidth=1,
          font=dict(size=11.5, color='#334155'),
      ),
      hoverlabel=dict(
          bgcolor='#0f172a',
          bordercolor='#334155',
          font=dict(
              family='Inter, system-ui, sans-serif', size=12, color='#f8fafc'
          ),
          align='left',
      ),
      margin=dict(
          t=_OVERVIEW_TOP_PX, b=_OVERVIEW_BOTTOM_PX, l=135, r=25
      ),
      paper_bgcolor='#ffffff',
      plot_bgcolor=_OVERVIEW_PLOT_BG_COLOR,
      height=height,
      meta=dict(
          panels=[p.name for p in panels],
          # Read by _FIT_LEGEND_JS.
          fit_legend=dict(
              min_top_px=_OVERVIEW_TOP_PX,
              title_top_px=_OVERVIEW_TITLE_TOP_PX,
              title_gap_px=_FIT_LEGEND_TITLE_GAP_PX,
          ),
      ),
      **yaxes,
  )
  return fig


def build_recovery_breakdown_figure(
    incident: metrics.RecoveryIncident,
    run_title: str,
) -> go.Figure:
  """Builds a Recovery Time Breakdown bar chart with Duration (s) on X-axis."""
  fig = go.Figure()

  labels_rev = []
  values_rev = []
  colors_rev = []
  border_colors_rev = []
  texts_rev = []
  hovers_rev = []

  # Add the 10 recovery sub-phases from top to bottom.
  for key in reversed(metrics.RECOVERY_PHASE_ORDER):
    raw_lbl = metrics.RECOVERY_PHASE_LABELS[key]
    val_s = float(incident.phases.get(key, 0.0))
    pct = (val_s / incident.mttr_s * 100.0) if incident.mttr_s > 0 else 0.0
    is_active = val_s > 0.05
    color = _PHASE_COLORS.get(key, '#64748b') if is_active else '#e2e8f0'
    b_color = 'rgba(15, 23, 42, 0.12)' if is_active else '#cbd5e1'
    styled_lbl = (
        f'<b>{raw_lbl}</b>'
        if is_active
        else f'<span style="color:#94a3b8">{raw_lbl}</span>'
    )
    txt = (
        f' <b>{_fmt_duration(val_s)}</b> ({val_s:.1f}s \u2022 {pct:.1f}%)'
        if is_active
        else ' <span style="color:#94a3b8">0.0s (0.0%)</span>'
    )
    hover = (
        f'<b>{raw_lbl}</b><br><span style="color:#cbd5e1">Duration:</span>'
        f' <b>{_fmt_duration(val_s)}</b> ({val_s:.2f}s)<br><span'
        ' style="color:#cbd5e1">Share of Incident MTTR:</span>'
        f' <b>{pct:.1f}%</b>'
    )
    if key == 'lost_progress' and incident.rolled_back_steps > 0:
      hover += (
          '<br><span style="color:#cbd5e1">Repeated steps:</span>'
          f' <b>{incident.rolled_back_steps}</b>'
      )
    labels_rev.append(styled_lbl)
    values_rev.append(val_s)
    colors_rev.append(color)
    border_colors_rev.append(b_color)
    texts_rev.append(txt)
    hovers_rev.append(hover)

  max_val = max(values_rev) if values_rev else 1.0
  track_max = max(1.0, max_val * 1.32)

  # Subtle background track bars for visual alignment
  fig.add_trace(
      go.Bar(
          y=labels_rev,
          x=[max(0.5, max_val * 1.02)] * len(labels_rev),
          base=0,
          orientation='h',
          width=0.64,
          marker=dict(
              color='#f1f5f9',
              cornerradius=4,
              line=dict(color='#e2e8f0', width=0.5),
          ),
          hoverinfo='skip',
          showlegend=False,
      )
  )

  fig.add_trace(
      go.Bar(
          y=labels_rev,
          x=values_rev,
          base=0,
          orientation='h',
          width=0.64,
          marker=dict(
              color=colors_rev,
              cornerradius=4,
              line=dict(color=border_colors_rev, width=1.0),
          ),
          text=texts_rev,
          textposition='outside',
          textfont=dict(size=11.5, color='#1e293b'),
          cliponaxis=False,
          customdata=hovers_rev,
          hovertemplate='%{customdata}<extra></extra>',
          showlegend=False,
      )
  )

  fig.update_xaxes(
      title=dict(
          text='Duration (seconds)', font=dict(size=12, color='#475569')
      ),
      tickfont=dict(size=11, color='#475569'),
      range=[0, track_max],
      gridcolor='#e2e8f0',
      griddash='dot',
      zeroline=True,
      zerolinecolor='#94a3b8',
      zerolinewidth=1.2,
  )
  fig.update_yaxes(
      title='',
      fixedrange=True,
      tickfont=dict(size=11.5, color='#1e293b'),
  )
  fig.update_layout(
      font=dict(
          family=(
              'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,'
              ' sans-serif'
          ),
          color='#0f172a',
      ),
      title=dict(
          text=(
              f'<b>{run_title} \u2014 Recovery Time Breakdown</b> <span'
              ' style="font-size:12px;color:#475569;font-weight:normal">(Total'
              f' MTTR: {_fmt_duration(incident.mttr_s)} /'
              f' {incident.mttr_s:.1f}s \u2022 Time to Recover:'
              f' {_fmt_duration(incident.downtime_s + incident.phases.get("jit_compile_overhead", 0.0))}'
              ' \u2022 Lost Progress:'
              f' {_fmt_duration(incident.lost_progress_s)})</span>'
          ),
          x=0.0,
          xanchor='left',
          font=dict(size=13.5, color='#0f172a'),
      ),
      barmode='overlay',
      hoverlabel=dict(
          bgcolor='#0f172a',
          bordercolor='#334155',
          font=dict(
              family='Inter, system-ui, sans-serif', size=12, color='#f8fafc'
          ),
          align='left',
      ),
      paper_bgcolor='#ffffff',
      plot_bgcolor='#f8fafc',
      margin=dict(t=48, b=42, l=270, r=110),
      height=365,
  )
  return fig


def build_figure(segments: list[builder.Segment], title: str) -> go.Figure:
  """Builds a continuous bar chart for a run with wall-clock Time (UTC)."""
  if not segments:
    fig = go.Figure()
    fig.update_layout(title=f'No data - {title}')
    return fig

  run_start = min(s.start for s in segments)
  run_end = max(s.end for s in segments)
  total_span = max(1.0, run_end - run_start)

  # Sort segments so containers draw first and sub-events draw on top.
  sorted_segs = sorted(
      segments,
      key=lambda s: (_Z_ORDER_PRIORITY.get(s.kind, 50), -s.duration_s, s.start),
  )

  # Containers partition the run; sub-events are capped at the end of the
  # container they sit in, and per-step Data Loading (inside a stepping
  # container) is not drawn.
  containers = sorted(
      (s for s in segments if s.kind in _OVERVIEW_KIND_LABELS),
      key=lambda s: s.start,
  )
  container_starts = [c.start for c in containers]

  def _container_of(seg: builder.Segment) -> Optional[builder.Segment]:
    i = bisect.bisect_right(container_starts, seg.start) - 1
    if i >= 0 and seg.start < containers[i].end + 1e-3:
      return containers[i]
    return None

  per_run_legend_map = dict(_OVERVIEW_LEGEND_LABELS)
  per_run_legend_map[events.EventType.INTERRUPTION] = 'Infra Recovery'

  records = []
  for seg in sorted_segs:
    style = events.EVENT_STYLE_MAP.get(seg.kind)
    color = style.color if style else _FALLBACK_COLOR
    is_interruption = seg.kind == events.EventType.INTERRUPTION
    display_title = 'Infra Recovery' if is_interruption else seg.label

    true_dur = max(0.0, seg.duration_s)
    # Ensure short sub-events remain clickable over wide holding bars.
    is_sub = seg.kind not in _OVERVIEW_KIND_LABELS
    vis_dur = (
        max(true_dur, total_span * 0.004) if is_sub else max(true_dur, 0.05)
    )
    container = _container_of(seg) if is_sub else None
    if seg.kind == events.EventType.DATA_LOADING and (
        container is None or container.kind != events.EventType.STARTUP
    ):
      # Per-step data loading is asynchronous and fully overlapped with the
      # step, so it is not drawn (the overview skips it as well).
      continue

    extra = ''
    if is_interruption:
      extra += (
          '<br><b style="color:#fca5a5">Detection, Teardown, Infra Recovery</b>'
      )
    if 'step_range' in seg.meta:
      s0, s1 = seg.meta['step_range']
      cnt = int(seg.meta.get('count') or (s1 - s0 + 1))
      extra += (
          f'<br><span style="color:#cbd5e1">Steps:</span> <b>{s0}\u2013{s1}</b>'
          f' ({cnt} total)'
      )
      if cnt > 0 and true_dur > 0:
        extra += (
            '<br><span style="color:#cbd5e1">Avg step time:</span>'
            f' <b>{true_dur / cnt:.2f}s/step</b>'
        )
    if 'last_step' in seg.meta and seg.meta['last_step'] is not None:
      extra += (
          '<br><span style="color:#cbd5e1">Last step before disruption:</span>'
          f' {seg.meta["last_step"]}'
      )
    if 'custom_type' in seg.meta:
      extra += (
          '<br><span style="color:#cbd5e1">Event type:</span>'
          f' {seg.meta["custom_type"]}'
      )

    finish_ts = seg.end
    if is_sub:
      finish_ts = min(run_end, seg.start + vis_dur)
      if container is not None:
        # Keep the visibility widening inside the container the sub-event
        # sits in so it never spills onto the next phase.
        finish_ts = min(finish_ts, max(container.end, seg.end))
    records.append({
        'Task': 'Run Timeline',
        'Start': _to_dt(seg.start),
        'Finish': _to_dt(finish_ts),
        'Color': color,
        'Hover': _hover(seg, extra, title_override=display_title),
        'ZOrder': _Z_ORDER_PRIORITY.get(seg.kind, 50),
    })

  df = pd.DataFrame(records)
  df = df.sort_values('ZOrder', kind='mergesort')

  fig = px.timeline(
      df,
      x_start='Start',
      x_end='Finish',
      y='Task',
      color='Color',
      color_discrete_map={c: c for c in df['Color'].unique()},
      category_orders={'Task': ['Run Timeline']},
  )
  _fix_legend(
      fig,
      kind_label_map=per_run_legend_map,
  )
  _add_topmost_hover_layer(fig, df)

  fig.update_xaxes(
      title=dict(
          text='Time (UTC)',
          font=dict(size=12, color='#475569'),
      ),
      tickfont=dict(size=11, color='#475569'),
      mirror='allticks',
      showspikes=True,
      spikemode='across',
      spikesnap='cursor',
      spikecolor='#64748b',
      spikethickness=1,
      spikedash='dot',
      gridcolor='#e2e8f0',
      griddash='dot',
      linecolor='#cbd5e1',
  )
  height = _RUN_TOP_PX + _RUN_PLOT_PX + _RUN_BOTTOM_PX
  fig.update_layout(
      font=dict(
          family=(
              'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,'
              ' sans-serif'
          ),
          color='#0f172a',
      ),
      title=dict(
          text=(
              f'<b>{title} \u2014 Continuous Event Timeline</b> <span'
              ' style="font-size:12px;color:#475569;font-weight:normal">(Total'
              f' Span: {_fmt_duration(total_span)} / {total_span:.1f}s)</span>'
          ),
          y=1 - _RUN_TITLE_TOP_PX / height,
          x=0.0,
          xanchor='left',
          yanchor='top',
          font=dict(size=13.5, color='#0f172a'),
      ),
      barmode='overlay',
      yaxis_title='',
      legend_title=dict(
          text='<b>Event Type</b>', font=dict(size=11.5, color='#334155')
      ),
      legend=dict(
          orientation='h',
          yanchor='bottom',
          y=1 + _RUN_LEGEND_GAP_PX / _RUN_PLOT_PX,
          xanchor='left',
          x=0,
          bgcolor='rgba(248, 250, 252, 0.92)',
          bordercolor='#e2e8f0',
          borderwidth=1,
          font=dict(size=11.5, color='#334155'),
      ),
      hoverlabel=dict(
          bgcolor='#0f172a',
          bordercolor='#334155',
          font=dict(
              family='Inter, system-ui, sans-serif', size=12, color='#f8fafc'
          ),
          align='left',
      ),
      margin=dict(t=_RUN_TOP_PX, b=_RUN_BOTTOM_PX, l=115, r=25),
      paper_bgcolor='#ffffff',
      plot_bgcolor='#f8fafc',
      height=height,
      # Read by _FIT_LEGEND_JS.
      meta=dict(
          fit_legend=dict(
              min_top_px=_RUN_TOP_PX,
              title_top_px=_RUN_TITLE_TOP_PX,
              title_gap_px=_FIT_LEGEND_TITLE_GAP_PX,
          )
      ),
  )
  fig.update_yaxes(
      fixedrange=True,
      tickfont=dict(size=12, color='#1e293b'),
      linecolor='#cbd5e1',
  )
  return fig


_NA = 'N/A'
# Shown under a KPI card whose metric failed; the error is in the Warnings box.
_SEE_WARNINGS = '<a href="#warnings" style="color:inherit;">See Warnings</a>'

_T = TypeVar('_T')


def _metric_or_warn(
    label: str,
    compute: Callable[[], _T],
    warnings: list[tuple[str, str]],
) -> Optional[_T]:
  """Returns compute(), or None after adding its error to `warnings`.

  A metric that fails shows N/A in its card and its error in the Warnings box,
  instead of a misleading 0 or the other metrics being dropped.

  Args:
    label: The metric, e.g. 'Goodput'; the first part of the warning.
    compute: Computes the metric.
    warnings: (label, error) pairs to append to.

  Returns:
    The metric, or None if compute raised.
  """
  try:
    return compute()
  except Exception as e:  # pylint: disable=broad-exception-caught
    warnings.append((label, f'{type(e).__name__}: {e}'))
    return None


def _duration_or_na(seconds: Optional[float]) -> str:
  return _NA if seconds is None else _fmt_duration(seconds)


def _build_warnings_html(warnings: Sequence[tuple[str, str]]) -> str:
  """Builds the Warnings box listing every metric error; '' if none."""
  if not warnings:
    return ''
  items = ''.join(
      '<li style="margin:0.3rem 0;"><b>'
      f'{html_lib.escape(label)}</b>: <code style="font-size:0.8rem;'
      ' background:#fee2e2; padding:1px 5px; border-radius:4px;'
      ' white-space:pre-wrap; word-break:break-word;">'
      f'{html_lib.escape(error)}</code></li>'
      for label, error in warnings
  )
  n = len(warnings)
  return (
      '<div id="warnings" class="warnings-box" role="alert" style="margin:'
      ' 0.5rem 0 1.25rem; padding: 1rem 1.25rem; border: 1px solid #fca5a5;'
      ' border-left: 6px solid #dc2626; border-radius: 10px; background:'
      ' #fef2f2;">'
      '<div style="font-size: 0.95rem; font-weight: 700; color: #991b1b;'
      ' margin-bottom: 0.35rem;">\u26a0 '
      f'{n} warning{"s" if n != 1 else ""}: some metrics are unavailable'
      ' (N/A)</div>'
      '<ul style="margin: 0; padding-left: 1.25rem; color: #7f1d1d;'
      f' font-size: 0.85rem; line-height: 1.45;">{items}</ul>'
      '</div>'
  )


def _build_summary_html(
    segments: list[builder.Segment],
    workload_id: str,
    entries: Union[list[dict[str, Any]], pd.DataFrame],
    rec_stats: Optional[metrics.RecoverySummaryStats],
    warnings: list[tuple[str, str]],
) -> str:
  """Builds a 5-KPI banner and breakdown table with inline bars.

  Each metric is computed on its own: one that fails shows N/A and adds its
  error to `warnings`, and the other metrics are still shown.

  Args:
    segments: The workload timeline.
    workload_id: The workload, for the Goodput calculation.
    entries: Goodput log payloads, or a DataFrame of LogRecords.
    rec_stats: The recovery incidents; None if they could not be extracted.
    warnings: (label, error) pairs; the errors of the metrics are appended.

  Returns:
    The summary HTML; '' if it could not be built, with the error in
    `warnings`.
  """
  try:
    goodput = _metric_or_warn(
        'Goodput',
        lambda: metrics.calculate_goodput_metrics(entries, workload_id),
        warnings,
    )
    timeline_metrics = _metric_or_warn(
        'MTBI / MTTR',
        lambda: metrics.calculate_timeline_metrics(segments),
        warnings,
    )
    mtbi = timeline_metrics['mtbi'] if timeline_metrics else None
    rec_ok = rec_stats is not None
    if rec_stats is None:
      rec_stats = metrics.RecoverySummaryStats()

    if rec_ok and rec_stats.num_incidents > 0:
      mttr_val = rec_stats.mean_mttr_s
      jit_mean = rec_stats.phase_stats.get('jit_compile_overhead', {}).get(
          'mean', 0.0
      )
      downtime_val = rec_stats.mean_downtime_s + jit_mean
    elif rec_ok and timeline_metrics:
      mttr_val = timeline_metrics['mttr']
      downtime_val = mttr_val
    else:
      mttr_val = None
      downtime_val = None

    # Build Per-Recovery Incident MTTR Breakdown (10 sub-phases summing to 100%
    # of MTTR)
    mttr_rows = []
    for key in metrics.RECOVERY_PHASE_ORDER:
      label = metrics.RECOVERY_PHASE_LABELS[key]
      pstat = rec_stats.phase_stats.get(
          key, {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'pct_of_mttr': 0.0}
      )
      mean_s = pstat['mean']
      min_s = pstat['min']
      max_s = pstat['max']
      pct = pstat['pct_of_mttr']
      phase_color = _PHASE_COLORS.get(key, '#3b82f6')
      is_active = mean_s > 0.05
      if key == 'lost_progress':
        row_bg = '#fffbeb' if is_active else '#f8fafc'
        text_color = '#92400e' if is_active else '#94a3b8'
        min_max_str = (
            f'{_fmt_duration(min_s)} / {_fmt_duration(max_s)} (Avg'
            f' {rec_stats.mean_rolled_back_steps:.1f} repeated steps)'
        )
      else:
        row_bg = '#ffffff' if is_active else '#f8fafc'
        text_color = '#0f172a' if is_active else '#94a3b8'
        min_max_str = f'{_fmt_duration(min_s)} / {_fmt_duration(max_s)}'

      bar_width_pct = min(100.0, max(0.0, pct))
      color_dot = (
          '<span style="display:inline-block;width:9px;height:9px;'
          f'border-radius:3px;background:{phase_color if is_active else "#cbd5e1"};'
          'margin-right:8px;vertical-align:middle;"></span>'
      )
      pct_cell = (
          '<div style="display:flex;align-items:center;justify-content:flex-end;gap:8px;">'
          '<div style="width:72px;height:6px;background:#e2e8f0;border-radius:999px;overflow:hidden;">'
          '<div'
          f' style="width:{bar_width_pct:.1f}%;height:100%;background:{phase_color if is_active else "#cbd5e1"};border-radius:999px;"></div>'
          '</div>'
          '<span'
          f' style="min-width:42px;text-align:right;font-variant-numeric:tabular-nums;">{pct:.1f}%</span>'
          '</div>'
      )

      mttr_rows.append(
          f'<tr style="background:{row_bg}; color:{text_color};'
          ' border-bottom:1px solid #f1f5f9;"><td style="padding:7px 10px;'
          f' font-weight:{500 if is_active else 400};">{color_dot}{label}</td><td'
          ' style="padding:7px 10px; text-align:right; font-weight:600;'
          f' font-variant-numeric:tabular-nums;">{_fmt_duration(mean_s)}</td><td'
          ' style="padding:7px 10px; text-align:right; font-size:0.8rem;'
          ' color:#475569;'
          f' font-variant-numeric:tabular-nums;">{min_max_str}</td><td'
          f' style="padding:7px 10px; text-align:right;">{pct_cell}</td></tr>'
      )

    if mttr_val:
      downtime_pct_text = f'{(downtime_val or 0.0) / mttr_val * 100.0:.1f}%'
      lost_pct_text = (
          f'{rec_stats.mean_lost_progress_s / mttr_val * 100.0:.1f}%'
      )
    elif mttr_val is None:
      downtime_pct_text = lost_pct_text = _NA
    else:
      downtime_pct_text = lost_pct_text = '0.0%'

    if rec_ok:
      mttr_rows.append(
          '<tr style="background:#eff6ff; font-weight:700; color:#1e40af;'
          ' border-top:2px solid #3b82f6;"><td style="padding:9px 10px;">Total'
          ' Time to Recover (MTTR)</td><td style="padding:9px 10px;'
          ' text-align:right;'
          f' font-variant-numeric:tabular-nums;">{_duration_or_na(mttr_val)}</td><td'
          ' style="padding:9px 10px; text-align:right; font-size:0.8rem;'
          f' font-variant-numeric:tabular-nums;">{_fmt_duration(rec_stats.min_mttr_s)}'
          f' / {_fmt_duration(rec_stats.max_mttr_s)}</td><td style="padding:9px'
          ' 10px; text-align:right;'
          ' font-variant-numeric:tabular-nums;">100.0%</td></tr>'
      )
    else:
      mttr_rows = [
          '<tr style="background:#fef2f2; color:#991b1b;"><td colspan="4"'
          f' style="padding:9px 10px;">{_NA}: the recovery incidents could'
          f' not be extracted. {_SEE_WARNINGS}.</td></tr>'
      ]

    # Goodput card: green with the value, red with N/A when it failed.
    if goodput is not None:
      goodput_val, _, last_step, total_job_time = goodput
      productive_s = (
          (goodput_val / 100.0) * total_job_time if total_job_time > 0 else 0.0
      )
      goodput_text = f'{goodput_val:.2f}%'
      goodput_detail = (
          f'{_fmt_duration(productive_s)} preserved productive /'
          f' {_fmt_duration(total_job_time)} total job time &bull; Step'
          f' {last_step}'
      )
      goodput_colors = ('#ecfdf5', '#a7f3d0', '#10b981', '#047857', '#065f46')
    else:
      goodput_text = _NA
      goodput_detail = f'Goodput unavailable. {_SEE_WARNINGS}.'
      goodput_colors = ('#fef2f2', '#fecaca', '#dc2626', '#b91c1c', '#991b1b')
    g_bg, g_border, g_accent, g_title, g_text = goodput_colors

    if rec_ok:
      incidents_text = f'{rec_stats.num_incidents} Incidents'
      disruptions_label = f'{rec_stats.num_incidents} Disruption' + (
          's' if rec_stats.num_incidents != 1 else ''
      )
      mttr_min_max = (
          f'Min: {_fmt_duration(rec_stats.min_mttr_s)} / Max:'
          f' {_fmt_duration(rec_stats.max_mttr_s)}'
      )
      lost_text = _fmt_duration(rec_stats.mean_lost_progress_s)
      lost_detail = (
          f'Avg <b>{rec_stats.mean_rolled_back_steps:.1f} repeated steps</b>'
          f' ({lost_pct_text} of MTTR)'
      )
    else:
      incidents_text = _NA
      disruptions_label = f'{_NA} Disruptions'
      mttr_min_max = _SEE_WARNINGS
      lost_text = _NA
      lost_detail = _SEE_WARNINGS

    return f"""
<div class="goodput-summary-box" style="margin: 0.5rem 0 1.5rem; padding: 1.35rem 1.5rem; border: 1px solid #e2e8f0; border-radius: 12px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(15,23,42,0.04), 0 1px 2px rgba(15,23,42,0.02);">
  <h3 style="margin-bottom: 0.9rem; font-size: 1.05rem; font-weight: 700; color: #0f172a;">Workload Metrics &amp; Recovery Summary</h3>

  <!-- Top Row: 5 Executive KPI Cards -->
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.85rem; margin-bottom: 1.35rem;">
    <div style="padding: 0.9rem 1.05rem; border-radius: 10px; background: {g_bg}; border: 1px solid {g_border}; border-left: 4px solid {g_accent};">
      <div style="font-size: 0.72rem; font-weight: 700; color: {g_title}; text-transform: uppercase; letter-spacing: 0.04em;">Goodput %</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: {g_text}; margin: 0.2rem 0; font-variant-numeric: tabular-nums;">{goodput_text}</div>
      <div style="font-size: 0.76rem; color: {g_text};">{goodput_detail}</div>
    </div>

    <div style="padding: 0.9rem 1.05rem; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #64748b;">
      <div style="font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Disruptions &amp; MTBI</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0; font-variant-numeric: tabular-nums;">{incidents_text}</div>
      <div style="font-size: 0.76rem; color: #475569;">MTBI: <b>{_duration_or_na(mtbi)}</b> between disruptions</div>
    </div>

    <div style="padding: 0.9rem 1.05rem; border-radius: 10px; background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6;">
      <div style="font-size: 0.72rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.04em;">Total MTTR</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: #1e3a8a; margin: 0.2rem 0; font-variant-numeric: tabular-nums;">{_duration_or_na(mttr_val)}</div>
      <div style="font-size: 0.76rem; color: #1e40af;">Time to Recover + Lost Progress &bull; {mttr_min_max}</div>
    </div>

    <div style="padding: 0.9rem 1.05rem; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #475569;">
      <div style="font-size: 0.72rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Time to Recover</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0; font-variant-numeric: tabular-nums;">{_duration_or_na(downtime_val)}</div>
      <div style="font-size: 0.76rem; color: #475569;">Phases 1\u20139 ({downtime_pct_text} of MTTR)</div>
    </div>

    <div style="padding: 0.9rem 1.05rem; border-radius: 10px; background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b;">
      <div style="font-size: 0.72rem; font-weight: 700; color: #b45309; text-transform: uppercase; letter-spacing: 0.04em;">Lost Progress</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: #92400e; margin: 0.2rem 0; font-variant-numeric: tabular-nums;">{lost_text}</div>
      <div style="font-size: 0.76rem; color: #92400e;">{lost_detail}</div>
    </div>
  </div>

  <!-- Second Row: Per-Recovery Incident Breakdown Table -->
  <div>
    <h4 style="font-size: 0.82rem; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">
      Per-Recovery Incident Breakdown (Mean across {disruptions_label})
    </h4>
    <div style="border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
      <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem;">
        <thead>
          <tr style="background: #f1f5f9; color: #334155; border-bottom: 1px solid #cbd5e1;">
            <th style="padding: 8px 10px; text-align: left; font-weight: 600;">Recovery Sub-Phase</th>
            <th style="padding: 8px 10px; text-align: right; font-weight: 600;">Mean</th>
            <th style="padding: 8px 10px; text-align: right; font-weight: 600;">Min / Max</th>
            <th style="padding: 8px 10px; text-align: right; font-weight: 600;">% MTTR</th>
          </tr>
        </thead>
        <tbody>
          {''.join(mttr_rows)}
        </tbody>
      </table>
    </div>
  </div>
</div>
"""
  except Exception as e:  # pylint: disable=broad-exception-caught
    warnings.append(('Metrics summary', f'{type(e).__name__}: {e}'))
    return ''


def _assemble_sections(
    segments: list[builder.Segment],
    workload_id: str,
    include_plotlyjs: bool,
    entries: Optional[Union[list[dict[str, Any]], pd.DataFrame]] = None,
) -> str:
  """Combines navigation, warnings, metrics summary, overview, and runs."""
  run_indices = sorted({s.run_idx for s in segments})

  # (label, error) of each metric that could not be computed.
  warnings: list[tuple[str, str]] = []
  rec_stats = None
  summary_html = ''
  if entries is not None:
    rec_stats = _metric_or_warn(
        'Recovery incidents',
        lambda: metrics.extract_recovery_incidents(entries, segments),
        warnings,
    )
    summary_html = _build_summary_html(
        segments, workload_id, entries, rec_stats, warnings
    )

  nav_items = [
      '<span class="label">Jump to:</span>',
      '<a class="ov" href="#overview">Overview</a>',
  ]
  for run_idx in run_indices:
    nav_items.append(
        f'<a href="#run-{run_idx}">Run {run_idx + 1}'
        '<span style="font-size:0.68rem;color:#64748b">'
        f' ({_run_label(run_idx)})</span></a>'
    )
  if warnings:
    nav_items.append(
        '<a href="#warnings" style="color:#b91c1c;border-color:#fca5a5;'
        f'background:#fef2f2;">\u26a0 Warnings ({len(warnings)})</a>'
    )

  parts = [
      '<nav>' + ''.join(nav_items) + '</nav>',
      _build_warnings_html(warnings),
      summary_html,
  ]

  overview_fig = build_overview_figure(
      segments,
      workload_id,
      entries=entries,
      # Without them (see Warnings), the hover texts leave out the recovery
      # phases rather than extracting the incidents again.
      rec_stats=(
          metrics.RecoverySummaryStats()
          if rec_stats is None and entries is not None
          else rec_stats
      ),
  )
  overview_div = overview_fig.to_html(
      full_html=False,
      include_plotlyjs=include_plotlyjs,
      post_script=_FIT_LEGEND_JS,
  )
  panel_labels = {
      'step_time': 'Step Time vs. Baseline',
      'slices': 'Elastic Slices',
  }
  badge_parts = ['Full Timeline &amp; Sub-Event Overlay'] + [
      panel_labels[p]
      for p in (overview_fig.layout.meta or {}).get('panels', [])
      if p in panel_labels
  ]
  parts.append(
      '<section id="overview" class="section-card"><div'
      ' class="section-header"><h2>Workload Overview</h2><span'
      f' class="run-meta-badge">{" &bull; ".join(badge_parts)}</span></div>'
      f'{overview_div}</section>'
  )

  for run_idx in run_indices:
    run_segs = [
        s
        for s in segments
        if (s.run_idx == run_idx and s.kind != events.EventType.INTERRUPTION)
        or (
            s.kind == events.EventType.INTERRUPTION and s.run_idx == run_idx - 1
        )
    ]
    label = _run_label(run_idx)
    run_title = f'Run {run_idx + 1} \u2014 {label}'

    meta_badge = ''
    if run_segs:
      r_start = min(s.start for s in run_segs)
      r_end = max(s.end for s in run_segs)
      r_span = max(0.0, r_end - r_start)
      step_ranges = [
          s.meta['step_range']
          for s in run_segs
          if 'step_range' in s.meta and s.run_idx == run_idx
      ]
      step_txt = ''
      if step_ranges:
        min_s = min(r[0] for r in step_ranges)
        max_s = max(r[1] for r in step_ranges)
        step_txt = f' &bull; Steps {min_s}\u2013{max_s}'
      meta_badge = (
          f'<span class="run-meta-badge">{_fmt_dt(r_start)} \u2192'
          f' {_fmt_dt(r_end)} ({_fmt_duration(r_span)}){step_txt}</span>'
      )

    breakdown_div = ''
    if run_idx > 0 and rec_stats and rec_stats.incidents:
      target_inc = next(
          (inc for inc in rec_stats.incidents if inc.incident_idx == run_idx),
          None,
      )
      if target_inc is not None:
        breakdown_fig = build_recovery_breakdown_figure(target_inc, run_title)
        breakdown_div = (
            breakdown_fig.to_html(full_html=False, include_plotlyjs=False)
            + '<div class="chart-divider"></div>'
        )

    run_fig = build_figure(run_segs, run_title)
    run_div = run_fig.to_html(
        full_html=False, include_plotlyjs=False, post_script=_FIT_LEGEND_JS
    )
    parts.append(
        f'<section id="run-{run_idx}" class="section-card">'
        '<div class="section-header">'
        f'<h2>{run_title}</h2>'
        '<div style="display:flex;align-items:center;gap:0.5rem;">'
        f'{meta_badge}'
        '<a class="back" href="#overview">\u2191 Overview</a>'
        '</div>'
        '</div>'
        f'{breakdown_div}'
        f'{run_div}'
        '</section>'
    )

  return '\n'.join(parts)


def to_html_file(
    segments: list[builder.Segment],
    workload_id: str,
    path: str,
    entries: Optional[Union[list[dict[str, Any]], pd.DataFrame]] = None,
) -> None:
  """Writes the full interactive goodput timeline to a standalone HTML file."""
  body = _assemble_sections(
      segments, workload_id, include_plotlyjs=True, entries=entries
  )
  html = _HTML_PAGE.format(
      title=f'Goodput Events Timeline - {html_lib.escape(workload_id)}',
      workload_id=html_lib.escape(workload_id),
      body=body,
  )
  with open(path, 'w', encoding='utf-8') as f:
    f.write(html)


def to_html_div(
    segments: list[builder.Segment],
    workload_id: str,
    entries: Optional[Union[list[dict[str, Any]], pd.DataFrame]] = None,
) -> str:
  """Returns the HTML div snippet containing the goodput timeline tabs."""
  return _assemble_sections(
      segments, workload_id, include_plotlyjs=False, entries=entries
  )
