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

import datetime

from .builder import Segment
from .events import EVENT_STYLE_MAP, ROW_ORDER, EventType

try:
  import pandas as pd
  import plotly.express as px
  import plotly.graph_objects as go
except ImportError as exc:
  raise ImportError(
      'Required dependencies plotly pandas'
  ) from exc

_FALLBACK_STYLE_COLOR = '#95a5a6'
_FALLBACK_STYLE_LABEL = 'Unknown'


def _to_dt(unix: float) -> datetime.datetime:
  return datetime.datetime.fromtimestamp(unix, tz=datetime.timezone.utc)


def _fmt_duration(seconds: float) -> str:
  if seconds < 60:
    return f'{seconds:.1f}s'
  if seconds < 3600:
    return f'{seconds / 60:.1f}m'
  return f'{seconds / 3600:.2f}h'


def _row_label(seg: Segment, multi_run: bool) -> str:
  style = EVENT_STYLE_MAP.get(seg.kind)
  label = style.row_label if style else _FALLBACK_STYLE_LABEL
  return f'Run {seg.run_idx + 1}: {label}' if multi_run else label


def build_figure(segments: list[Segment], workload_id: str) -> go.Figure:
  if not segments:
    fig = go.Figure()
    fig.update_layout(title=f'No data for {workload_id}')
    return fig

  multi_run = len({s.run_idx for s in segments}) > 1

  records = []
  for seg in segments:
    style = EVENT_STYLE_MAP.get(seg.kind)
    color = style.color if style else _FALLBACK_STYLE_COLOR

    hover = (
        f'<b>{seg.label}</b><br>'
        f'Duration: {_fmt_duration(seg.duration_s)}<br>'
        f'Start: {_to_dt(seg.start).strftime("%Y-%m-%d %H:%M:%S")} UTC'
    )
    if 'step_range' in seg.meta:
      s0, s1 = seg.meta['step_range']
      hover += f'<br>Steps: {s0}-{s1} ({seg.meta["count"]} total)'
    if 'last_step' in seg.meta and seg.meta['last_step'] is not None:
      hover += f'<br>Last step before interrupt: {seg.meta["last_step"]}'
    if 'custom_type' in seg.meta:
      hover += f'<br>Event type: {seg.meta["custom_type"]}'

    records.append({
        'Task': _row_label(seg, multi_run),
        'Start': _to_dt(seg.start),
        'Finish': _to_dt(seg.end),
        'Color': color,
        'Kind': seg.kind,
        'Hover': hover,
    })

  df = pd.DataFrame(records)

  if multi_run:
    run_count = max(s.run_idx for s in segments) + 1
    row_order = [
        f'Run {r + 1}: {row}'
        for r in range(run_count)
        for row in ROW_ORDER
        if f'Run {r + 1}: {row}' in df['Task'].values
    ]
  else:
    row_order = [r for r in ROW_ORDER if r in df['Task'].values]

  fig = px.timeline(
      df,
      x_start='Start',
      x_end='Finish',
      y='Task',
      color='Color',
      color_discrete_map={c: c for c in df['Color'].unique()},
      category_orders={'Task': row_order},
      hover_name='Hover',
      title=f'Goodput Events Timeline - {workload_id}',
  )

  seen_kinds: set[EventType] = set()
  for trace in fig.data:
    trace_color = getattr(trace.marker, 'color', None)
    matched_kind = next(
        (k for k, s in EVENT_STYLE_MAP.items() if s.color == trace_color), None
    )
    if matched_kind and matched_kind not in seen_kinds:
      trace.name = EVENT_STYLE_MAP[matched_kind].row_label
      trace.showlegend = True
      seen_kinds.add(matched_kind)
    else:
      trace.showlegend = False
    trace.hovertemplate = '%{customdata[0]}<extra></extra>'
    trace.customdata = [[r['Hover']] for r in records if r['Color'] == trace_color]

  fig.update_layout(
      xaxis_title='Time (UTC)',
      yaxis_title='',
      legend_title='Event Type',
      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
      plot_bgcolor='#f8f9fa',
      height=max(400, 50 * len(row_order) + 150),
  )
  fig.update_yaxes(autorange='reversed')
  return fig


def to_html_div(segments: list[Segment], workload_id: str) -> str:
  return build_figure(segments, workload_id).to_html(
      full_html=False, include_plotlyjs='cdn'
  )


def to_html_file(segments: list[Segment], workload_id: str, path: str) -> None:
  build_figure(segments, workload_id).write_html(path, include_plotlyjs=True)
