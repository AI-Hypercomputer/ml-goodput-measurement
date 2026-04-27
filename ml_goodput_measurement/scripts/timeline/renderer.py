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
  raise ImportError('Required: pip install plotly pandas') from exc

_FALLBACK_COLOR = '#95a5a6'
_FALLBACK_LABEL = 'Unknown'

_OVERVIEW_KIND_LABELS: dict[EventType, str] = {
    EventType.STEP_BLOCK:   'Stepping',
    EventType.STARTUP:      'Startup / Recovery',
    EventType.INTERRUPTION: 'Interruption',
}
_OVERVIEW_ROW_ORDER = ['Stepping', 'Startup / Recovery', 'Interruption']

_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; padding: 0 1rem 3rem; color: #222; }}
    h1 {{ padding: 1rem 0 0.75rem; font-size: 1.3rem; }}
    nav {{ position: sticky; top: 0; background: #fff;
           border-bottom: 1px solid #e0e0e0; padding: 0.4rem 0;
           z-index: 100; display: flex; flex-wrap: wrap; gap: 0.25rem;
           align-items: center; }}
    nav .label {{ font-size: 0.75rem; color: #888; margin-right: 0.25rem;
                  white-space: nowrap; }}
    nav a {{ font-size: 0.75rem; color: #1a73e8; text-decoration: none;
             padding: 0.2rem 0.5rem; border-radius: 3px; white-space: nowrap; }}
    nav a:hover {{ background: #e8f0fe; }}
    nav a.ov {{ font-weight: 600; border: 1px solid #c5d8f8; }}
    section {{ margin-top: 2rem; scroll-margin-top: 3.5rem; }}
    h2 {{ font-size: 1rem; color: #333; margin-bottom: 0.25rem; }}
    a.back {{ font-size: 0.75rem; color: #bbb; margin-left: 0.75rem;
              text-decoration: none; }}
    a.back:hover {{ color: #1a73e8; }}
  </style>
</head>
<body>
  <h1>Goodput Events Timeline &mdash; {workload_id}</h1>
  {body}
</body>
</html>
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


def _fix_legend(fig: go.Figure, kind_label_map: dict | None = None) -> None:
  seen: set[EventType] = set()
  for trace in fig.data:
    trace_color = getattr(trace.marker, 'color', None)
    matched = next(
        (k for k, s in EVENT_STYLE_MAP.items() if s.color == trace_color), None
    )
    label = (
        kind_label_map[matched]
        if (kind_label_map and matched and matched in kind_label_map)
        else (EVENT_STYLE_MAP[matched].row_label if matched else None)
    )
    if matched and matched not in seen and label:
      trace.name = label
      trace.showlegend = True
      seen.add(matched)
    else:
      trace.showlegend = False


def _hover(seg: Segment, extra_lines: str = '') -> str:
  return (
      f'<b>{seg.label}</b><br>'
      f'Start:    {_fmt_dt(seg.start)}<br>'
      f'End:      {_fmt_dt(seg.end)}<br>'
      f'Duration: {_fmt_duration(seg.duration_s)}'
      + extra_lines
  )

def build_overview_figure(segments: list[Segment], workload_id: str) -> go.Figure:
  overview_segs = [s for s in segments if s.kind in _OVERVIEW_KIND_LABELS]
  if not overview_segs:
    fig = go.Figure()
    fig.update_layout(title=f'No data for {workload_id}')
    return fig

  records = []
  for seg in overview_segs:
    style = EVENT_STYLE_MAP.get(seg.kind)
    records.append({
        'Task':   _OVERVIEW_KIND_LABELS[seg.kind],
        'Start':  _to_dt(seg.start),
        'Finish': _to_dt(seg.end),
        'Color':  style.color if style else _FALLBACK_COLOR,
        'Hover':  _hover(seg, f'<br>Run: {seg.run_idx + 1} — {_run_label(seg.run_idx)}'),
    })

  df = pd.DataFrame(records)
  row_order = [r for r in _OVERVIEW_ROW_ORDER if r in df['Task'].values]

  fig = px.timeline(
      df,
      x_start='Start', x_end='Finish', y='Task',
      color='Color',
      color_discrete_map={c: c for c in df['Color'].unique()},
      category_orders={'Task': row_order},
      custom_data=['Hover'],
      title=f'Overview — {workload_id}',
  )
  fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')
  _fix_legend(fig, kind_label_map=_OVERVIEW_KIND_LABELS)

  fig.update_xaxes(
      title='Time (UTC)',
      rangeslider_visible=True,
      rangeselector=dict(buttons=[
          dict(count=1,  label='1h',  step='hour', stepmode='backward'),
          dict(count=6,  label='6h',  step='hour', stepmode='backward'),
          dict(count=12, label='12h', step='hour', stepmode='backward'),
          dict(count=1,  label='1d',  step='day',  stepmode='backward'),
          dict(step='all', label='All'),
      ]),
  )
  fig.update_layout(
      yaxis_title='',
      legend_title='Event Type',
      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
      plot_bgcolor='#f8f9fa',
      height=380,
  )
  fig.update_yaxes(autorange='reversed', fixedrange=True)
  return fig

def build_figure(segments: list[Segment], title: str) -> go.Figure:
  if not segments:
    fig = go.Figure()
    fig.update_layout(title=f'No data - {title}')
    return fig

  multi_run = len({s.run_idx for s in segments}) > 1

  records = []
  for seg in segments:
    style = EVENT_STYLE_MAP.get(seg.kind)
    row = (
        f'Run {seg.run_idx + 1}: {style.row_label if style else _FALLBACK_LABEL}'
        if multi_run
        else (style.row_label if style else _FALLBACK_LABEL)
    )
    extra = ''
    if 'step_range' in seg.meta:
      s0, s1 = seg.meta['step_range']
      extra += f'<br>Steps: {s0}-{s1} ({seg.meta["count"]} total)'
    if 'last_step' in seg.meta and seg.meta['last_step'] is not None:
      extra += f'<br>Last step before interrupt: {seg.meta["last_step"]}'
    if 'custom_type' in seg.meta:
      extra += f'<br>Event type: {seg.meta["custom_type"]}'
    records.append({
        'Task':   row,
        'Start':  _to_dt(seg.start),
        'Finish': _to_dt(seg.end),
        'Color':  style.color if style else _FALLBACK_COLOR,
        'Hover':  _hover(seg, extra),
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
      x_start='Start', x_end='Finish', y='Task',
      color='Color',
      color_discrete_map={c: c for c in df['Color'].unique()},
      category_orders={'Task': row_order},
      custom_data=['Hover'],
      title=title,
  )
  fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')
  _fix_legend(fig)

  fig.update_xaxes(
      title='Time (UTC)',
      mirror='allticks',
      showspikes=True,
      spikemode='across',
      spikesnap='cursor',
      spikecolor='#999',
      spikethickness=1,
      spikedash='dot',
  )
  fig.update_layout(
      yaxis_title='',
      legend_title='Event Type',
      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
      plot_bgcolor='#f8f9fa',
      height=max(400, 50 * len(row_order) + 150),
  )
  fig.update_yaxes(autorange='reversed', fixedrange=True)
  return fig


def _assemble_sections(
    segments: list[Segment],
    workload_id: str,
    include_plotlyjs,
) -> str:
  run_indices = sorted({s.run_idx for s in segments})

  nav_items = [
      '<span class="label">Jump to:</span>',
      '<a class="ov" href="#overview">Overview</a>',
  ]
  for run_idx in run_indices:
    nav_items.append(
        f'<a href="#run-{run_idx}">Run {run_idx + 1}'
        f'<span style="font-size:0.65rem;color:#888">'
        f' ({_run_label(run_idx)})</span></a>'
    )

  parts = ['<nav>' + ''.join(nav_items) + '</nav>']

  overview_div = build_overview_figure(segments, workload_id).to_html(
      full_html=False, include_plotlyjs=include_plotlyjs
  )
  parts.append(
      f'<section id="overview">'
      f'<h2>Workload Overview</h2>'
      f'{overview_div}'
      f'</section>'
  )

  for run_idx in run_indices:
    run_segs = [s for s in segments if s.run_idx == run_idx]
    label = _run_label(run_idx)
    run_title = f'Run {run_idx + 1} — {label}'
    run_div = build_figure(run_segs, run_title).to_html(
        full_html=False, include_plotlyjs=False
    )
    parts.append(
        f'<section id="run-{run_idx}">'
        f'<h2>{run_title}'
        f'<a class="back" href="#overview">↑ overview</a></h2>'
        f'{run_div}'
        f'</section>'
    )

  return '\n'.join(parts)


def to_html_file(segments: list[Segment], workload_id: str, path: str) -> None:
  body = _assemble_sections(segments, workload_id, include_plotlyjs=True)
  html = _HTML_PAGE.format(
      title=f'Goodput Events Timeline - {workload_id}',
      workload_id=workload_id,
      body=body,
  )
  with open(path, 'w', encoding='utf-8') as f:
    f.write(html)


def to_html_div(segments: list[Segment], workload_id: str) -> str:
  return _assemble_sections(segments, workload_id, include_plotlyjs=False)
