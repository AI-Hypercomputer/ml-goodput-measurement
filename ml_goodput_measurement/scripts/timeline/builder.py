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

from dataclasses import dataclass, field

from .event_parser import ParsedLogs
from .events import EventType

_INF = float('inf')


@dataclass
class Segment:
  kind: EventType
  start: float
  end: float
  run_idx: int
  label: str = ''
  meta: dict = field(default_factory=dict)

  @property
  def duration_s(self) -> float:
    return self.end - self.start


def build_timeline(parsed: ParsedLogs) -> list[Segment]:
  segments: list[Segment] = []
  job_starts = parsed.job_starts

  if not job_starts:
    return segments

  sorted_steps = sorted(parsed.steps.items())

  overall_end = (
      parsed.job_ends[0] if parsed.job_ends
      else (sorted_steps[-1][1] if sorted_steps
            else _max_timestamp(parsed))
  )

  for run_idx, job_start in enumerate(job_starts):
    next_job_start = (
        job_starts[run_idx + 1] if run_idx + 1 < len(job_starts) else None
    )
    run_end_cap = next_job_start if next_job_start is not None else overall_end

    # Steps in this run started after job_start and before the next restart.
    run_steps = [
        (s, t)
        for s, t in sorted_steps
        if job_start <= t < (run_end_cap if run_end_cap is not None else _INF)
    ]

    first_step_time = run_steps[0][1] if run_steps else run_end_cap
    last_step_time = run_steps[-1][1] if run_steps else None

    # Interruption.
    if next_job_start is not None:
      gap_start = last_step_time if last_step_time is not None else job_start
      segments.append(Segment(
          kind=EventType.INTERRUPTION,
          start=gap_start,
          end=next_job_start,
          run_idx=run_idx,
          label=f'Interruption after run {run_idx + 1}',
          meta={'last_step': run_steps[-1][0] if run_steps else None},
      ))

    # Startup/recovery.
    startup_end = first_step_time if first_step_time is not None else run_end_cap
    startup_label = 'Startup' if run_idx == 0 else f'Recovery (restart {run_idx})'
    if startup_end is not None and startup_end > job_start:
      segments.append(Segment(
          kind=EventType.STARTUP,
          start=job_start,
          end=startup_end,
          run_idx=run_idx,
          label=startup_label,
      ))

    # Sub-phases within the startup/recovery.
    window_end = startup_end if startup_end is not None else _INF
    for kind, intervals, lbl in [
        (EventType.TPU_INIT, parsed.tpu_init, 'Device Init (TPU)'),
        (EventType.TRAINING_PREP, parsed.training_prep, 'Training Prep (incl. checkpoint restore)'),
    ]:
      for t_start, t_end in intervals:
        if job_start <= t_start < window_end:
          segments.append(Segment(
              kind=kind, start=t_start, end=t_end, run_idx=run_idx, label=lbl
          ))

    # Synchronous data loading in the recovery window.
    for dl_start, dl_end in parsed.data_loading:
      if job_start <= dl_start < window_end:
        segments.append(Segment(
            kind=EventType.DATA_LOADING,
            start=dl_start,
            end=dl_end,
            run_idx=run_idx,
            label='Data Loading',
        ))

    # Stepping.
    if run_steps:
      step_nums = [s for s, _ in run_steps]
      last_step_end = (
          run_end_cap if run_end_cap is not None
          else run_steps[-1][1] + _avg_step_duration(run_steps)
      )
      segments.append(Segment(
          kind=EventType.STEP_BLOCK,
          start=run_steps[0][1],
          end=last_step_end,
          run_idx=run_idx,
          label=f'Steps {step_nums[0]}-{step_nums[-1]} ({len(run_steps)} steps)',
          meta={'step_range': (step_nums[0], step_nums[-1]), 'count': len(run_steps)},
      ))

    # Custom events.
    for event_type, intervals in parsed.custom.items():
      for c_start, c_end in intervals:
        if job_start <= c_start < (run_end_cap if run_end_cap is not None else _INF):
          segments.append(Segment(
              kind=EventType.CUSTOM,
              start=c_start,
              end=c_end,
              run_idx=run_idx,
              label=f'Custom: {event_type}',
              meta={'custom_type': event_type},
          ))

  return sorted(segments, key=lambda s: s.start)


def _max_timestamp(parsed: ParsedLogs) -> float:
  candidates: list[float] = list(parsed.job_starts) + list(parsed.job_ends)
  for start, end in parsed.tpu_init + parsed.training_prep + parsed.data_loading:
    candidates.extend([start, end])
  candidates.extend(parsed.steps.values())
  for intervals in parsed.custom.values():
    for start, end in intervals:
      candidates.extend([start, end])
  return max(candidates) if candidates else 0.0


def _avg_step_duration(run_steps: list[tuple[int, float]]) -> float:
  if len(run_steps) < 2:
    return 0.0
  return (run_steps[-1][1] - run_steps[0][1]) / (len(run_steps) - 1)
