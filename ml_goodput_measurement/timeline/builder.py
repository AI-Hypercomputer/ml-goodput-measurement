"""Timeline segment builder for goodput logs."""

import dataclasses
from typing import Any

from . import event_parser
from . import events
from . import step_times

# Alias for convenience
EventType = events.EventType


@dataclasses.dataclass
class Segment:
  kind: events.EventType
  start: float
  end: float
  run_idx: int
  label: str = ''
  meta: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def duration_s(self) -> float:
    return self.end - self.start


def _subtract_intervals(
    window_start: float,
    window_end: float,
    excludes: list[tuple[float, float]],
) -> list[tuple[float, float]]:
  """Returns the parts of [window_start, window_end] outside the excludes.

  Used to carve the JIT compile overhead out of a stepping container: the
  remaining stretches are drawn as Lost Progress (in Wasted Progress) or
  Productive Steps (in Stepping). Excludes are clipped to the window and
  merged when they overlap; gaps of 1 ms or less are ignored.

  Args:
    window_start: Start of the window, Unix seconds.
    window_end: End of the window, Unix seconds.
    excludes: (start, end) intervals to remove, in any order.

  Returns:
    The remaining (start, end) intervals, in time order.
  """
  merged: list[tuple[float, float]] = []
  for st, ed in sorted(excludes):
    st = max(window_start, st)
    ed = min(window_end, ed)
    if st >= ed:
      continue
    if merged and st <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], ed))
    else:
      merged.append((st, ed))

  remaining: list[tuple[float, float]] = []
  curr = window_start
  for c_st, c_ed in merged:
    if c_st > curr + 1e-3:
      remaining.append((curr, c_st))
    curr = max(curr, c_ed)
  if curr < window_end - 1e-3:
    remaining.append((curr, window_end))
  return remaining


def build_timeline(parsed: event_parser.ParsedLogs) -> list[Segment]:
  """Constructs high-level runs and detailed sub-event timeline segments."""
  segments: list[Segment] = []

  step_starts = sorted([t for s, t in parsed.steps])
  tpu_starts = sorted([st for st, ed in parsed.tpu_init])
  elastic_reinit_starts = sorted(
      [st for st, ed in parsed.elastic_reinitialization]
  )

  if not step_starts and not tpu_starts and not elastic_reinit_starts:
    return segments

  filtered_tpu_starts = []
  for t_st in tpu_starts:
    inside_elastic = False
    for e_st, e_ed in parsed.elastic_reinitialization:
      if e_st <= t_st <= e_ed:
        inside_elastic = True
        break
    if not inside_elastic:
      filtered_tpu_starts.append(t_st)

  # Baseline step time (lower median) for JIT Compile Overhead detection. With
  # elastic slice counts, it is the step time with all slices active, and a
  # step with fewer active slices is compared with the step time measured with
  # them (see step_times.Baseline).
  baseline = step_times.Baseline(parsed.steps, parsed.elastic_slice_counts)
  global_baseline_step = baseline.step_time_s or 60.0

  # Map each elastic_reinitialization start to its preceding elastic_scale_up
  # or elastic_slice_down start
  all_elastic_waits = sorted(
      [(ws, we, 'scale_up') for ws, we in parsed.elastic_scale_up]
      + [(ws, we, 'slice_down') for ws, we in parsed.elastic_slice_down]
  )
  adjusted_elastic_starts = []
  for e_st in elastic_reinit_starts:
    matched_ws = e_st
    for ws, we, _ in all_elastic_waits:
      if ws <= e_st and 0.0 <= (e_st - we) <= 30.0:
        prev_steps = [ts for _, ts in parsed.steps if ts <= ws]
        if not prev_steps:
          matched_ws = ws
        else:
          last_ts = prev_steps[-1]
          ref_dur = (
              baseline.expected_s(last_ts, last_ts) or global_baseline_step
          )
          if (
              len(prev_steps) >= 2
              and 0 < (prev_steps[-1] - prev_steps[-2]) < 600
          ):
            ref_dur = prev_steps[-1] - prev_steps[-2]
          if ws >= last_ts + 0.8 * ref_dur:
            matched_ws = ws
        break
    adjusted_elastic_starts.append(matched_ws)

  effective_starts = sorted(
      list(set(filtered_tpu_starts + adjusted_elastic_starts))
  )
  earliest_step = step_starts[0] if step_starts else float('inf')
  earliest_job = parsed.job_starts[0] if parsed.job_starts else float('inf')
  first_activity = min(earliest_step, earliest_job)
  if not effective_starts:
    if first_activity < float('inf'):
      effective_starts = [first_activity]
  elif first_activity < effective_starts[0]:
    has_early_steps = any(
        first_activity <= ts < effective_starts[0] for ts in step_starts
    )
    if has_early_steps:
      effective_starts = [first_activity] + effective_starts
    else:
      effective_starts[0] = first_activity

  scale_up_starts = {ws for ws, _ in parsed.elastic_scale_up}

  def _synthesize_aot_segments(
      window_start: float, window_end: float, target_run: int
  ) -> None:
    run_comps = [
        (cs, ce)
        for cs, ce in parsed.compilation
        if max(cs, window_start) < min(ce, window_end)
    ]
    if run_comps:
      for cs, ce in run_comps:
        segments.append(
            Segment(
                kind=events.EventType.AOT_COMPILE,
                start=max(cs, window_start),
                end=min(ce, window_end),
                run_idx=target_run,
                label='AOT Compilation',
            )
        )
      return

    run_preps = [
        (ps, pe)
        for ps, pe in parsed.training_prep
        if max(ps, window_start) < min(pe, window_end)
    ]
    run_dloads = [
        (ds, de)
        for ds, de in parsed.data_loading
        if max(ds, window_start) < min(de, window_end)
    ]
    run_reinits = [
        (rs, re)
        for rs, re in parsed.elastic_reinitialization
        if max(rs, window_start) < min(re, window_end)
    ]
    run_ckpts = [
        (cs, ce)
        for cs, ce in parsed.checkpoint_restore
        if max(cs, window_start) < min(ce, window_end)
    ]
    if run_preps:
      prep_end = run_preps[-1][1]
      if run_reinits and run_reinits[-1][1] > prep_end:
        aot_end = min(window_end, run_reinits[-1][1])
      elif run_dloads and run_dloads[0][0] > prep_end:
        aot_end = min(window_end, run_dloads[0][0])
      else:
        aot_end = window_end
      if aot_end - prep_end > 0.05:
        # Exclude any checkpoint_restore sub-intervals that occurred after
        # training_prep
        free_intervals = [(prep_end, aot_end)]
        for cs, ce in run_ckpts:
          next_free = []
          for fs, fe in free_intervals:
            if ce <= fs or cs >= fe:
              next_free.append((fs, fe))
            else:
              if cs > fs:
                next_free.append((fs, cs))
              if ce < fe:
                next_free.append((ce, fe))
          free_intervals = next_free
        for fs, fe in free_intervals:
          if fe - fs > 0.05:
            segments.append(
                Segment(
                    kind=events.EventType.AOT_COMPILE,
                    start=fs,
                    end=fe,
                    run_idx=target_run,
                    label='AOT Compilation',
                )
            )

  max_step_before_run = -1
  current_run_idx = 0
  run_end_boundaries: list[tuple[int, float]] = []

  for i, st in enumerate(effective_starts):
    nxt = (
        effective_starts[i + 1]
        if i + 1 < len(effective_starts)
        else float('inf')
    )

    # Steps belonging to this recovery attempt
    my_step_pairs = sorted(
        [(sn, ts) for sn, ts in parsed.steps if st <= ts < nxt],
        key=lambda x: x[1],
    )
    my_steps = [ts for _, ts in my_step_pairs]

    if not my_step_pairs:
      # Aborted pre-step attempt: stays in current_run_idx if future steps exist
      has_future_steps = any(ts >= nxt for ts in step_starts)
      if not has_future_steps and step_starts:
        continue
      if nxt < float('inf'):
        # Determine how far active startup sub-events progressed before failing
        active_sub_ends = [st]
        for interval_list in (
            parsed.elastic_scale_up,
            parsed.elastic_slice_down,
            parsed.elastic_reinitialization,
            parsed.tpu_init,
            parsed.training_prep,
            parsed.checkpoint_restore,
            parsed.compilation,
            parsed.data_loading,
        ):
          for ev_s, ev_e in interval_list:
            if st <= ev_s < nxt:
              active_sub_ends.append(min(ev_e, nxt))
        active_end = max(active_sub_ends)
        # If there is a large dead gap (>30s) before nxt and a new job_start or
        # tpu_init follows, mark the active portion as STARTUP and the dead gap
        # as INTERRUPTION.
        has_job_restart_in_gap = any(
            active_end <= js <= nxt for js in parsed.job_starts
        )
        if nxt - active_end > 30.0 and has_job_restart_in_gap:
          if active_end > st:
            segments.append(
                Segment(
                    kind=events.EventType.STARTUP,
                    start=st,
                    end=active_end,
                    run_idx=current_run_idx,
                    label=(
                        f'Recovery (restart {current_run_idx})'
                        if current_run_idx > 0
                        else 'Startup'
                    ),
                )
            )
            _synthesize_aot_segments(st, active_end, current_run_idx)
          if current_run_idx > 0:
            segments.append(
                Segment(
                    kind=events.EventType.INTERRUPTION,
                    start=active_end,
                    end=nxt,
                    run_idx=current_run_idx - 1,
                    label='Interruption',
                    meta={'last_step': max_step_before_run},
                )
            )
        else:
          segments.append(
              Segment(
                  kind=events.EventType.STARTUP,
                  start=st,
                  end=nxt,
                  run_idx=current_run_idx,
                  label=(
                      f'Recovery (restart {current_run_idx})'
                      if current_run_idx > 0
                      else 'Startup'
                  ),
              )
          )
          _synthesize_aot_segments(st, nxt, current_run_idx)
      continue

    # 1. Startup / Recovery Phase (attempt reached training steps)
    recovery_end = my_steps[0]
    if recovery_end < float('inf'):
      # Merge with an immediately preceding contiguous STARTUP of the same
      # current_run_idx
      merged_startup = False
      for prev_seg in reversed(segments):
        if (
            prev_seg.kind == events.EventType.STARTUP
            and prev_seg.run_idx == current_run_idx
            and abs(prev_seg.end - st) < 1e-3
        ):
          prev_seg.end = recovery_end
          merged_startup = True
          break
      if not merged_startup:
        segments.append(
            Segment(
                kind=events.EventType.STARTUP,
                start=st,
                end=recovery_end,
                run_idx=current_run_idx,
                label=(
                    f'Recovery (restart {current_run_idx})'
                    if current_run_idx > 0
                    else 'Startup'
                ),
            )
        )
      _synthesize_aot_segments(st, recovery_end, current_run_idx)

    # 2. Stepping (split into Wasted Progress vs. Productive Steps)
    run_deltas = step_times.consecutive_step_deltas(my_step_pairs)
    run_base = step_times.lower_median([d.delta_s for d in run_deltas])
    if run_deltas:
      step_dur = run_base
    elif len(my_steps) > 1:
      step_dur = (my_steps[-1] - my_steps[0]) / (len(my_steps) - 1)
    else:
      step_dur = 60.0

    step_block_end = min(my_steps[-1] + step_dur, nxt)

    # Extend step_block_end to include any blocking checkpoint_save during this
    # step block
    for cs, ce in parsed.checkpoint_save:
      if my_steps[0] <= cs < nxt and (cs - my_steps[-1]) <= max(
          step_dur * 5.0, 300.0
      ):
        step_block_end = min(nxt, max(step_block_end, ce))

    # If the next event is a clean elastic_scale_up right as the last step
    # finishes, align step_block_end with nxt so no bogus Interruption gap is
    # created.
    if (
        nxt in scale_up_starts
        and my_steps[-1] < nxt <= my_steps[-1] + max(step_dur * 1.5, 90.0)
        and nxt >= step_block_end
    ):
      step_block_end = nxt

    # Synthesize JIT Compile Overhead sub-event across initial 3 steps. A step
    # is compared with the smaller of its expected time and the run's lower
    # median, or with the latter if its expected time is unknown.
    run_jit_segs: list[Segment] = []
    if run_deltas:
      for sd in run_deltas[:3]:
        expected = baseline.expected_s(sd.start, sd.end)
        ref_base = min(expected, run_base) if expected else run_base
        if sd.delta_s > ref_base * 1.15:
          jit_seg = Segment(
              kind=events.EventType.JIT_COMPILE,
              start=sd.start,
              end=sd.start + (sd.delta_s - ref_base),
              run_idx=current_run_idx,
              label='JIT Compile Overhead',
          )
          segments.append(jit_seg)
          run_jit_segs.append(jit_seg)

    repeated_pairs = (
        [(sn, ts) for sn, ts in my_step_pairs if sn <= max_step_before_run]
        if current_run_idx > 0 and max_step_before_run >= 0
        else []
    )
    productive_pairs = (
        [(sn, ts) for sn, ts in my_step_pairs if sn > max_step_before_run]
        if current_run_idx > 0 and max_step_before_run >= 0
        else my_step_pairs
    )

    if repeated_pairs:
      wasted_start = repeated_pairs[0][1]
      wasted_end = (
          productive_pairs[0][1] if productive_pairs else step_block_end
      )
      r_first, r_last = repeated_pairs[0][0], repeated_pairs[-1][0]

      # 1. Outer container segment: Wasted Progress (holding container)
      segments.append(
          Segment(
              kind=events.EventType.WASTED_PROGRESS,
              start=wasted_start,
              end=wasted_end,
              run_idx=current_run_idx,
              label=(
                  f'Wasted Progress ({len(repeated_pairs)} repeated steps:'
                  f' {r_first}\u2013{r_last})'
              ),
              meta={
                  'step_range': (r_first, r_last),
                  'count': len(repeated_pairs),
              },
          )
      )

      # 2. Inner sub-events: Lost Progress (pure repeated step execution,
      # excluding JIT Compile Overhead). Checkpoint Save and custom badput
      # events are drawn as overlay sub-events and stay counted as Lost
      # Progress.
      lost_intervals = _subtract_intervals(
          wasted_start, wasted_end, [(s.start, s.end) for s in run_jit_segs]
      )

      def _step_end(i: int) -> float:
        if i + 1 < len(repeated_pairs):
          return repeated_pairs[i + 1][1]
        elif productive_pairs:
          return productive_pairs[0][1]
        else:
          return step_block_end

      for seg_st, seg_ed in lost_intervals:
        steps_in_seg = [
            sn
            for idx, (sn, ts) in enumerate(repeated_pairs)
            if max(ts, seg_st) < min(_step_end(idx), seg_ed) - 1e-3
        ]
        if steps_in_seg:
          s0, s1 = steps_in_seg[0], steps_in_seg[-1]
          cnt = len(steps_in_seg)
        else:
          s0, s1 = r_first, r_last
          cnt = len(repeated_pairs) if len(lost_intervals) == 1 else 0

        if len(lost_intervals) == 1:
          lbl = (
              f'Lost Progress ({len(repeated_pairs)} repeated steps:'
              f' {r_first}\u2013{r_last})'
          )
        elif s0 == s1:
          lbl = f'Lost Progress (repeated step {s0})'
        else:
          lbl = f'Lost Progress ({cnt} repeated steps: {s0}\u2013{s1})'

        segments.append(
            Segment(
                kind=events.EventType.LOST_PROGRESS,
                start=seg_st,
                end=seg_ed,
                run_idx=current_run_idx,
                label=lbl,
                meta={
                    'step_range': (s0, s1),
                    'count': cnt,
                },
            )
        )

    if productive_pairs:
      prod_start = productive_pairs[0][1]
      p_first, p_last = productive_pairs[0][0], productive_pairs[-1][0]
      # 1. Outer container segment: Stepping (holding container), the span of
      # the productive steps including the JIT compile overhead of the first
      # steps after a (re)start.
      segments.append(
          Segment(
              kind=events.EventType.STEP_BLOCK,
              start=prod_start,
              end=step_block_end,
              run_idx=current_run_idx,
              label=(
                  f'Stepping ({len(productive_pairs)} steps:'
                  f' {p_first}\u2013{p_last})'
              ),
              meta={
                  'step_range': (p_first, p_last),
                  'count': len(productive_pairs),
              },
          )
      )

      # 2. Inner sub-events: Productive Steps (the stepping span excluding
      # JIT Compile Overhead), one segment per stretch between JIT intervals,
      # like Lost Progress inside Wasted Progress.
      productive_intervals = _subtract_intervals(
          prod_start, step_block_end, [(s.start, s.end) for s in run_jit_segs]
      )

      def _productive_step_end(i: int) -> float:
        if i + 1 < len(productive_pairs):
          return productive_pairs[i + 1][1]
        return step_block_end

      for seg_st, seg_ed in productive_intervals:
        steps_in_seg = [
            sn
            for idx, (sn, ts) in enumerate(productive_pairs)
            if max(ts, seg_st) < min(_productive_step_end(idx), seg_ed) - 1e-3
        ]
        if steps_in_seg:
          s0, s1 = steps_in_seg[0], steps_in_seg[-1]
          cnt = len(steps_in_seg)
        else:
          s0, s1 = p_first, p_last
          cnt = len(productive_pairs) if len(productive_intervals) == 1 else 0

        if len(productive_intervals) == 1:
          lbl = (
              f'Productive Steps ({len(productive_pairs)} steps:'
              f' {p_first}\u2013{p_last})'
          )
        elif s0 == s1:
          lbl = f'Productive Steps (step {s0})'
        else:
          lbl = f'Productive Steps ({cnt} steps: {s0}\u2013{s1})'

        segments.append(
            Segment(
                kind=events.EventType.PRODUCTIVE_STEPS,
                start=seg_st,
                end=seg_ed,
                run_idx=current_run_idx,
                label=lbl,
                meta={
                    'step_range': (s0, s1),
                    'count': cnt,
                },
            )
        )

    max_step_before_run = max(
        max_step_before_run, max(sn for sn, _ in my_step_pairs)
    )

    # 3. Interruption (follows stepping when there is an uninstrumented gap
    # before nxt)
    if nxt < float('inf') and nxt - step_block_end > 1e-3:
      segments.append(
          Segment(
              kind=events.EventType.INTERRUPTION,
              start=step_block_end,
              end=nxt,
              run_idx=current_run_idx,
              label='Interruption',
              meta={'last_step': max_step_before_run},
          )
      )

    run_end_boundaries.append((current_run_idx, step_block_end))
    current_run_idx += 1

  def _resolve_run_idx(ev_ts: float) -> int:
    for r_idx, r_end in run_end_boundaries:
      if ev_ts < r_end - 1e-3:
        return r_idx
    return run_end_boundaries[-1][0] if run_end_boundaries else 0

  # 4. Append underlying sub-patterns precisely within their intervals
  for kind, intervals, lbl in [
      (events.EventType.TPU_INIT, parsed.tpu_init, 'Device Init (TPU)'),
      (
          events.EventType.TRAINING_PREP,
          parsed.training_prep,
          'Training Prep',
      ),
      (events.EventType.DATA_LOADING, parsed.data_loading, 'Data Loading'),
      (
          events.EventType.CHECKPOINT_SAVE,
          parsed.checkpoint_save,
          'Checkpoint Save',
      ),
      (
          events.EventType.CHECKPOINT_RESTORE,
          [
              i
              for i in parsed.checkpoint_restore
              if i not in parsed.snapshot_restore
              and i not in parsed.snapshot_save
          ],
          'Checkpoint Restore',
      ),
      (
          events.EventType.CHECKPOINT_RESTORE,
          parsed.snapshot_restore,
          'Snapshot Restore',
      ),
      (
          events.EventType.SNAPSHOT_SAVE,
          parsed.snapshot_save,
          'Snapshot Save',
      ),
      (
          events.EventType.ELASTIC_SCALE_UP,
          parsed.elastic_scale_up,
          'Elastic Scale Up',
      ),
      (
          events.EventType.ELASTIC_SLICE_DOWN,
          parsed.elastic_slice_down,
          'Elastic Slice Down',
      ),
      (
          events.EventType.ELASTIC_REINIT,
          parsed.elastic_reinitialization,
          'Elastic Reinitialization',
      ),
  ]:
    is_wait = kind in (
        events.EventType.ELASTIC_SCALE_UP,
        events.EventType.ELASTIC_SLICE_DOWN,
    )
    for st, ed in intervals:
      ref_ts = ed if is_wait else st
      target_run_idx = _resolve_run_idx(ref_ts)
      segments.append(
          Segment(
              kind=kind, start=st, end=ed, run_idx=target_run_idx, label=lbl
          )
      )

  for evt_type, intervals in parsed.custom.items():
    for st, ed in intervals:
      target_run_idx = _resolve_run_idx(st)
      segments.append(
          Segment(
              kind=events.EventType.CUSTOM,
              start=st,
              end=ed,
              run_idx=target_run_idx,
              label=evt_type,
          )
      )

  return sorted(segments, key=lambda s: s.start)
