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

"""Metrics calculation for goodput timeline using cloud_goodput library."""

from collections.abc import Collection
import dataclasses
import datetime
import tempfile
from typing import Any, Optional, Union
from cloud_goodput.ml_goodput_measurement.src import goodput
from cloud_goodput.ml_goodput_measurement.src import goodput_elastic
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.timeline import event_parser
from cloud_goodput.ml_goodput_measurement.timeline import step_times
from cloud_goodput.ml_goodput_measurement.timeline.builder import Segment
from cloud_goodput.ml_goodput_measurement.timeline.events import EventType
import pandas as pd

# Alias for convenience
UnproductiveTimeDict = goodput.UnproductiveTimeDict

_STARTUP_COMPILATION_WINDOW = 3


class MockCloudLogger:
  """Mock cloud logger to feed pre-parsed entries to GoodputCalculator."""

  def __init__(self, entries: list[dict[str, Any]]):
    self.entries = entries

  def read_cloud_logging_entries(
      self,
      start_time: Optional[datetime.datetime] = None,
      end_time: Optional[datetime.datetime] = None,
      last_entry_info: Optional[tuple[datetime.datetime, str]] = None,
  ) -> tuple[list[dict[str, Any]], None]:
    """Reads entries matching optional time range filters."""
    del last_entry_info
    # Filter entries by time if specified
    filtered = []
    for entry in self.entries:
      ts = goodput_utils.get_timestamp_from_log_entry(entry)
      if ts is not None:
        # Ensure ts is timezone-aware UTC for comparison
        if ts.tzinfo is None:
          ts = ts.replace(tzinfo=datetime.timezone.utc)
        if start_time:
          start_aware = (
              start_time.replace(tzinfo=datetime.timezone.utc)
              if start_time.tzinfo is None
              else start_time
          )
          if ts <= start_aware:
            continue
        if end_time:
          end_aware = (
              end_time.replace(tzinfo=datetime.timezone.utc)
              if end_time.tzinfo is None
              else end_time
          )
          if ts > end_aware:
            continue
      filtered.append(entry)
    return filtered, None

  def read_new_entries(
      self,
      last_entry_info: Optional[tuple[datetime.datetime, str]],
      end_time: Optional[datetime.datetime],
      overlap_seconds: float,
      known_entry_ids: Collection[str],
  ) -> tuple[
      list[dict[str, Any]],
      tuple[Optional[datetime.datetime], Optional[str]],
      list[tuple[str, datetime.datetime]],
  ]:
    """Reads the entries after the cursor, like `_CloudLogger.read_new_entries`.

    The entries have no insertId, so an entry's index in `entries` stands in
    for it. An entry without a timestamp counts as logged at the epoch, so only
    the first read, which has no cursor, returns it.

    Args:
      last_entry_info: `(timestamp, insert_id)` of the newest entry read so far,
        or None to read all entries.
      end_time: Inclusive upper bound of the timestamps, or None for no bound.
      overlap_seconds: How far before the cursor to read again.
      known_entry_ids: Ids of entries read before; they are skipped.

    Returns:
      The new entries in `entries` order, the cursor advanced to the newest of
      them, and `(insert_id, timestamp)` of each new entry.
    """
    start_time = None
    if last_entry_info is not None:
      last_entry_info = (_as_utc(last_entry_info[0]), last_entry_info[1])
      start_time = last_entry_info[0] - datetime.timedelta(
          seconds=overlap_seconds
      )
    if end_time is not None:
      end_time = _as_utc(end_time)

    cursor = last_entry_info
    payloads = []
    entry_ids = []
    for index, entry in enumerate(self.entries):
      insert_id = str(index)
      ts = goodput_utils.get_timestamp_from_log_entry(entry)
      ts = _EPOCH if ts is None else _as_utc(ts)
      if insert_id in known_entry_ids or (ts, insert_id) == last_entry_info:
        continue
      if start_time is not None and ts <= start_time:
        continue
      if end_time is not None and ts > end_time:
        continue
      payloads.append(entry)
      entry_ids.append((insert_id, ts))
      if cursor is None or (ts, insert_id) > (cursor[0], cursor[1] or ''):
        cursor = (ts, insert_id)
    return payloads, cursor or (None, None), entry_ids


_EPOCH = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)


def _as_utc(t: datetime.datetime) -> datetime.datetime:
  """Returns `t`, as UTC if it has no time zone."""
  return t.replace(tzinfo=datetime.timezone.utc) if t.tzinfo is None else t


def _is_elastic_workload(entries: list[dict[str, Any]]) -> bool:
  for e in entries:
    if 'elastic_wait_start_time' in e or 'elastic_reinit_start_time' in e:
      return True
    if 'custom_badput_event_type' in e and e['custom_badput_event_type'] in [
        'elastic_scale_up',
        'elastic_slice_down',
        'elastic_reinitialization',
    ]:
      return True
  return False


def _reconstruct_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
  """Reconstructs raw entries from a DataFrame of LogRecords."""
  entries = []
  for _, row in df.iterrows():
    # Only process GOODPUT source or rows with GOODPUT event types
    event_type = row.get('event_type')
    if not isinstance(event_type, str) or not event_type.startswith('GOODPUT_'):
      continue

    ts_sec = row['timestamp'] / 1e6

    entry = {}
    if event_type == 'GOODPUT_JOB_START':
      entry['job_start_time'] = ts_sec
    elif event_type == 'GOODPUT_JOB_END':
      entry['job_end_time'] = ts_sec
    elif event_type == 'GOODPUT_JOB_DISRUPTION':
      entry['job_disruption_time'] = ts_sec
    elif event_type == 'GOODPUT_STEP_START':
      # Need both start time and step count
      if 'step' in row and pd.notna(row['step']):
        entry['step_start_time'] = ts_sec
        entry['step_count'] = int(row['step'])
    elif event_type == 'GOODPUT_TPU_INIT_START':
      entry['tpu_init_start_time'] = ts_sec
    elif event_type == 'GOODPUT_TPU_INIT_END':
      entry['tpu_init_end_time'] = ts_sec
    elif event_type == 'GOODPUT_TRAINING_PREP_START':
      entry['training_prep_start_time'] = ts_sec
    elif event_type == 'GOODPUT_TRAINING_PREP_END':
      entry['training_prep_end_time'] = ts_sec
    elif event_type == 'GOODPUT_DATA_ITERATOR_INIT_START':
      entry['data_iterator_init_start_time'] = ts_sec
    elif event_type == 'GOODPUT_DATA_ITERATOR_INIT_END':
      entry['data_iterator_init_end_time'] = ts_sec
    elif event_type == 'GOODPUT_COMPILATION_START':
      entry['compilation_start_time'] = ts_sec
    elif event_type == 'GOODPUT_COMPILATION_END':
      entry['compilation_end_time'] = ts_sec
    elif event_type == 'GOODPUT_DATA_LOADING_START':
      entry['data_loading_start_time'] = ts_sec
    elif event_type == 'GOODPUT_DATA_LOADING_END':
      entry['data_loading_end_time'] = ts_sec
    elif event_type == 'GOODPUT_CHECKPOINT_RESTORE_START':
      if 'goodput_duration_secs' in row and pd.notna(
          row['goodput_duration_secs']
      ):
        entry['checkpoint_manager_start_time'] = ts_sec
        entry['checkpoint_manager_duration_secs'] = row['goodput_duration_secs']
      else:
        entry['checkpoint_restore_start_time'] = ts_sec
    elif event_type == 'GOODPUT_CHECKPOINT_RESTORE_END':
      entry['checkpoint_restore_end_time'] = ts_sec
    elif event_type == 'GOODPUT_CHECKPOINT_SAVE_START':
      # Need both start and duration
      if 'goodput_duration_secs' in row and pd.notna(
          row['goodput_duration_secs']
      ):
        entry['checkpoint_manager_blocking_start_time'] = ts_sec
        entry['checkpoint_manager_blocking_duration_secs'] = row[
            'goodput_duration_secs'
        ]
    elif event_type == 'GOODPUT_ELASTIC_SCALE_UP_START':
      entry['elastic_wait_start_time'] = ts_sec
      entry['elastic_wait_event_type'] = 'elastic_scale_up'
    elif event_type == 'GOODPUT_ELASTIC_SCALE_UP_END':
      entry['elastic_wait_end_time'] = ts_sec
      entry['elastic_wait_event_type'] = 'elastic_scale_up'
    elif event_type == 'GOODPUT_ELASTIC_SLICE_DOWN_START':
      entry['elastic_wait_start_time'] = ts_sec
      entry['elastic_wait_event_type'] = 'elastic_slice_down'
    elif event_type == 'GOODPUT_ELASTIC_SLICE_DOWN_END':
      entry['elastic_wait_end_time'] = ts_sec
      entry['elastic_wait_event_type'] = 'elastic_slice_down'
    elif event_type == 'GOODPUT_ELASTIC_REINITIALIZATION_START':
      entry['elastic_reinit_start_time'] = ts_sec
    elif event_type == 'GOODPUT_ELASTIC_REINITIALIZATION_END':
      entry['elastic_reinit_end_time'] = ts_sec
    elif event_type == 'GOODPUT_CUSTOM_BADPUT_START':
      # Need both start and custom type
      if 'custom_badput_event_type' in row and pd.notna(
          row['custom_badput_event_type']
      ):
        entry['custom_badput_event_start_time'] = ts_sec
        entry['custom_badput_event_type'] = row['custom_badput_event_type']
    elif event_type == 'GOODPUT_CUSTOM_BADPUT_END':
      # Need both end and custom type
      if 'custom_badput_event_type' in row and pd.notna(
          row['custom_badput_event_type']
      ):
        entry['custom_badput_event_end_time'] = ts_sec
        entry['custom_badput_event_type'] = row['custom_badput_event_type']

    if entry:
      entries.append(entry)
  return entries


# Suffixes of duration fields whose names contain 'time', e.g. the
# time_between_consecutive_saves_sec field of checkpoint logs.
_DURATION_SUFFIXES = ('_sec', '_secs', '_seconds')


def _find_min_max_timestamps(
    entries: list[dict[str, Any]],
) -> tuple[Optional[float], Optional[float]]:
  """Finds the minimum and maximum timestamp values across all entries.

  Durations are skipped: taken for the job start, a duration of a few seconds
  would put the start in 1970.
  """
  min_ts = None
  max_ts = None
  for e in entries:
    for k, v in e.items():
      if (
          'time' in k
          and not k.endswith(_DURATION_SUFFIXES)
          and isinstance(v, (int, float))
      ):
        if min_ts is None or v < min_ts:
          min_ts = v
        if max_ts is None or v > max_ts:
          max_ts = v
  return min_ts, max_ts


def to_raw_entries(
    entries: Union[list[dict[str, Any]], pd.DataFrame],
) -> list[dict[str, Any]]:
  """Goodput log payloads of entries; a DataFrame of LogRecords is converted."""
  if isinstance(entries, pd.DataFrame):
    return _reconstruct_from_dataframe(entries.sort_values('timestamp'))
  return entries


def calculate_goodput_metrics(
    entries: Union[list[dict[str, Any]], pd.DataFrame],
    job_name: str,
) -> tuple[float, UnproductiveTimeDict, int, float]:
  """Computes Goodput and Badput breakdown using cloud_goodput library."""
  raw_entries = to_raw_entries(entries)

  has_start = any('job_start_time' in e for e in raw_entries)
  has_end = any('job_end_time' in e for e in raw_entries)

  if (not has_start or not has_end) and raw_entries:
    min_ts, max_ts = _find_min_max_timestamps(raw_entries)
    if not has_start and min_ts is not None:
      raw_entries.append({'job_start_time': min_ts})
    if not has_end and max_ts is not None:
      raw_entries.append({'job_end_time': max_ts})

  mock_logger = MockCloudLogger(raw_entries)
  is_elastic = _is_elastic_workload(raw_entries)

  with tempfile.TemporaryDirectory() as temp_dir:
    if is_elastic:
      calculator = goodput_elastic.ElasticGoodputCalculator(
          job_name=job_name,
          logger_name=f'goodput_{job_name}',
          cloud_logger=mock_logger,  # type: ignore[arg-type]
          using_pathways=True,
          cache_dir=temp_dir,
      )
    else:
      calculator = goodput.GoodputCalculator(
          job_name=job_name,
          logger_name=f'goodput_{job_name}',
          cloud_logger=mock_logger,  # type: ignore[arg-type]
          cache_dir=temp_dir,
      )

    # Errors propagate: compare shows Goodput as unavailable and the timeline
    # page shows the error, rather than a misleading Goodput of 0.
    goodput_val, badput_breakdown, last_step = calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    with calculator._goodput_cache_lock:  # pylint: disable=protected-access
      info = calculator._goodput_cache.get_goodput_info()  # pylint: disable=protected-access
      total_job_time = info.total_elapsed_time if info else 0.0
    return goodput_val, badput_breakdown, last_step, total_job_time


def stepping_windows(
    segments: list[Segment],
) -> dict[int, tuple[float, float]]:
  """The span of each run's Productive Steps and Wasted Progress, by run_idx."""
  windows: dict[int, tuple[float, float]] = {}
  for s in segments:
    if s.kind in (EventType.STEP_BLOCK, EventType.WASTED_PROGRESS):
      lo, hi = windows.get(s.run_idx, (s.start, s.end))
      windows[s.run_idx] = (min(lo, s.start), max(hi, s.end))
  return windows


def calculate_timeline_metrics(segments: list[Segment]) -> dict[str, float]:
  """Calculates MTTR and MTBI from timeline segments."""
  windows = sorted(stepping_windows(segments).values(), key=lambda w: w[0])

  ttrs = []
  for i in range(1, len(windows)):
    prev_end = windows[i - 1][1]
    curr_start = windows[i][0]
    if curr_start > prev_end:
      ttrs.append(curr_start - prev_end)

  mttr = sum(ttrs) / len(ttrs) if ttrs else 0.0

  mtbi = sum(e - s for s, e in windows) / len(windows) if windows else 0.0

  return {
      'mttr': mttr,
      'mtbi': mtbi,
  }


RECOVERY_PHASE_ORDER = [
    'fault_detection_teardown',
    'slice_wait_reschedule',
    'pathways_cluster_init',
    'tpu_mesh_model_init',
    'data_iterator_init',
    'checkpoint_restore',
    'aot_xla_compilation',
    'first_batch_data_loading',
    'jit_compile_overhead',
    'lost_progress',
]

RECOVERY_PHASE_LABELS = {
    'fault_detection_teardown': '1. Fault Detection / Teardown Gap',
    'slice_wait_reschedule': '2. Slice Wait / Infra Reschedule',
    'pathways_cluster_init': '3. Pathways Client & Cluster Barrier Init',
    'tpu_mesh_model_init': '4. TPU Mesh & Model Init',
    'data_iterator_init': '5. Data Pipeline (RemoteIterator) Init',
    'checkpoint_restore': '6. Checkpoint / Snapshot Restore',
    'aot_xla_compilation': '7. AOT XLA Compilation & JIT Lowering',
    'first_batch_data_loading': '8. First Batch Data Loading',
    'jit_compile_overhead': '9. JIT Compile Overhead',
    'lost_progress': '10. Lost Progress (Repeated Steps, excl. JIT compile)',
}


@dataclasses.dataclass
class RecoveryIncident:
  """Detailed breakdown of a single recovery incident between step blocks."""

  incident_idx: int
  start_ts: float
  end_ts: float
  downtime_s: float
  lost_progress_s: float
  rolled_back_steps: int
  mttr_s: float = 0.0
  phases: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RecoverySummaryStats:
  """Summary statistics across all recovery incidents in a workload."""

  num_incidents: int = 0
  mean_mttr_s: float = 0.0
  min_mttr_s: float = 0.0
  max_mttr_s: float = 0.0
  mean_downtime_s: float = 0.0
  min_downtime_s: float = 0.0
  max_downtime_s: float = 0.0
  mean_lost_progress_s: float = 0.0
  total_lost_progress_s: float = 0.0
  mean_rolled_back_steps: float = 0.0
  # Steady-state step time (lower median of consecutive step deltas; for
  # elastic workloads, of the steps with all slices active; see
  # step_times.Baseline).
  baseline_step_time_s: float = 0.0
  phase_stats: dict[str, dict[str, float]] = dataclasses.field(
      default_factory=dict
  )
  incidents: list[RecoveryIncident] = dataclasses.field(default_factory=list)


def _overlap_duration(
    intervals: list[tuple[float, float]], window_start: float, window_end: float
) -> float:
  """Computes total overlap of intervals within [window_start, window_end]."""
  total = 0.0
  for s, e in intervals:
    ov_s = max(s, window_start)
    ov_e = min(e, window_end)
    if ov_e > ov_s:
      total += ov_e - ov_s
  return total


def _overlap_between_interval_lists(
    list_a: list[tuple[float, float]],
    list_b: list[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> float:
  """Computes total overlap between two interval lists in a time window."""
  total = 0.0
  for sa, ea in list_a:
    ca_s = max(sa, window_start)
    ca_e = min(ea, window_end)
    if ca_e <= ca_s:
      continue
    for sb, eb in list_b:
      ov_s = max(ca_s, sb)
      ov_e = min(ca_e, eb)
      if ov_e > ov_s:
        total += ov_e - ov_s
  return total


def extract_recovery_incidents(
    entries: Union[list[dict[str, Any]], pd.DataFrame],
    segments: list[Segment],
) -> RecoverySummaryStats:
  """Extracts all recovery incidents and decomposes each into 10 sub-phases.

  Without two runs there are no incidents; the returned stats then only have
  the baseline step time.

  Args:
    entries: Goodput log payloads, or a DataFrame of LogRecords.
    segments: The timeline of the workload, see builder.build_timeline.

  Returns:
    The recovery stats.
  """
  parsed = event_parser.parse_entries(to_raw_entries(entries))
  baseline = step_times.Baseline(parsed.steps, parsed.elastic_slice_counts)
  # 0.0 when no two consecutive steps are known.
  baseline_step_time = baseline.step_time_s

  run_step_bounds = stepping_windows(segments)
  sorted_runs = sorted(run_step_bounds)
  if len(sorted_runs) < 2:
    return RecoverySummaryStats(baseline_step_time_s=baseline_step_time)

  incidents: list[RecoveryIncident] = []
  all_wait = sorted(parsed.elastic_scale_up + parsed.elastic_slice_down)

  for idx in range(1, len(sorted_runs)):
    prev_run = sorted_runs[idx - 1]
    curr_run = sorted_runs[idx]
    prev_start, t_crash = run_step_bounds[prev_run]
    t_resume, curr_end = run_step_bounds[curr_run]
    if t_resume <= t_crash:
      continue

    downtime_s = t_resume - t_crash
    w_pad_start = t_crash - 5.0
    w_pad_end = t_resume + 5.0

    # Find sub-events inside this recovery window
    waits = [
        (s, e) for s, e in all_wait if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    reinits = [
        (s, e)
        for s, e in parsed.elastic_reinitialization
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    tpus = [
        (s, e)
        for s, e in parsed.tpu_init
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    preps = [
        (s, e)
        for s, e in parsed.training_prep
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    data_iters = [
        (s, e)
        for s, e in parsed.data_iterator_init
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    ckpts = [
        (s, e)
        for s, e in parsed.checkpoint_restore
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    comps = [
        (s, e)
        for s, e in parsed.compilation
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    dloads = [
        (s, e)
        for s, e in parsed.data_loading
        if max(s, w_pad_start) < min(e, w_pad_end)
    ]
    j_starts = [t for t in parsed.job_starts if t_crash <= t <= t_resume]
    j_disrupts = [
        t for t in parsed.job_disruptions if w_pad_start <= t <= t_resume
    ]

    # Phase 4: TPU Mesh & Model Init
    tpu_dur = _overlap_duration(tpus, t_crash, t_resume)

    # Phase 6: Checkpoint / Snapshot Restore & Phase 5: Data Iterator Init
    prep_dur = _overlap_duration(preps, t_crash, t_resume)
    ckpt_dur = _overlap_duration(ckpts, t_crash, t_resume)
    data_iter_dur = _overlap_duration(data_iters, t_crash, t_resume)

    if data_iter_dur == 0 and prep_dur > 0:
      ckpt_in_prep = _overlap_between_interval_lists(
          preps, ckpts, t_crash, t_resume
      )
      dl_in_prep = _overlap_between_interval_lists(
          preps, dloads, t_crash, t_resume
      )
      comp_in_prep = _overlap_between_interval_lists(
          preps, comps, t_crash, t_resume
      )
      data_iter_dur = max(
          0.0, prep_dur - ckpt_in_prep - dl_in_prep - comp_in_prep
      )
    elif ckpt_dur == 0 and prep_dur > 0 and data_iter_dur > 0:
      ckpt_dur = max(0.0, prep_dur - data_iter_dur)
    elif data_iters and ckpts:
      ov_di_ckpt = _overlap_between_interval_lists(
          data_iters, ckpts, t_crash, t_resume
      )
      data_iter_dur = max(0.0, data_iter_dur - ov_di_ckpt)

    # Phase 7: AOT XLA Compilation
    comp_dur = _overlap_duration(comps, t_crash, t_resume)
    if comp_dur == 0:
      aot_segs = [
          (s.start, s.end)
          for s in segments
          if s.kind == EventType.AOT_COMPILE
          and max(s.start, t_crash) < min(s.end, t_resume)
      ]
      if aot_segs:
        comp_dur = _overlap_duration(aot_segs, t_crash, t_resume)
      elif preps:
        prep_end = preps[-1][1]
        if reinits:
          reinit_end = reinits[-1][1]
          if reinit_end > prep_end:
            aot_start = max(t_crash, prep_end)
            aot_end = min(t_resume, reinit_end)
            ckpt_in_aot = _overlap_duration(ckpts, aot_start, aot_end)
            comp_dur = max(0.0, (aot_end - aot_start) - ckpt_in_aot)
        else:
          next_ev = dloads[0][0] if dloads else t_resume
          if next_ev > prep_end:
            aot_start = max(t_crash, prep_end)
            aot_end = min(t_resume, next_ev)
            ckpt_in_aot = _overlap_duration(ckpts, aot_start, aot_end)
            comp_dur = max(0.0, (aot_end - aot_start) - ckpt_in_aot)
    comp_dur = max(0.0, comp_dur)

    # Phase 8: First Batch Data Loading
    dl_dur = _overlap_duration(dloads, t_crash, t_resume)

    # Phase 9: JIT Compile Overhead across initial _STARTUP_COMPILATION_WINDOW
    # steps after start/restart
    jit_segs_for_run = [
        s
        for s in segments
        if s.kind == EventType.JIT_COMPILE and s.run_idx == curr_run
    ]
    steps_after = [
        (s_num, ts)
        for s_num, ts in parsed.steps
        if t_resume - 1.0 <= ts <= curr_end + 1.0
    ]
    if jit_segs_for_run:
      jit_compile = sum(s.duration_s for s in jit_segs_for_run)
    else:
      jit_compile = 0.0
      run_deltas = step_times.consecutive_step_deltas(steps_after)
      if run_deltas:
        run_baseline = step_times.lower_median([d.delta_s for d in run_deltas])
        for d in run_deltas[:_STARTUP_COMPILATION_WINDOW]:
          # The smaller of the step's expected time and the run's lower
          # median, or the latter if the expected time is unknown.
          expected = baseline.expected_s(d.start, d.end)
          ref_baseline = (
              min(expected, run_baseline) if expected else run_baseline
          )
          if d.delta_s > ref_baseline * 1.15:
            jit_compile += max(0.0, d.delta_s - ref_baseline)

    # Pre-TPU-init phases (Phases 1, 2, 3) inside [t_crash, t_tpu_start]
    t_tpu_start = (
        tpus[0][0]
        if tpus
        else (preps[0][0] if preps else (dloads[0][0] if dloads else t_resume))
    )
    t_tpu_start = min(t_resume, max(t_crash, t_tpu_start))
    pre_tpu_span = max(0.0, t_tpu_start - t_crash)

    first_elastic_ts = (
        waits[0][0] if waits else (reinits[0][0] if reinits else float('inf'))
    )
    pod_restart_before_elastic = bool(
        j_starts and j_starts[0] < first_elastic_ts
    )

    if (waits or reinits) and not pod_restart_before_elastic:
      t_wait_start = (
          waits[0][0] if waits else (reinits[0][0] if reinits else t_crash)
      )
      t_wait_end = waits[-1][1] if waits else t_wait_start
      t_wait_start = min(t_tpu_start, max(t_crash, t_wait_start))
      t_wait_end = min(t_tpu_start, max(t_wait_start, t_wait_end))

      fault_dur = max(0.0, t_wait_start - t_crash)
      slice_wait_dur = _overlap_duration(waits, t_wait_start, t_wait_end)
      if slice_wait_dur == 0:
        slice_wait_dur = max(0.0, t_wait_end - t_wait_start)
      cluster_init_dur = max(0.0, pre_tpu_span - fault_dur - slice_wait_dur)
    else:
      t_disrupt = j_disrupts[0] if j_disrupts else t_crash
      t_disrupt = min(t_tpu_start, max(t_crash, t_disrupt))
      fault_dur = max(0.0, t_disrupt - t_crash)
      if j_starts:
        t_js = min(t_tpu_start, max(t_disrupt, j_starts[0]))
        slice_wait_dur = max(0.0, t_js - t_disrupt)
        cluster_init_dur = max(0.0, t_tpu_start - t_js)
      else:
        slice_wait_dur = max(0.0, t_tpu_start - t_disrupt)
        cluster_init_dur = 0.0

    # Positionally reconcile any uninstrumented gap or overlap inside the
    # post-TPU window [t_tpu_start, t_resume] so Phases 1-8 sum to downtime_s.
    post_tpu_span = max(0.0, t_resume - t_tpu_start)
    post_accounted = tpu_dur + data_iter_dur + ckpt_dur + comp_dur + dl_dur
    post_residual = post_tpu_span - post_accounted
    if post_residual > 0:
      t_tpu_end = (
          min(t_resume, max(t_tpu_start, tpus[-1][1])) if tpus else t_tpu_start
      )
      post_tpu_starts = [
          max(t_tpu_end, iv[0])
          for iv in preps + data_iters + ckpts + comps + dloads
          if iv[1] > t_tpu_end
      ]
      if post_tpu_starts:
        gap_after_tpu = max(0.0, min(post_tpu_starts) - t_tpu_end)
        add_tpu = min(post_residual, gap_after_tpu)
        tpu_dur += add_tpu
        comp_dur += post_residual - add_tpu
      else:
        comp_dur += post_residual
    elif post_residual < 0:
      rem_neg = abs(post_residual)
      for attr_val in (
          'data_iter_dur',
          'comp_dur',
          'ckpt_dur',
          'dl_dur',
          'tpu_dur',
      ):
        if rem_neg <= 0:
          break
        cur_v = locals()[attr_val]
        deduct = min(cur_v, rem_neg)
        if attr_val == 'data_iter_dur':
          data_iter_dur -= deduct
        elif attr_val == 'comp_dur':
          comp_dur -= deduct
        elif attr_val == 'ckpt_dur':
          ckpt_dur -= deduct
        elif attr_val == 'dl_dur':
          dl_dur -= deduct
        elif attr_val == 'tpu_dur':
          tpu_dur -= deduct
        rem_neg -= deduct

    # Phase 10: Lost Progress (Repeated Steps, excl. JIT compile)
    lost_progress_s = 0.0
    rolled_back_steps = 0
    all_steps_before = [
        (s_num, ts) for s_num, ts in parsed.steps if ts <= t_crash + 1.0
    ]
    steps_before = [
        (s_num, ts)
        for s_num, ts in parsed.steps
        if prev_start - 1.0 <= ts <= t_crash + 1.0
    ]
    lost_segs = [
        s
        for s in segments
        if s.kind == EventType.LOST_PROGRESS and s.run_idx == curr_run
    ]
    wasted_segs = [
        s
        for s in segments
        if s.kind == EventType.WASTED_PROGRESS and s.run_idx == curr_run
    ]
    if wasted_segs:
      rolled_back_steps = sum(int(s.meta.get('count', 0)) for s in wasted_segs)
    elif lost_segs:
      rolled_back_steps = sum(int(s.meta.get('count', 0)) for s in lost_segs)

    if lost_segs:
      lost_progress_s = sum(s.duration_s for s in lost_segs)
    elif wasted_segs:
      wasted_total = sum(s.duration_s for s in wasted_segs)
      jit_segs = [(s.start, s.end) for s in jit_segs_for_run]
      wasted_intervals = [(s.start, s.end) for s in wasted_segs]
      jit_in_wasted = _overlap_between_interval_lists(
          wasted_intervals, jit_segs, t_resume - 1.0, curr_end + 1.0
      )
      lost_progress_s = max(0.0, wasted_total - jit_in_wasted)
    elif all_steps_before and steps_after:
      last_step_before = max(s_num for s_num, _ in all_steps_before)
      first_step_after, _ = steps_after[0]
      if first_step_after <= last_step_before:
        rolled_back_steps = last_step_before - first_step_after + 1
        rollback_matches = [
            ts for s_num, ts in steps_before if s_num >= first_step_after
        ]
        if rollback_matches:
          lost_progress_s = max(0.0, t_crash - rollback_matches[0])
        else:
          # Unknown baseline: assume 1s per step.
          lost_progress_s = rolled_back_steps * (baseline_step_time or 1.0)

    # Total Time to Recover (MTTR) includes Pre-Step Downtime + JIT Compile
    # Overhead + Lost Progress
    total_mttr_s = downtime_s + jit_compile + lost_progress_s

    phases_dict = {
        'fault_detection_teardown': fault_dur,
        'slice_wait_reschedule': slice_wait_dur,
        'pathways_cluster_init': cluster_init_dur,
        'tpu_mesh_model_init': tpu_dur,
        'data_iterator_init': data_iter_dur,
        'checkpoint_restore': ckpt_dur,
        'aot_xla_compilation': comp_dur,
        'first_batch_data_loading': dl_dur,
        'jit_compile_overhead': jit_compile,
        'lost_progress': lost_progress_s,
    }
    incidents.append(
        RecoveryIncident(
            incident_idx=curr_run,
            start_ts=t_crash,
            end_ts=t_resume,
            downtime_s=downtime_s,
            lost_progress_s=lost_progress_s,
            rolled_back_steps=rolled_back_steps,
            mttr_s=total_mttr_s,
            phases=phases_dict,
        )
    )

  if not incidents:
    return RecoverySummaryStats()

  mttrs = [inc.mttr_s for inc in incidents]
  downtimes = [inc.downtime_s for inc in incidents]
  lost = [inc.lost_progress_s for inc in incidents]
  rb_steps = [inc.rolled_back_steps for inc in incidents]

  mean_mttr = sum(mttrs) / len(mttrs)
  mean_downtime = sum(downtimes) / len(downtimes)
  mean_lost = sum(lost) / len(lost)

  phase_stats: dict[str, dict[str, float]] = {}
  for key in RECOVERY_PHASE_ORDER:
    vals = [inc.phases.get(key, 0.0) for inc in incidents]
    mean_val = sum(vals) / len(vals)
    pct = (mean_val / mean_mttr * 100.0) if mean_mttr > 0 else 0.0
    phase_stats[key] = {
        'mean': mean_val,
        'min': min(vals),
        'max': max(vals),
        'total': sum(vals),
        'pct_of_mttr': pct,
    }

  return RecoverySummaryStats(
      num_incidents=len(incidents),
      mean_mttr_s=mean_mttr,
      min_mttr_s=min(mttrs),
      max_mttr_s=max(mttrs),
      mean_downtime_s=mean_downtime,
      min_downtime_s=min(downtimes),
      max_downtime_s=max(downtimes),
      mean_lost_progress_s=mean_lost,
      total_lost_progress_s=sum(lost),
      mean_rolled_back_steps=sum(rb_steps) / len(rb_steps),
      baseline_step_time_s=baseline_step_time,
      phase_stats=phase_stats,
      incidents=incidents,
  )
