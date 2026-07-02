"""""Goodput Elasticimplementations."""

import datetime
from typing import Any, Optional, Union

from cloud_goodput.ml_goodput_measurement.src import goodput
from cloud_goodput.ml_goodput_measurement.src import goodput_utils

_JOB_NAME = 'job_name'
_ACTIVE_SLICES = 'active_slices'
_TOTAL_SLICES = 'total_slices'
_AVAILABLE_SLICES = 'available_slices'
_ELASTIC_SLICE_COUNTS_TIMESTAMP = 'elastic_slice_counts_timestamp'

_ELASTIC_WAIT_EVENT_TYPE = 'elastic_wait_event_type'
_ELASTIC_WAIT_START_TIME = 'elastic_wait_start_time'
_ELASTIC_WAIT_END_TIME = 'elastic_wait_end_time'
_ELASTIC_REINIT_START_TIME = 'elastic_reinit_start_time'
_ELASTIC_REINIT_END_TIME = 'elastic_reinit_end_time'
_TPU_INIT_START_TIME = 'tpu_init_start_time'
_TPU_INIT_END_TIME = 'tpu_init_end_time'
_TRAINING_PREPARATION_START_TIME = 'training_prep_start_time'
_TRAINING_PREPARATION_END_TIME = 'training_prep_end_time'

GoodputRecorder = goodput.GoodputRecorder


class ElasticGoodputRecorder(GoodputRecorder):
  """The Goodput recorder (child class) for elastic."""

  def record_elastic_slice_counts(
      self,
      active_slices: int,
      total_slices: int,
      available_slices: int,
      timestamp: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if timestamp is None:
      timestamp = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ACTIVE_SLICES: int(active_slices),
        _TOTAL_SLICES: int(total_slices),
        _AVAILABLE_SLICES: int(available_slices),
        _ELASTIC_SLICE_COUNTS_TIMESTAMP: timestamp.timestamp(),
    })

  def record_elastic_wait_start_time(
      self,
      event_type: str,
      start_time: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ELASTIC_WAIT_EVENT_TYPE: event_type,
        _ELASTIC_WAIT_START_TIME: start_time.timestamp(),
    })

  def record_elastic_wait_end_time(
      self,
      event_type: str,
      end_time: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ELASTIC_WAIT_EVENT_TYPE: event_type,
        _ELASTIC_WAIT_END_TIME: end_time.timestamp(),
    })

  def record_elastic_reinit_start_time(
      self,
      start_time: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ELASTIC_REINIT_START_TIME: start_time.timestamp(),
    })

  def record_elastic_reinit_end_time(
      self,
      end_time: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ELASTIC_REINIT_END_TIME: end_time.timestamp(),
    })


class ElasticGoodputCalculator(goodput.GoodputCalculator):
  """Calculator for elastic training jobs."""

  @staticmethod
  def _extract_elastic_wait_intervals(
      entries: list[dict[str, Any]],
  ) -> list[tuple[float, float, str]]:
    intervals = []
    active: dict[str, float] = {}
    for entry in entries:
      if _ELASTIC_WAIT_START_TIME in entry:
        etype = entry.get(_ELASTIC_WAIT_EVENT_TYPE, 'unknown')
        active[etype] = entry[_ELASTIC_WAIT_START_TIME]
      elif _ELASTIC_WAIT_END_TIME in entry:
        etype = entry.get(_ELASTIC_WAIT_EVENT_TYPE, 'unknown')
        if etype in active:
          start = active.pop(etype)
          end = entry[_ELASTIC_WAIT_END_TIME]
          if start < end:
            intervals.append((start, end, etype))
    return intervals

  @staticmethod
  def _extract_elastic_reinit_intervals(
      entries: list[dict[str, Any]],
  ) -> list[tuple[float, float]]:
    intervals = []
    pending: Optional[float] = None
    for entry in entries:
      if _ELASTIC_REINIT_START_TIME in entry:
        pending = entry[_ELASTIC_REINIT_START_TIME]
      elif _ELASTIC_REINIT_END_TIME in entry and pending is not None:
        end = entry[_ELASTIC_REINIT_END_TIME]
        if pending < end:
          intervals.append((pending, end))
        pending = None
    return intervals

  @staticmethod
  def _extract_init_intervals(
      entries: list[dict[str, Any]],
  ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Returns (tpu_init_intervals, training_prep_intervals)."""
    tpu, prep = [], []
    tpu_start = prep_start = None
    for entry in entries:
      if _TPU_INIT_START_TIME in entry:
        tpu_start = entry[_TPU_INIT_START_TIME]
      elif _TPU_INIT_END_TIME in entry and tpu_start is not None:
        if tpu_start < entry[_TPU_INIT_END_TIME]:
          tpu.append((tpu_start, entry[_TPU_INIT_END_TIME]))
        tpu_start = None
      if _TRAINING_PREPARATION_START_TIME in entry:
        prep_start = entry[_TRAINING_PREPARATION_START_TIME]
      elif _TRAINING_PREPARATION_END_TIME in entry and prep_start is not None:
        if prep_start < entry[_TRAINING_PREPARATION_END_TIME]:
          prep.append((prep_start, entry[_TRAINING_PREPARATION_END_TIME]))
        prep_start = None
    return tpu, prep

  @staticmethod
  def _overlap_with_reinit(
      intervals: list[tuple[float, float]],
      reinit_intervals: list[tuple[float, float]],
  ) -> float:
    """Returns total seconds of overlap between intervals and any reinit interval."""
    total = 0.0
    for start, end in intervals:
      for r_start, r_end in reinit_intervals:
        overlap = min(end, r_end) - max(start, r_start)
        if overlap > 0:
          total += overlap
    return total

  @staticmethod
  def _extract_slice_count_entries(
      entries: list[dict[str, Any]],
  ) -> list[tuple[float, int, int, int]]:
    """Returns list of (timestamp, stepping, available, total)."""
    records = []
    for entry in entries:
      if _ELASTIC_SLICE_COUNTS_TIMESTAMP in entry:
        records.append((
            entry[_ELASTIC_SLICE_COUNTS_TIMESTAMP],
            entry.get(_ACTIVE_SLICES, 0),
            entry.get(_AVAILABLE_SLICES, 0),
            max(entry.get(_TOTAL_SLICES, 1), 1),
        ))
    return sorted(records, key=lambda x: x[0])

  @staticmethod
  def _compute_time_weighted_efficiency(
      slice_records: list[tuple[float, int, int, int]],
      interval_start_ts: float,
      interval_end_ts: float,
  ) -> tuple[float, float]:
    """Returns slice efficiency as time-weighted averages."""
    if not slice_records or interval_end_ts <= interval_start_ts:
      return 0.0, 0.0

    # Seed from the last record at or before interval_start.
    initial = None
    for r in slice_records:
      if r[0] <= interval_start_ts:
        initial = r

    # Build change-points within the interval.
    timeline: list[tuple[float, int, int, int]] = []
    if initial is not None:
      timeline.append((interval_start_ts, initial[1], initial[2], initial[3]))
    for r in slice_records:
      if interval_start_ts < r[0] < interval_end_ts:
        timeline.append(r)

    if not timeline:
      return 0.0, 0.0

    total_duration = interval_end_ts - interval_start_ts
    weighted_stepping = weighted_available = 0.0
    for i, (ts, stepping, available, total) in enumerate(timeline):
      next_ts = timeline[i + 1][0] if i + 1 < len(timeline) else interval_end_ts
      duration = next_ts - ts
      weighted_stepping += (stepping / total) * duration
      weighted_available += (available / total) * duration

    return (
        weighted_stepping / total_duration,
        weighted_available / total_duration,
    )

  def _get_current_productive_and_unproductive_time(
      self, interval_query: Optional[bool] = False
  ) -> tuple[float, goodput.UnproductiveTimeDict, int, int]:
    result = super()._get_current_productive_and_unproductive_time(
        interval_query
    )
    productive_time, unproductive_time, max_step, last_step = result

    if interval_query:
      entries = self._interval_entries
    else:
      with self._goodput_cache_lock:
        entries = list(self._goodput_cache.get_cached_entries())

    # Badput from ELASTIC_SLICE_DOWN and ELASTIC_SCALE_UP
    for start, end, etype in self._extract_elastic_wait_intervals(entries):
      duration = end - start
      bt = (
          goodput_utils.BadputType.ELASTIC_SCALE_UP
          if 'scale_up' in etype.lower()
          else goodput_utils.BadputType.ELASTIC_SLICE_DOWN
      )
      unproductive_time[bt] = unproductive_time.get(bt, 0.0) + duration  # pyrefly: ignore[unsupported-operation]

    # Badput from ELASTIC_REINITIALIZATION.
    reinit_intervals = self._extract_elastic_reinit_intervals(entries)
    for start, end in reinit_intervals:
      unproductive_time[goodput_utils.BadputType.ELASTIC_REINITIALIZATION] = (
          unproductive_time.get(  # pyrefly: ignore[unsupported-operation]
              goodput_utils.BadputType.ELASTIC_REINITIALIZATION, 0.0
          )
          + (end - start)
      )

    # On elastic retries TPU_INIT and TRAINING_PREP are already counted inside
    # ELASTIC_REINITIALIZATION, so discount the overlap.
    if reinit_intervals:
      tpu_intervals, prep_intervals = self._extract_init_intervals(entries)
      for bt, intervals in (
          (goodput_utils.BadputType.TPU_INITIALIZATION, tpu_intervals),
          (goodput_utils.BadputType.TRAINING_PREP, prep_intervals),
      ):
        overlap = self._overlap_with_reinit(intervals, reinit_intervals)
        if overlap > 0 and bt in unproductive_time:
          unproductive_time[bt] = max(0.0, unproductive_time[bt] - overlap)  # pyrefly: ignore[unsupported-operation]

    return productive_time, unproductive_time, max_step, last_step

  def _get_job_badput_breakdown(
      self, total_unproductive_time, total_job_time
  ) -> goodput.UnproductiveTimeDict:
    breakdown = super()._get_job_badput_breakdown(
        total_unproductive_time, total_job_time
    )
    for bt in (
        goodput_utils.BadputType.ELASTIC_SLICE_DOWN,
        goodput_utils.BadputType.ELASTIC_SCALE_UP,
        goodput_utils.BadputType.ELASTIC_REINITIALIZATION,
    ):
      raw = total_unproductive_time.get(bt, 0.0)
      breakdown[bt] = (
          (raw / total_job_time) * 100 if 0 < raw < total_job_time else 0.0
      )
    return breakdown

  def get_interval_metric_details(
      self,
      interval_start: datetime.datetime,
      interval_end: datetime.datetime,
  ) -> goodput_utils.ElasticIntervalWorkloadMetricDetails:
    self._interval_entries = []
    result = super().get_interval_metric_details(interval_start, interval_end)

    entries = self._interval_entries
    slice_records = self._extract_slice_count_entries(entries)
    if slice_records:
      stepping_slice_efficiency, available_slice_efficiency = (
          self._compute_time_weighted_efficiency(
              slice_records,
              interval_start.timestamp(),
              interval_end.timestamp(),
          )
      )
      result['stepping_slice_efficiency'] = stepping_slice_efficiency  # pyrefly: ignore[bad-typed-dict-key]
      result['available_slice_efficiency'] = available_slice_efficiency  # pyrefly: ignore[bad-typed-dict-key]

    return result  # pyrefly: ignore[bad-return]

  def get_job_goodput_details(
      self,
  ) -> goodput_utils.ElasticWorkloadMetricDetails:
    result = super().get_job_goodput_details()
    with self._goodput_cache_lock:
      entries = list(self._goodput_cache.get_cached_entries())
      job_start_time = self._goodput_cache.get_job_start_time()

    if job_start_time and entries:
      slice_records = self._extract_slice_count_entries(entries)
      if slice_records:
        total_elapsed = result.get(
            goodput_utils.MetricType.TOTAL_ELAPSED_TIME.value, 0.0
        )
        end_ts = job_start_time.timestamp() + total_elapsed
        stepping_eff, available_eff = self._compute_time_weighted_efficiency(
            slice_records,
            job_start_time.timestamp(),
            end_ts,
        )
        result['stepping_slice_efficiency'] = stepping_eff  # pyrefly: ignore[bad-typed-dict-key]
        result['available_slice_efficiency'] = available_eff  # pyrefly: ignore[bad-typed-dict-key]
    return result  # pyrefly: ignore[bad-return]
