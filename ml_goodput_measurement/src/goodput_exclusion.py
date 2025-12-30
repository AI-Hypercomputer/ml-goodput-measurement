"""Mixin class for Goodput Exclusion logic."""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from cloud_goodput.ml_goodput_measurement.src import exclusion_utils
from cloud_goodput.ml_goodput_measurement.src import goodput_utils

BadputType = goodput_utils.BadputType
UnproductiveTimeDict = Dict[BadputType, Union[float, Dict[str, float]]]

_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_STEP_START_TIME = 'step_start_time'
_STEP_COUNT = 'step_count'
_TPU_INIT_START_TIME = 'tpu_init_start_time'
_TPU_INIT_END_TIME = 'tpu_init_end_time'
_TRAINING_PREPARATION_START_TIME = 'training_prep_start_time'
_TRAINING_PREPARATION_END_TIME = 'training_prep_end_time'
_DATA_LOADING_START_TIME = 'data_loading_start_time'
_DATA_LOADING_END_TIME = 'data_loading_end_time'


class GoodputExclusion:
  """Mixin that provides exclusion-aware Goodput calculation methods.

  This class expects to be inherited by GoodputCalculator.
  """

  _goodput_cache: Any
  _goodput_cache_lock: Any
  _number_of_interruptions: int
  _last_disrupted_step: int
  _last_disruption_time: float

  def _fetch_new_entries(
      self, query_time: datetime.datetime
  ) -> List[Dict[str, Any]]:
    raise NotImplementedError

  def _get_total_job_time(self, query_time: datetime.datetime) -> float:
    raise NotImplementedError

  def _get_job_badput_breakdown(
      self, unproductive_time: UnproductiveTimeDict, total_time: float
  ) -> Dict[BadputType, Union[float, Dict[str, float]]]:
    raise NotImplementedError

  def _extract_custom_sync_intervals(
      self, entries: List[Dict[str, Any]]
  ) -> List[Tuple[float, float, str]]:
    raise NotImplementedError

  def _accumulate_unproductive_time(
      self,
      source: UnproductiveTimeDict,
      target: UnproductiveTimeDict,
  ) -> None:
    raise NotImplementedError

  def _sanitize_unproductive_times(
      self, unproductive_time: UnproductiveTimeDict, total_time: float
  ) -> None:
    raise NotImplementedError

  def _compute_other_unproductive_time(
      self,
      total_time: float,
      productive_time: float,
      unproductive_time: UnproductiveTimeDict,
  ) -> float:
    raise NotImplementedError

  def get_job_goodput_with_exclusions(
      self,
      exclusion_intervals: List[Tuple[datetime.datetime, datetime.datetime]],
      include_badput_breakdown: bool = False,
      configured_ideal_step_time: Optional[float] = None,
  ) -> Tuple[
      float, Dict[BadputType, Union[float, Dict[str, float]]], int, float
  ]:
    """Computes Goodput with exclusion intervals applied.

    Args:
      exclusion_intervals: List of (start, end) datetimes to exclude.
      include_badput_breakdown: Whether to return the breakdown.
      configured_ideal_step_time: Optional override for ideal step time.

    Returns:
      A tuple containing:
        - Job Goodput percentage (float).
        - Badput breakdown dictionary (dict).
        - Max productive step (int).
        - Total excluded time in seconds (float).

    Raises:
      ValueError: If adjusted job time is zero.
    """
    query_time = datetime.datetime.now(datetime.timezone.utc)

    self._fetch_new_entries(query_time)
    with self._goodput_cache_lock:
      entries = list(self._goodput_cache.get_cached_entries())

    exclusion_intervals_ts = exclusion_utils.normalize_intervals(
        exclusion_intervals
    )

    # Run the exclusion-aware calculation loop.
    (
        productive_time,
        unproductive_time,
        max_step,
        last_step,
        total_excluded_time,
        adjusted_step_times_map,
    ) = self._get_exclusion_adjusted_metrics(entries, exclusion_intervals_ts)

    # Calculate adjusted total time.
    raw_total_time = self._get_total_job_time(query_time)
    adjusted_total_time = max(0.0, raw_total_time - total_excluded_time)

    if adjusted_total_time == 0.0:
      return 0.0, {}, max_step, total_excluded_time

    # Sanitize the unproductive times and compute the "Unknown/Other" time.
    self._sanitize_unproductive_times(unproductive_time, adjusted_total_time)

    unproductive_time[BadputType.OTHER] = self._compute_other_unproductive_time(
        adjusted_total_time, productive_time, unproductive_time
    )

    # Compute Metrics.
    job_goodput = (float(productive_time) / adjusted_total_time) * 100

    job_badput_breakdown = (
        self._get_job_badput_breakdown(unproductive_time, adjusted_total_time)
        if include_badput_breakdown
        else {}
    )

    # Update Cache with Adjusted Data.
    self._compute_step_info_with_exclusions(
        adjusted_step_times_map, configured_ideal_step_time
    )

    with self._goodput_cache_lock:
      self._goodput_cache.update_goodput_info(
          goodput_utils.GoodputInfo(
              total_productive_time=productive_time,
              total_elapsed_time=adjusted_total_time,
              total_unproductive_time=unproductive_time,
              max_productive_step=max_step,
              last_recorded_step=last_step,
              last_updated_timestamp=datetime.datetime.now(
                  datetime.timezone.utc
              ),
              number_of_disruptions=self._number_of_interruptions,
              total_excluded_time=total_excluded_time,
          )
      )

    return job_goodput, job_badput_breakdown, max_step, total_excluded_time

  def _calculate_adjusted_delta(
      self,
      start: float,
      end: float,
      exclusion_intervals_ts: List[Tuple[float, float]],
  ) -> Tuple[float, float]:
    """Helper to subtract exclusion overlap from a raw time delta."""
    raw_delta = end - start
    exclusion_overlap = exclusion_utils.get_exclusion_overlap(
        start, end, exclusion_intervals_ts
    )
    return max(0.0, raw_delta - exclusion_overlap), exclusion_overlap

  def _compute_step_info_with_exclusions(
      self,
      adjusted_step_times_map: Dict[int, float],
      configured_ideal_step_time: Optional[float],
  ) -> None:
    """Updates the cache with step info derived from adjusted step times."""
    step_times_list = list(adjusted_step_times_map.values())

    if configured_ideal_step_time:
      ideal_step_time = configured_ideal_step_time
    else:
      ideal_step_time = goodput_utils.compute_ideal_step_time(step_times_list)

    if ideal_step_time is None:
      return

    step_deviations = {}
    for step_num, duration in adjusted_step_times_map.items():
      step_deviations[step_num] = duration - ideal_step_time

    with self._goodput_cache_lock:
      self._goodput_cache.update_step_info(
          goodput_utils.StepInfo(
              ideal_step_time=ideal_step_time, step_deviations=step_deviations
          )
      )

  def _get_exclusion_adjusted_metrics(
      self,
      entries: List[Dict[str, Any]],
      exclusion_intervals_ts: List[Tuple[float, float]],
  ) -> Tuple[float, UnproductiveTimeDict, int, int, float, Dict[int, float]]:
    """Main loop for exclusion-aware metric accumulation."""
    productive_training_time = 0.0
    total_unproductive_time: UnproductiveTimeDict = {}
    total_excluded_time = 0.0
    all_adjusted_step_times: Dict[int, float] = {}

    step_start_data = {}
    job_start_time = None
    job_end_time = None
    tpu_init_start = None
    training_prep_start = None
    data_loading_start = None

    tpu_initialization_badput = 0.0
    training_prep_badput = 0.0
    data_loading_badput = 0.0
    sync_data_loading = True

    # Initialize counters for this calculation scope.
    self._number_of_interruptions = 0
    max_prod_step = 0
    last_step = 0

    for payload in entries:
      if _JOB_START_TIME in payload:
        job_start_time = payload[_JOB_START_TIME]
      if _JOB_END_TIME in payload:
        job_end_time = payload[_JOB_END_TIME]

      if _STEP_START_TIME in payload:
        curr_step = int(payload[_STEP_COUNT])

        if curr_step in step_start_data:
          # Disruption detected. Compute segment metrics and reset counters
          # for the next segment.
          self._number_of_interruptions += 1
          self._last_disrupted_step = list(step_start_data.keys())[-1]
          self._last_disruption_time = step_start_data[
              self._last_disrupted_step
          ]

          (
              segment_productive_time,
              segment_unproductive_time,
              segment_max_step,
              segment_exclusion,
              segment_steps_map,
          ) = self._process_segment_with_exclusions(
              step_start_data, curr_step, entries, exclusion_intervals_ts
          )

          productive_training_time += segment_productive_time
          total_excluded_time += segment_exclusion
          self._accumulate_unproductive_time(
              segment_unproductive_time, total_unproductive_time
          )
          if segment_max_step > max_prod_step:
            max_prod_step = segment_max_step
          all_adjusted_step_times.update(segment_steps_map)

          if (
              job_start_time
              and self._last_disruption_time
              and job_start_time > self._last_disruption_time
          ):
            # Adjust infrastructure recovery time after a disruption.
            adj_gap, excl_gap = self._calculate_adjusted_delta(
                self._last_disruption_time,
                job_start_time,
                exclusion_intervals_ts,
            )

            current_val = total_unproductive_time.get(
                BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION, 0.0
            )
            if isinstance(current_val, dict):
              current_val = 0.0

            total_unproductive_time[
                BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION
            ] = (current_val + adj_gap)

            total_excluded_time += excl_gap

          step_start_data = {}
          sync_data_loading = True

        step_start_data[curr_step] = payload[_STEP_START_TIME]

      if _TPU_INIT_START_TIME in payload:
        tpu_init_start = payload[_TPU_INIT_START_TIME]
      elif _TPU_INIT_END_TIME in payload and tpu_init_start:
        adj, excl = self._calculate_adjusted_delta(
            tpu_init_start, payload[_TPU_INIT_END_TIME], exclusion_intervals_ts
        )
        tpu_initialization_badput += adj
        total_excluded_time += excl
        tpu_init_start = None

      elif _TRAINING_PREPARATION_START_TIME in payload:
        training_prep_start = payload[_TRAINING_PREPARATION_START_TIME]
      elif _TRAINING_PREPARATION_END_TIME in payload and training_prep_start:
        adj, excl = self._calculate_adjusted_delta(
            training_prep_start,
            payload[_TRAINING_PREPARATION_END_TIME],
            exclusion_intervals_ts,
        )
        training_prep_badput += adj
        total_excluded_time += excl
        training_prep_start = None

      elif _DATA_LOADING_START_TIME in payload:
        data_loading_start = payload[_DATA_LOADING_START_TIME]
      elif _DATA_LOADING_END_TIME in payload and data_loading_start:
        adj_dl, excl_dl = self._calculate_adjusted_delta(
            data_loading_start,
            payload[_DATA_LOADING_END_TIME],
            exclusion_intervals_ts,
        )
        data_loading_badput += adj_dl
        total_excluded_time += excl_dl

        if sync_data_loading:
          current_dl = total_unproductive_time.get(
              BadputType.DATA_LOADING_SYNC, 0.0
          )
          if isinstance(current_dl, dict):
            current_dl = 0.0

          total_unproductive_time[BadputType.DATA_LOADING_SYNC] = (
              current_dl + adj_dl
          )
          sync_data_loading = False
        data_loading_start = None

    if step_start_data:
      last_step = max(step_start_data.keys())
      (
          segment_productive_time,
          segment_unproductive_time,
          segment_max_step,
          segment_exclusion,
          segment_steps_map,
      ) = self._process_segment_with_exclusions(
          step_start_data, last_step, entries, exclusion_intervals_ts
      )
      productive_training_time += segment_productive_time
      total_excluded_time += segment_exclusion
      self._accumulate_unproductive_time(
          segment_unproductive_time, total_unproductive_time
      )
      if segment_max_step > max_prod_step:
        max_prod_step = segment_max_step
      all_adjusted_step_times.update(segment_steps_map)

      if job_end_time:
        adj_end, excl_end = self._calculate_adjusted_delta(
            step_start_data[last_step], job_end_time, exclusion_intervals_ts
        )
        productive_training_time += adj_end
        total_excluded_time += excl_end
        max_prod_step = last_step

    total_unproductive_time[BadputType.TPU_INITIALIZATION] = (
        tpu_initialization_badput
    )
    total_unproductive_time[BadputType.TRAINING_PREP] = training_prep_badput

    sync_dl_val = total_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0.0)
    if isinstance(sync_dl_val, dict):
      sync_dl_val = 0.0

    async_dl = data_loading_badput - sync_dl_val
    total_unproductive_time[BadputType.DATA_LOADING_ASYNC] = async_dl

    return (
        productive_training_time,
        total_unproductive_time,
        max_prod_step,
        last_step,
        total_excluded_time,
        all_adjusted_step_times,
    )

  def _process_segment_with_exclusions(
      self,
      step_start_data: Dict[int, float],
      curr_step: int,
      entries: List[Dict[str, Any]],
      exclusion_intervals_ts: List[Tuple[float, float]],
  ) -> Tuple[float, UnproductiveTimeDict, int, float, Dict[int, float]]:
    """Calculates metrics for a single segment."""
    step_items = list(step_start_data.items())
    min_step = min(step_start_data.keys())
    custom_sync_intervals = self._extract_custom_sync_intervals(entries)

    total_prod = 0.0
    first_step_time = 0.0
    wasted_progress = 0.0
    sync_breakdown: Dict[str, float] = {}
    total_excluded_in_segment = 0.0
    segment_step_times_map = {}

    steps_in_seg = 0
    max_prod_step_count = 0

    for i in range(1, len(step_items)):
      prev_step, prev_time = step_items[i - 1]
      curr_step_num, curr_time = step_items[i]

      raw_delta = curr_time - prev_time

      excluded_in_delta = exclusion_utils.get_exclusion_overlap(
          prev_time, curr_time, exclusion_intervals_ts
      )
      total_excluded_in_segment += excluded_in_delta

      if curr_step_num <= curr_step:
        if curr_step_num - 1 != prev_step:
          continue

        sync_in_delta = 0.0
        for custom_start, custom_end, custom_type in custom_sync_intervals:
          overlap = exclusion_utils.get_exclusion_overlap(
              prev_time, curr_time, [(custom_start, custom_end)]
          )
          sync_in_delta += overlap
          sync_breakdown[custom_type] = (
              sync_breakdown.get(custom_type, 0.0) + overlap
          )

        adjusted_delta = max(0.0, raw_delta - sync_in_delta - excluded_in_delta)
        total_prod += adjusted_delta

        segment_step_times_map[prev_step] = adjusted_delta

        if prev_step == min_step:
          first_step_time = adjusted_delta

        steps_in_seg += 1
        max_prod_step_count = prev_step

      else:
        wasted_progress += max(0.0, raw_delta - excluded_in_delta)

    if steps_in_seg == 0:
      return (
          0.0,
          {BadputType.WASTED_PROGRESS_FROM_DISRUPTION: wasted_progress},
          0,
          total_excluded_in_segment,
          {},
      )

    non_first_total_time = sum(
        t
        for step_num, t in segment_step_times_map.items()
        if step_num != min_step
    )
    non_first_steps = steps_in_seg - 1

    avg_step = (
        non_first_total_time / non_first_steps
        if non_first_steps > 0
        else first_step_time
    )

    startup_badput = max(0.0, first_step_time - avg_step)
    final_productive = total_prod - startup_badput

    if min_step in segment_step_times_map:
      segment_step_times_map[min_step] = min(first_step_time, avg_step)

    unproductive_time_dict: UnproductiveTimeDict = {
        BadputType.PROGRAM_STARTUP: startup_badput,
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION: wasted_progress,
        BadputType.CUSTOM_BADPUT_EVENTS: sync_breakdown,
    }

    return (
        final_productive,
        unproductive_time_dict,
        max_prod_step_count,
        total_excluded_in_segment,
        segment_step_times_map,
    )
