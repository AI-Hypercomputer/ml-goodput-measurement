"""Goodput package API implementations.

This file contains all the methods exposed through the cloud_goodput library
for users to log necessary information to compute Goodput, and to query the
computed Goodput.
"""

import datetime
import logging
from typing import Any, Optional, Union

from ml_goodput_measurement.src import checkpoint_badput_calculator
from ml_goodput_measurement.src import goodput_cache
from ml_goodput_measurement.src import goodput_utils


get_timestamp_from_log_entry = goodput_utils.get_timestamp_from_log_entry
get_extra_time_from_anomalous_steps = (
    goodput_utils.get_extra_time_from_anomalous_steps
)
compute_ideal_step_time = goodput_utils.compute_ideal_step_time
StepInfo = goodput_utils.StepInfo
BadputType = goodput_utils.BadputType
GoodputType = goodput_utils.GoodputType
GoodputCache = goodput_cache.GoodputCache
GoodputInfo = goodput_utils.GoodputInfo
CheckpointLoggerOptions = checkpoint_badput_calculator.CheckpointLoggerOptions
CheckpointBadputCalculator = (
    checkpoint_badput_calculator.CheckpointBadputCalculator
)
_JOB_NAME = 'job_name'  
_STEP_COUNT = 'step_count'
_STEP_START_TIME = 'step_start_time'
_STEP_COUNT = 'step_count'
_STEP_START_TIME = 'step_start_time'
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_TPU_INIT_START_TIME = 'tpu_init_start_time'
_TPU_INIT_END_TIME = 'tpu_init_end_time'
_TRAINING_PREPARATION_START_TIME = 'training_prep_start_time'
_TRAINING_PREPARATION_END_TIME = 'training_prep_end_time'
_DATA_LOADING_START_TIME = 'data_loading_start_time'
_DATA_LOADING_END_TIME = 'data_loading_end_time'

_CLOUD_LOGGING_PAGE_SIZE = 1000000

logger = logging.getLogger(__name__)


class _CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_name: Name of a specific job.
    logger: The Cloud Logging logger object.
    job_start_time: Start time of the job run.
  """

  def __init__(self, job_name: str, log_name: str):
    """_CloudLogger constructor.

    Args:
      job_name: Name of the job the _CloudLogger is for.
      log_name: Name of the log being written.
    """
    import google.cloud.logging  # pylint: disable=g-import-not-at-top

    self.job_name = job_name
    logging_client = google.cloud.logging.Client()
    self.logger = logging_client.logger(log_name)
    self.job_start_time = None

  def write_cloud_logging_entry(self, entry) -> None:
    """Writes an entry to the Cloud Logging logger at INFO level.

    Args:
      entry: JSON-serializable structured log dictionary.
    """
    if entry is None:
      return
    if entry[_JOB_NAME] == self.job_name:
      self.logger.log_struct(
          entry,
          severity='INFO',
      )

  def _get_filter_msg(
      self,
      start_time: Optional[datetime.datetime],
      end_time: Optional[datetime.datetime],
  ) -> str:
    """Gets the filter message for the Cloud Logging query."""
    filter_entries = [
        'severity=INFO',
        f'jsonPayload.job_name="{self.job_name}"',
    ]
    # Add a filter to bind an end-time to the query window.
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)
    elif end_time.tzinfo is None:
      end_time = end_time.replace(tzinfo=datetime.timezone.utc)

    filter_entries.append(f'timestamp<"{end_time.isoformat()}"')

    # Add a filter to bind a start-time to the query window (if available).
    if start_time is None:
      if self.job_start_time is not None:
        start_time = self.job_start_time - datetime.timedelta(days=1)

    if start_time is not None:
      if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=datetime.timezone.utc)
      filter_entries.append(f'timestamp>="{start_time.isoformat()}"')
    return ' AND '.join(filter_entries)

  def _update_job_start_time(self, entries: list[Any]):
    if self.job_start_time:
      return
    for entry in entries:
      if _JOB_START_TIME in entry and self.job_start_time is None:
        self.job_start_time = datetime.datetime.fromtimestamp(
            entry[_JOB_START_TIME]
        )
        break

  def read_cloud_logging_entries(
      self,
      start_time: Optional[datetime.datetime] = None,
      end_time: Optional[datetime.datetime] = None,
  ):
    """Queries Cloud Logging entries for the specific job.

    Args:
      start_time: The start time of the query window.
      end_time: The end time of the query window.

    Returns:
      Filtered entries in ascending order of timestamp.
    """
    import google.cloud.logging  # pylint: disable=g-import-not-at-top

    entries = self.logger.list_entries(
        filter_=self._get_filter_msg(start_time, end_time),
        order_by=google.cloud.logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entry_payload = [entry.payload for entry in entries]
    self._update_job_start_time(entry_payload)
    return entry_payload


class GoodputRecorder:
  """The Goodput recorder class, responsible for recording Goodput metrics from the user application.

  Attributes:
    job_name: Name of the job the GoodputRecorder is for.
  """

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      logging_enabled=False,
      cloud_logger: Optional[_CloudLogger] = None,
  ):
    """GoodputRecorder constructor.

    Args:
      job_name: Name of the job the GoodputRecorder is for.
      logger_name: The name of the Cloud Logging logger object that the
        application wants logs to be written to and read from.
      logging_enabled: A boolean value to indicate whether the current process
        should send logs to Cloud Logging or not. The application should set
        this value to True if the Recorder is being called from TPU worker 0 and
        the application's configurations request Goodput logging.
      cloud_logger: Should never be passed directly by the user.
    """
    self.job_name = job_name
    # If logging is disabled for this process, do not create a _cloud_logger
    # object and exit early if any record record_* API is called.
    if not logging_enabled:
      self._cloud_logger = None
      logging.info('Logging is disabled for this process.')
      return

    if cloud_logger is not None:
      self._cloud_logger = cloud_logger
    else:
      self._cloud_logger = _CloudLogger(job_name, logger_name)

  def record_step_start_time(
      self, step: int, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log an individual step's start time.

    Args:
      step: The count of the step that timing information is recorded for.
      start_time: Optional backfill start time of the training step. If
        provided, it has to be in UTC time.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _STEP_COUNT: int(step),
        _STEP_START_TIME: start_time.timestamp(),
    })

  def record_checkpoint_progress(self, step, checkpoint_start_time):
    """Main recorder function to log information on a successful checkpoint.

    This method is intended to log the progress for a checkpoint (last step
    included in the checkpoint) and when the checkpoint starts. This information
    will be retrieved in the future to determine whether training progress from
    a completed step contributes to Goodput or wasted progress Badput.

    Args:
      step: The step count of the last step included in the saved checkpoint.
      checkpoint_start_time: Timestamp at which the checkpoint containing
        progress upto "step" starts to save.
    """
    pass

  def record_job_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log a job's start time.

    Args:
      start_time: Optional backfill start time of the job. If provided, it has
        to be in UTC time.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _JOB_START_TIME: start_time.timestamp(),
    })

  def record_job_end_time(self, end_time: Optional[datetime.datetime] = None):
    """Main recorder function to log a job's end time.

    Args:
      end_time: Optional backfull end time of the job. If provided, it has to be
        in UTC time.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _JOB_END_TIME: end_time.timestamp(),
    })

  def record_tpu_init_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time for TPU initialization.

    Note: TPU initialization may include the time spent in completing
    jax.devices() which is responsible for device scanning and Slice Builder
    initialization.

    Args:
      start_time: Start time of TPU initialization.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TPU_INIT_START_TIME: start_time.timestamp(),
    })

  def record_tpu_init_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time for TPU initialization.

    Args:
      end_time: End time of TPU initialization.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TPU_INIT_END_TIME: end_time.timestamp(),
    })

  def record_training_preparation_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time of training preparation before starting a training loop.

    Note: Training preparation may include the time spent in creation of
    checkpoint managers, checkpoint loading, running mesh and model optimizers
    etc.

    Args:
      start_time: Start time of training preparation.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TRAINING_PREPARATION_START_TIME: start_time.timestamp(),
    })

  def record_training_preparation_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time of training preparation before starting a training loop.

    Args:
      end_time: End time of training preparation.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _TRAINING_PREPARATION_END_TIME: end_time.timestamp(),
    })

  def record_data_loading_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the start time of training's data loading.

    Args:
      start_time: Start time of data loading.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _DATA_LOADING_START_TIME: start_time.timestamp(),
    })

  def record_data_loading_end_time(
      self, end_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log the end time of training's data loading.

    Args:
      end_time: End time of data loading.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.utcnow()

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _DATA_LOADING_END_TIME: end_time.timestamp(),
    })


class GoodputCalculator:
  """The Goodput calculator class, responsible for querying necessary information and computing Goodput metrics to return to the user application.

  Attributes:
    job_name: Name of the job the GoodputCalculator is for.
    using_pathways: Whether or not the job uses Pathways.
  """

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      cloud_logger: Optional[_CloudLogger] = None,
      using_pathways: bool = False,
  ):
    """GoodputCalculator constructor.

    Args:
      job_name: Name of the job the GoodputCalculator is for.
      logger_name: Name of the log being written.
      cloud_logger: Should never be passed directly by the user.
      using_pathways: Whether or not the job uses Pathways.
    """
    self.job_name = job_name
    self.using_pathways = using_pathways
    if cloud_logger is not None:
      self._cloud_logger = cloud_logger
    else:
      self._cloud_logger = _CloudLogger(job_name, logger_name)
    self._current_entries = []
    self._goodput_cache = GoodputCache()
    self._interval_entries = []
    self._interval_start_time = None
    self._interval_end_time = None
    self._number_of_interruptions = 0
    self._gcm_last_recorded_timestamp = None

  def _get_total_productive_and_unproductive_time(
      self,
  ) -> tuple[float, dict[BadputType, float], int]:
    """Helper function to compute the total productive training time, unproductive time and the last step recorded till now.

    Returns:
      A tuple of the total productive training time, the total unproductive time
      (dict of BadputType and unproductive time) and the last step recorded till
      now.
    """
    (
        cached_productive_time,
        cached_unproductive_time,
        cached_last_recorded_step,
    ) = self._get_cached_productive_and_unproductive_time()
    current_productive_time, current_unproductive_time, last_recorded_step = (
        self._get_current_productive_and_unproductive_time()
    )
    total_productive_time = cached_productive_time + current_productive_time
    total_unproductive_time = cached_unproductive_time
    for badput_type, unproductive_time in current_unproductive_time.items():
      if badput_type in cached_unproductive_time:
        total_unproductive_time[badput_type] = (
            unproductive_time + cached_unproductive_time[badput_type]
        )
      else:
        total_unproductive_time[badput_type] = unproductive_time
    last_recorded_step = max(cached_last_recorded_step, last_recorded_step)

    return (
        total_productive_time,
        total_unproductive_time,
        last_recorded_step,
    )

  def _get_cached_productive_and_unproductive_time(
      self,
  ) -> tuple[float, dict[BadputType, float], int]:
    """Helper function to retrieve the cached productive training time and unproductive time."""
    goodput_info = self._goodput_cache._goodput_info
    if not self._goodput_cache.is_cache_empty() and goodput_info is not None:
      return (
          goodput_info.total_productive_time,
          goodput_info.total_unproductive_time,
          goodput_info.last_recorded_step,
      )
    return (0.0, {}, 0)

  def _get_current_productive_and_unproductive_time(
      self, interval_query: Optional[bool] = False
  ) -> tuple[float, dict[BadputType, float], int]:
    """Helper function to compute the current productive training time, unproductive time and the last step recorded till now.

    Args:
      interval_query: A boolean value to indicate whether the current query is
        for an interval or not.

    Returns:
      A tuple of the productive training time, the unproductive time
      (dict of BadputType and unproductive time) and the last step recorded till
      now based on the latest entries retrieved from Cloud Logging.
    """

    def get_segment_productive_and_unproductive_time(
        step_start_data: dict[int, float], curr_step: int
    ) -> tuple[float, dict[BadputType, float]]:
      """Helper function to compute the segment productive and unproductive time.

      This method computes productive training time between the beginning of a
      segment of step start time data and current step.

      This function also returns a dictionary of some Badput Types and the
      unproductive time associated with these.

      Args:
        step_start_data: Dictionary containing a segment of step time data.
        curr_step: The current step until which the segment productive time is
          calculated.

      Returns:
        The job's segment productive training time and segment unproductive
        time.
      """
      if curr_step == 0:
        return (0.0, {})

      segment_productive_total_time = 0.0
      first_step_time = 0.0
      steps_in_segment = 0
      min_step = min(list(step_start_data.keys()))
      step_times = []
      for step, start_time in step_start_data.items():
        if step <= curr_step and step - 1 in step_start_data:
          segment_productive_total_time += (
              start_time - step_start_data[step - 1]
          )
          if step - 1 == min_step:
            first_step_time = segment_productive_total_time
          else:
            # Collect all non-first step times to compute possible Badput
            # from anomalous failures.
            step_times.append(start_time - step_start_data[step - 1])
          steps_in_segment += 1

      if steps_in_segment == 0:
        return (0.0, {})

      if steps_in_segment == 1:
        # Extra step time is not computable with only one step, so it is not
        # discounted in this case.
        return (first_step_time, {})

      # Compute Badput from the first step
      first_step_extra_time = 0.0
      average_step_time = (segment_productive_total_time - first_step_time) / (
          steps_in_segment - 1
      )
      if first_step_time > average_step_time:
        first_step_extra_time = first_step_time - average_step_time
      extra_time_from_anomalous_steps = 0.0
      if self.using_pathways:
        extra_time_from_anomalous_steps = get_extra_time_from_anomalous_steps(
            step_times
        )
      total_segment_productive_time = (
          segment_productive_total_time
          - first_step_extra_time
          - extra_time_from_anomalous_steps
      )
      total_segment_unproductive_time = {
          BadputType.PROGRAM_STARTUP: first_step_extra_time
      }
      return (
          total_segment_productive_time,
          total_segment_unproductive_time,
      )

    def _accumulate_segment_unproductive_time(
        segment_unproductive_time: dict[BadputType, float],
        total_unproductive_time: dict[BadputType, float],
    ):
      for badput_type, unproductive_time in segment_unproductive_time.items():
        if badput_type in total_unproductive_time:
          total_unproductive_time[badput_type] += unproductive_time
        else:
          total_unproductive_time[badput_type] = unproductive_time

    # Build a deserialized dictionary from cloud logging entries to store step
    # start times. The dictionary maps from step count to start time and will be
    # used to each step's productive time by looking for its completion in the
    # next step's start.
    # Note in the instance where progress is lost due to a disruption and the
    # last successful checkpoint did not include all the steps, the last set of
    # records of the step information will be kept and the previous set will be
    # overwritten by design so as to correct for the the previously computed
    # additional time that was counted as productive but lost due to a
    # disruption.
    productive_training_time = 0.0
    total_unproductive_time = {}
    step_start_data = {}
    job_start_time = None
    job_end_time = None
    tpu_init_start_time = None
    training_prep_start_time = None
    data_loading_start_time = None
    tpu_initialization_badput = 0.0
    training_prep_badput = 0.0
    data_loading_badput = 0.0
    entries_to_process = (
        self._interval_entries if interval_query else self._current_entries
    )
    self._number_of_interruptions = 0
    for payload in entries_to_process:
      if _JOB_START_TIME in payload:
        # Keep track of the latest start to compute badput due to disruption.
        job_start_time = payload[_JOB_START_TIME]
      if _STEP_START_TIME in payload:
        curr_step = int(payload[_STEP_COUNT])
        if curr_step not in step_start_data:
          step_start_data[curr_step] = payload[_STEP_START_TIME]
        else:
          # In this case, the job restarted from Step (curr_step). It means that
          # all progress till Step (curr_step - 1) has been preserved. So we
          # can get the productive time since the previous start/restart and
          # then clear the step_start_data dict.
          segment_productive_time, segment_unproductive_time = (
              get_segment_productive_and_unproductive_time(
                  step_start_data, curr_step
              )
          )
          productive_training_time += segment_productive_time
          # Since the current step has been recorded again, the progress
          # between the previously recorded curr_step and recently recorded
          # curr_step has been lost to a disruption and partially recovered
          # due to a checkpoint of curr_step - 1. Accumulate the lost time in
          # this segment as unproductive time.
          # Note this unproductive time is divided into two buckets:
          #   1. Wasted training progress after the last successfully
          #      checkpointed step and the disruption time until the job
          #      restarts.
          #   2. TPU re-init, training prep, data loading, program startup,
          #      checkpoint loading etc. after the job restarts and before
          #      training progress resumes.

          # The first bucket can be calculated as the time between the start
          # time of curr_step and the job restart time immediately prior.
          if job_start_time is not None:
            wasted_progress_from_disruption = (
                job_start_time - step_start_data[curr_step]
            )
            segment_unproductive_time[
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION
            ] = wasted_progress_from_disruption
            self._number_of_interruptions += 1

          # The second bucket is individually computed either from recorded
          # logs (TPU initialization, training preparation, data loading) or
          # computed from the first step time after start or restart
          # (segment unproductive time). All unproductive time is accumulated
          # as we go.
          _accumulate_segment_unproductive_time(
              segment_unproductive_time, total_unproductive_time
          )
          step_start_data = {curr_step: payload[_STEP_START_TIME]}

      if _JOB_END_TIME in payload:
        # Locate the last instance of job's end time if the job has completed.
        job_end_time = payload[_JOB_END_TIME]

      # Compute badput due to TPU initialization.
      if _TPU_INIT_START_TIME in payload:
        tpu_init_start_time = payload[_TPU_INIT_START_TIME]
      elif _TPU_INIT_END_TIME in payload and tpu_init_start_time is not None:
        tpu_initialization_badput += (
            payload[_TPU_INIT_END_TIME] - tpu_init_start_time
        )
        tpu_init_start_time = None

      # Compute badput due to training preparation.
      elif _TRAINING_PREPARATION_START_TIME in payload:
        training_prep_start_time = payload[_TRAINING_PREPARATION_START_TIME]
      elif (
          _TRAINING_PREPARATION_END_TIME in payload
          and training_prep_start_time is not None
      ):
        training_prep_badput += (
            payload[_TRAINING_PREPARATION_END_TIME] - training_prep_start_time
        )
        training_prep_start_time = None

      # Compute badput due to data loading.
      elif _DATA_LOADING_START_TIME in payload:
        data_loading_start_time = payload[_DATA_LOADING_START_TIME]
      elif (
          _DATA_LOADING_END_TIME in payload
          and data_loading_start_time is not None
      ):
        data_loading_badput += (
            payload[_DATA_LOADING_END_TIME] - data_loading_start_time
        )
        data_loading_start_time = None

    # Compute unproductive time from checkpoint manager save and restore.
    checkpoint_logger_options = CheckpointLoggerOptions(use_goodput_logger=True)
    checkpoint_badput_calc = CheckpointBadputCalculator(
        checkpoint_logger_options
    )
    checkpoint_badput_calc.entries = entries_to_process
    checkpoint_manager_save_stats = (
        checkpoint_badput_calc.calculate_save_operation_checkpoint_manager_blocking_time()
    )
    checkpoint_manager_save_badput = (
        checkpoint_manager_save_stats.total_checkpoint_manager_blocking_time
    )
    checkpoint_manager_restore_stats = (
        checkpoint_badput_calc.calculate_restore_operation_checkpoint_manager_blocking_time()
    )
    checkpoint_manager_restore_badput = (
        checkpoint_manager_restore_stats.total_checkpoint_manager_time
    )

    # Populate some Badput buckets in total_unproductive_time.
    total_unproductive_time[BadputType.TPU_INITIALIZATION] = (
        tpu_initialization_badput
    )
    total_unproductive_time[BadputType.TRAINING_PREP] = training_prep_badput
    total_unproductive_time[BadputType.DATA_LOADING] = data_loading_badput
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME] = (
        checkpoint_manager_save_badput
    )
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME] = (
        checkpoint_manager_restore_badput
    )

    if not step_start_data:
      return 0.0, {}, 0

    last_step = max(list(step_start_data.keys()))
    segment_productive_time, segment_unproductive_time = (
        get_segment_productive_and_unproductive_time(step_start_data, last_step)
    )
    productive_training_time += segment_productive_time
    _accumulate_segment_unproductive_time(
        segment_unproductive_time, total_unproductive_time
    )

    if job_end_time is not None:
      productive_training_time += job_end_time - step_start_data[last_step]
    elif (
        interval_query
        and self._interval_end_time
        and self._interval_end_time.timestamp() > step_start_data[last_step]
    ):
      productive_training_time += (
          self._interval_end_time.timestamp() - step_start_data[last_step]
      )
    elif not interval_query:
      productive_training_time += (
          datetime.datetime.utcnow().timestamp() - step_start_data[last_step]
      )

    # Remove blocking checkpoint manager save time from productive time.
    productive_training_time -= checkpoint_manager_save_badput

    # Return a tuple of the total productive training time, the total
    # unproductive time (dict of BadputType and unproductive time) and the last
    # step recorded.
    return productive_training_time, total_unproductive_time, last_step

  def _get_total_job_time(self) -> float:
    """Helper function to compute the total job runtime."""
    return self._get_cached_total_job_time() + self._get_current_job_time()

  def _get_cached_total_job_time(self) -> float:
    """Helper function to retrieve cached total job runtime if available."""
    if not self._goodput_cache.is_cache_empty():
      goodput_info = self._goodput_cache._goodput_info
      if goodput_info:
        return goodput_info.total_elapsed_time_since_start
    return 0.0

  def _get_current_job_time(self) -> float:
    """Helper function to compute the current job runtime.

    Returns:
      The job's total runtime computed based on the last retrieved logs.
    """
    # Find the last entry's timestamp as current window's start
    # (present if entries are cached).
    start_time = self._goodput_cache._last_entry_timestamp
    if start_time:
      end_time = self._goodput_cache._job_end_time
      if end_time is not None:
        return end_time.timestamp() - start_time.timestamp()
      # If the job's end time is missing then job has not yet completed, use
      # current time to compute total job time.
      return (
          datetime.datetime.now(datetime.timezone.utc).timestamp()
          - start_time.timestamp()
      )

    # De-serealize job start and end times from cloud logging entries. These
    # will be used to compute total runtime of the job.
    job_start_time = None
    job_end_time = None
    for payload in self._current_entries:
      # Locate the earliest timestamp recorded for the job's start.
      if _JOB_START_TIME in payload and job_start_time is None:
        job_start_time = payload[_JOB_START_TIME]
      # Locate the latest timestamp recorded for the job's end.
      if _JOB_END_TIME in payload:
        job_end_time = payload[_JOB_END_TIME]

    if job_start_time is not None:
      if job_end_time is not None:
        return job_end_time - job_start_time
      # If the job's end time is missing then job has not yet completed, use
      # current time to compute total job time.
      return datetime.datetime.utcnow().timestamp() - job_start_time
    # The the job's start time is missing so the total job time cannot be
    # calculated. Caller of this function should raise an error if this happens.
    return 0.0

  def _update_log_entries(self):
    """Helper function to update the log entries."""
    current_query_time = datetime.datetime.now(datetime.timezone.utc)
    if not self._goodput_cache.is_cache_empty():
      last_entry_timestamp = self._goodput_cache._last_entry_timestamp
      self._current_entries = self._cloud_logger.read_cloud_logging_entries(
          last_entry_timestamp, current_query_time
      )
    else:
      self._current_entries = self._cloud_logger.read_cloud_logging_entries()

  def _get_interval_log_entries(
      self, start_time: datetime.datetime, end_time: datetime.datetime
  ):
    """Helper function to get log entries from an interval window."""
    if start_time is None or end_time is None:
      raise ValueError(
          'Start and end times are required to get log entries from an interval'
          ' window.'
      )
    self._interval_entries = self._cloud_logger.read_cloud_logging_entries(
        start_time, end_time
    )
    logging.info(
        'Inspecting interval entries between %s and %s', start_time, end_time
    )

    if not self._interval_entries:
      raise ValueError(
          'No log entries found within the interval window between %s and %s.'
          % (start_time, end_time)
      )

  def _get_total_job_time_from_interval(
      self, start_interval: datetime.datetime, end_interval: datetime.datetime
  ) -> float:
    """Helper function to compute the total job runtime from interval entries."""
    # Get the first and last entry's timestamps in the window
    first_entry_timestamp = get_timestamp_from_log_entry(
        self._interval_entries[0]
    )
    last_entry_timestamp = get_timestamp_from_log_entry(
        self._interval_entries[-1]
    )

    # Calculate effective start_time and end_time
    self._interval_start_time = (
        max(start_interval, first_entry_timestamp)
        if first_entry_timestamp
        else start_interval
    )
    self._interval_end_time = (
        min(end_interval, last_entry_timestamp)
        if last_entry_timestamp
        else end_interval
    )

    # Ensure start_time is not after end_time
    if self._interval_start_time >= self._interval_end_time:
      raise ValueError(
          'Start time is on or after end time, cannot compute total job time.'
      )

    return (
        self._interval_end_time.timestamp()
        - self._interval_start_time.timestamp()
    )

  def get_job_goodput(
      self, include_badput_breakdown=False
  ) -> tuple[float, dict[BadputType, float], int]:
    """Method to get the cumulative Goodput and Badput breakdown of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job.

    This method also returns the Badput breakdown of the job if
    `include_badput_breakdown` is set to True.

    Additionaly, this method returns the last step recorded for the job. This is
    primarily used for improving monitoring and observability of the job's
    overall Goodput as a function of number of executed steps.

    Args:
      include_badput_breakdown: Whether or not to return the badput breakdown.
        If False, returns {} for the badput breakdown.

    Returns:
      A tuple of the job's Goodput, optionally the Badput breakdown and the last
      step recorded for the job.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training time is invalid.
    """

    # Update the logs used to compute Goodput.
    self._update_log_entries()

    total_job_time = self._get_total_job_time()
    # No calculations can be made if total job time is zero. This can happen if
    # logs for the job are not present, sent to an invalid location or contain
    # bad data. Raise a ValueError if this happens.
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    productive_training_time, total_unproductive_time, last_step = (
        self._get_total_productive_and_unproductive_time()
    )
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )
    # Return a tuple of calculated Goodput & Badput of the job till now and the
    # last recorded step.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    if include_badput_breakdown:
      job_badput_breakdown = self._get_job_badput_breakdown(
          productive_training_time, total_unproductive_time, total_job_time
      )
    else:
      job_badput_breakdown = {}

    # Update the Goodput cache with new information.
    self._goodput_cache.update_cached_entries(self._current_entries)
    self._goodput_cache.update_goodput_info(
        GoodputInfo(
            total_productive_time=productive_training_time,
            total_elapsed_time_since_start=total_job_time,
            total_unproductive_time=total_unproductive_time,
            last_recorded_step=last_step,
            last_recorded_timestamp=self._goodput_cache._last_entry_timestamp,
        )
    )
    return job_goodput, job_badput_breakdown, last_step

  def get_job_goodput_interval(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> tuple[float, dict[BadputType, float], int, float, int]:
    """Method to get the Goodput and Badput breakdown of the job within an interval window.

    If the application is interested in retrieving the Goodput of the job within
    a specific window of time, this method provides the metrics computed between
    the start and end of this window.

    Additionaly, this method returns the last step recorded for the job. This is
    primarily used for improving monitoring and observability of the job's
    overall Goodput as a function of number of executed steps.

    Args:
      interval_start: The start time of the window for which Goodput is to be
        computed.
      interval_end: The end time of the window for which Goodput is to be
        computed.

    Returns:
      A tuple containing:
        - The job's Goodput percentage with respect to the total job time within
          the interval window.
        - The Badput Breakdown percentages with respect to the total job time
          within the interval window.
        - The last step recorded for the job within the interval window.
        - The total job time within the interval window.
        - The number of disruptions within the interval window.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training or unproductive time is invalid.
    """

    # Get the logs for the interval and validate the interval window.
    self._get_interval_log_entries(interval_start, interval_end)

    total_job_time = self._get_total_job_time_from_interval(
        interval_start, interval_end
    )

    productive_training_time, total_unproductive_time, last_step = (
        self._get_current_productive_and_unproductive_time(interval_query=True)
    )
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )
    # Return a tuple of calculated Goodput & Badput of the job till now and the
    # last recorded step.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    job_badput_breakdown = self._get_job_badput_breakdown(
        productive_training_time, total_unproductive_time, total_job_time
    )

    return (
        job_goodput,
        job_badput_breakdown,
        last_step,
        total_job_time,
        self._number_of_interruptions,
    )

  def _get_step_times(self):
    """Helper function to compute step times from the log entries."""
    step_times = {}
    previous_step_start_time = None
    previous_step_count = None
    for payload in self._current_entries:
      if _STEP_START_TIME in payload:
        step_start_time = payload[_STEP_START_TIME]
        step_count = int(payload[_STEP_COUNT])
        if (
            previous_step_start_time is not None
            and previous_step_count is not None
            and step_count == previous_step_count + 1
        ):
          step_times[previous_step_count] = (
              step_start_time - previous_step_start_time
          )
        previous_step_count = step_count
        previous_step_start_time = step_start_time
    return step_times

  def get_step_deviation(
      self, configured_ideal_step_time: Optional[float] = None
  ) -> dict[int, float]:
    """Method to get the step deviation of the current step based on the ideal step time.

    This method computes the ideal step time if one is not provided by the user
    and returns the step deviation of the current step.

    Args:
      configured_ideal_step_time: Optional user-defined ideal step time.

    Returns:
      A dictionary of step deviation for each step.
    """
    # Get the log entries.
    self._update_log_entries()
    # Compute step times from the log entries.
    step_times = self._get_step_times()
    # Get the previous ideal step time from the cache.
    previous_ideal_step_time = (
        self._goodput_cache._step_info.ideal_step_time
        if self._goodput_cache._step_info
        and self._goodput_cache._step_info.ideal_step_time
        else None
    )
    # Compute ideal step time.
    ideal_step_time = (
        configured_ideal_step_time
        if configured_ideal_step_time is not None
        else compute_ideal_step_time(
            step_times=list(step_times.values()),
            previous_ideal_step_time=previous_ideal_step_time,
        )
    )

    # Compute step deviation.
    step_deviations = {
        step_count: abs(step_time - ideal_step_time)
        for step_count, step_time in step_times.items()
    }
    # Update the step information in the cache.
    self._goodput_cache.update_step_info(
        StepInfo(
            ideal_step_time=ideal_step_time,
            step_deviations=step_deviations,
        )
    )
    return step_deviations

  def _get_job_badput_breakdown(
      self, total_productive_time, total_unproductive_time, total_job_time
  ):
    """Method to get the the Badput breakdown as percentage of total job time.

    This method provides a granular breakdown of the known components of Badput.

    Args:
      total_productive_time: The total productive training time.
      total_unproductive_time: A dictionary of computed unproductive time of
        each BadputType.
      total_job_time: The total job time.

    Returns:
      A dictionary of badput components and their percentage breakdown within
      total job time.
    """
    badput_breakdown = {}
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Badput cannot be calculated. Please fix the'
          ' logging entries.'
      )

    # TPU initialization badput.
    tpu_init_badput = total_unproductive_time.get(
        BadputType.TPU_INITIALIZATION, 0.0
    )
    badput_breakdown[BadputType.TPU_INITIALIZATION] = (
        (tpu_init_badput / total_job_time) * 100
        if 0 < tpu_init_badput < total_job_time
        else 0.0
    )

    # Training preparation badput.
    training_prep_badput = total_unproductive_time.get(
        BadputType.TRAINING_PREP, 0.0
    )
    badput_breakdown[BadputType.TRAINING_PREP] = (
        (training_prep_badput / total_job_time) * 100
        if 0 < training_prep_badput < total_job_time
        else 0.0
    )

    # Data loading badput.
    data_loading_badput = total_unproductive_time.get(
        BadputType.DATA_LOADING, 0.0
    )
    badput_breakdown[BadputType.DATA_LOADING] = (
        (data_loading_badput / total_job_time) * 100
        if 0 < data_loading_badput < total_job_time
        else 0.0
    )

    # Unproductive checkpoint save time badput.
    checkpoint_save_badput = total_unproductive_time.get(
        BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME, 0.0
    )
    badput_breakdown[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME] = (
        (checkpoint_save_badput / total_job_time) * 100
        if 0 < checkpoint_save_badput < total_job_time
        else 0.0
    )

    # Unproductive checkpoint restore time badput.
    checkpoint_restore_badput = total_unproductive_time.get(
        BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME, 0.0
    )
    badput_breakdown[BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME] = (
        (checkpoint_restore_badput / total_job_time) * 100
        if 0 < checkpoint_restore_badput < total_job_time
        else 0.0
    )

    # Program startup badput.
    program_startup_badput = total_unproductive_time.get(
        BadputType.PROGRAM_STARTUP, 0.0
    )
    badput_breakdown[BadputType.PROGRAM_STARTUP] = (
        (program_startup_badput / total_job_time) * 100
        if 0 < program_startup_badput < total_job_time
        else 0.0
    )

    # Wasted progress from disruption badput.
    wasted_progress_from_disruption_badput = total_unproductive_time.get(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION, 0.0
    )
    badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION] = (
        (wasted_progress_from_disruption_badput / total_job_time) * 100
        if 0 < wasted_progress_from_disruption_badput < total_job_time
        else 0.0
    )

    # Populate the 'Other/Unknown' badput bucket.
    other_badput = (
        total_job_time
        - total_productive_time
        - sum(total_unproductive_time.values())
    )
    badput_breakdown[BadputType.OTHER] = (
        other_badput / total_job_time * 100
        if 0 < other_badput < total_job_time
        else 0.0
    )

    return badput_breakdown

  def get_job_goodput_details(
      self,
  ) -> dict[str, dict[Union[BadputType, GoodputType], float]]:
    """Method to get the productive and non-productive time with breakdown of the job computed until now."""

    goodput_info = self._goodput_cache.get_goodput_info()
    if goodput_info is None:
      logger.warning(
          'Goodput information unavailable and will not be uploaded to GCM'
      )
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }

    (
        productive_training_time,
        total_unproductive_time,
        cache_last_recorded_timestamp,
    ) = (
        goodput_info.total_productive_time,
        goodput_info.total_unproductive_time,
        goodput_info.last_recorded_timestamp,
    )

    if (
        self._gcm_last_recorded_timestamp is not None  # Ignore the first entry.
        and self._gcm_last_recorded_timestamp == cache_last_recorded_timestamp
    ):
      logger.warning('No new data, skipping upload to GCM.')
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }

    if (
        self._gcm_last_recorded_timestamp is not None
        and self._gcm_last_recorded_timestamp > cache_last_recorded_timestamp
    ):
      logger.error(
          'GCM last recorded timestamp is greater than cache last recorded'
          ' timestamp. This should not happen.'
      )
      return {
          'goodput_time_dict': {},
          'badput_time_dict': {},
      }

    self._gcm_last_recorded_timestamp = cache_last_recorded_timestamp

    # Currently productive_time is not split based on productive activities, it
    # is just the total productive time. We will modify this to follow the same
    # format as badput_breakdown. Please update this code accordingly in the
    # future when we have more granular breakdown of productive time.

    total_productive_time = {GoodputType.TOTAL: productive_training_time}

    return {
        'goodput_time_dict': total_productive_time,
        'badput_time_dict': total_unproductive_time,
    }
