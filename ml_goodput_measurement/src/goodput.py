"""Goodput package API implementations.

This file contains all the methods exposed through the cloud_tpu_goodput library
for users to log necessary information to compute Goodput, and to query the
computed Goodput.
"""

import datetime
import enum
import logging
from typing import Any, Optional
import numpy as np
from scipy import stats

_JOB_NAME = 'job_name'
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


class _CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_name: Name of a specific job.
    logger: The Cloud Logging logger object.
  """

  def __init__(self, job_name: str, log_name: str):
    """_CloudLogger constructor.

    Args:
      job_name: Name of the job the _CloudLogger is for.
      log_name: Name of the log being written.
    """
    import google.cloud.logging   # pylint: disable=g-import-not-at-top
    self.job_name = job_name
    logging_client = google.cloud.logging.Client()
    self.logger = logging_client.logger(log_name)

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

  def read_cloud_logging_entries(self):
    """Queries Cloud Logging entries for the specific job.

    Returns:
      Filtered entries in ascending order of timestamp.

    """
    import google.cloud.logging   # pylint: disable=g-import-not-at-top
    filter_entries = [
        'severity=INFO',
        f'jsonPayload.job_name="{self.job_name}"',
    ]
    filter_entries = ' AND '.join(filter_entries)
    entries = self.logger.list_entries(
        filter_=filter_entries,
        order_by=google.cloud.logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entry_payload = [entry.payload for entry in entries]
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
      logger: Optional[_CloudLogger] = None,
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
      logger: Should never be passed directly by the user.
    """
    self.job_name = job_name
    # If logging is disabled for this process, do not create a _cloud_logger
    # object and exit early if any record record_* API is called.
    if not logging_enabled:
      self._cloud_logger = None
      logging.info('Logging is disabled for this process.')
      return

    if logger is not None:
      self._cloud_logger = logger
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


class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING = 4
  UNPRODUCTIVE_CHECKPOINTING = 5
  WASTED_PROGRESS_FROM_DISRUPTION = 6
  OTHER = 7


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
      logger: Optional[_CloudLogger] = None,
      using_pathways: bool = False,
  ):
    """GoodputCalculator constructor.

    Args:
      job_name: Name of the job the GoodputCalculator is for.
      logger_name: Name of the log being written.
      logger: Should never be passed directly by the user.
    """
    self.job_name = job_name
    self.using_pathways = using_pathways
    if logger is not None:
      self._cloud_logger = logger
    else:
      self._cloud_logger = _CloudLogger(job_name, logger_name)

  def _get_total_productive_training_time(self, entries: list[Any]) -> float:
    """Helper function to compute the total productive training time.

    Args:
      entries: Cloud Logging entries from user-specified logger for a specific
        job.

    Returns:
      The job's total productive training time.
    """

    def get_extra_time_from_anomalous_steps(step_times: list[Any]) -> float:
      def get_anomalous_and_normal_step_times(
          step_times: list[Any],
      ) -> tuple[list[Any], list[Any]]:
        mad = stats.median_abs_deviation(step_times)
        med = np.median(step_times)

        anomalous_step_times = []
        normal_step_times = []
        for step_time in step_times:
          if step_time > (med + mad * 3):
            anomalous_step_times.append(step_time)
          else:
            normal_step_times.append(step_time)

        return anomalous_step_times, normal_step_times

      anomalous_step_times, normal_step_times = (
          get_anomalous_and_normal_step_times(step_times)
      )
      normal_step_mean = np.mean(normal_step_times)
      return sum(anomalous_step_times) - (
          len(anomalous_step_times) * normal_step_mean
      )

    def get_segment_productive_time(
        step_start_data: dict[int, float], curr_step: int
    ) -> float:
      """Helper function to compute the segment productive time.

      This method computes productive training time between the beginning of a
      segment of step start time data and current step.

      Args:
        step_start_data: Dictionary containing a segment of step time data.
        curr_step: The current step until which the segment productive time is
          calculated.

      Returns:
        The job's segment productive training time.
      """
      if curr_step == 0:
        return 0.0

      segment_productive_total_time = 0.0
      first_step_time = 0.0
      steps_in_segment = 0
      min_step = min(list(step_start_data.keys()))
      step_times = []
      for step, start_time in step_start_data.items():
        if (
            step <= curr_step
            and step - 1 in step_start_data
        ):
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
        return 0.0

      if steps_in_segment == 1:
        # Extra step time is not computable with only one step, so it is not
        # discounted in this case.
        return first_step_time

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
      return (
          segment_productive_total_time
          - first_step_extra_time
          - extra_time_from_anomalous_steps
      )

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
    step_start_data = {}
    job_end_time = None
    for payload in entries:
      if _STEP_START_TIME in payload:
        curr_step = int(payload[_STEP_COUNT])
        if curr_step not in step_start_data:
          step_start_data[curr_step] = payload[_STEP_START_TIME]
        else:
          # In this case, the job restarted from Step (curr_step). It means that
          # all progress till Step (curr_step - 1) has been preserved. So we
          # can get the productive time since the previous start/restart and
          # then clear the step_start_data dict.
          productive_training_time += get_segment_productive_time(
              step_start_data, curr_step
          )
          step_start_data = {curr_step: payload[_STEP_START_TIME]}

      if _JOB_END_TIME in payload:
        # Locate the last instance of job's end time if the job has completed.
        job_end_time = payload[_JOB_END_TIME]

    if not step_start_data:
      return 0.0

    last_step = max(list(step_start_data.keys()))
    productive_training_time += get_segment_productive_time(
        step_start_data, last_step
    )

    if job_end_time is not None:
      productive_training_time += job_end_time - step_start_data[last_step]
    else:
      productive_training_time += (
          datetime.datetime.utcnow().timestamp() - step_start_data[last_step]
      )

    return productive_training_time

  def _get_total_job_time(self, entries: list[Any]) -> float:
    """Helper function to compute the total job runtime.

    Args:
      entries: Cloud Logging entries from user-specified logger for a specific
        job.

    Returns:
      The job's total runtime.
    """
    # De-serealize job start and end times from cloud logging entries. These
    # will be used to compute total runtime of the job.
    job_start_time = None
    job_end_time = None
    for payload in entries:
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

  def get_job_goodput(self):
    """Method to get the cumulative Goodput of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job.

    Returns:
      Goodput percentage of the entire job.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training time is invalid.
    """
    entries = self._cloud_logger.read_cloud_logging_entries()
    total_job_time = self._get_total_job_time(entries)
    # No calculations can be made if total job time is zero. This can happen if
    # logs for the job are not present, sent to an invalid location or contain
    # bad data. Raise a ValueError if this happens.
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    productive_training_time = self._get_total_productive_training_time(entries)
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )
    return (float(productive_training_time) / total_job_time) * 100

  def get_job_goodput_interval(self, interval_start, interval_end):
    """Method to get the Goodput of the job within an interval window.

    If the application is interested in retrieving the Goodput of the job within
    a specific window of time, this method provides the singular Goodput
    computation between the start and end of this window.

    Args:
      interval_start: The start time of the window for which Goodput is to be
        computed.
      interval_end: The end time of the window for which Goodput is to be
        computed.

    Returns:
      Goodput percentage of the job within specified time window.
    """
    pass

  def get_job_badput_breakdown(self):
    """Method to get the the Badput breakdown of the job.

    This method provides a granular breakdown of the known components of Badput.

    Returns:
      A dictionary of badput components and their percentage breakdown within
      total job time.
    """
    badput_breakdown = {}
    entries = self._cloud_logger.read_cloud_logging_entries()
    total_job_time = self._get_total_job_time(entries)
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Badput cannot be calculated. Please fix the'
          ' logging entries.'
      )

    tpu_init_start_time = None
    training_prep_start_time = None
    tpu_initialization_badput = 0.0
    training_prep_badput = 0.0
    for payload in entries:
      # Compute badput due to TPU initialization.
      if _TPU_INIT_START_TIME in payload:
        tpu_init_start_time = payload[_TPU_INIT_START_TIME]
      elif (
          _TPU_INIT_END_TIME in payload and tpu_init_start_time is not None
      ):
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

    if (
        tpu_initialization_badput > total_job_time
        or tpu_initialization_badput < 0.0
    ):
      raise ValueError(
          'Total badput from TPU initialization is invalid. Please fix the'
          ' logging entries.'
      )

    badput_breakdown[BadputType.TPU_INITIALIZATION] = (
        tpu_initialization_badput / total_job_time
    ) * 100

    if training_prep_badput > total_job_time or training_prep_badput < 0.0:
      raise ValueError(
          'Total badput due to training preparation is invalid. Please fix the'
          ' logging entries.'
      )
    badput_breakdown[BadputType.TRAINING_PREP] = (
        training_prep_badput / total_job_time
    ) * 100

    return badput_breakdown
