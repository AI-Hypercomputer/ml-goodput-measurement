"""Goodput package API implementations.

This file contains all the core implementations of the ml_goodput_measurement
library for users to measure and monitor Goodput, Badput and Step Time
Deviation.
"""

import datetime
import functools
import logging
import threading
from typing import Any, Optional, Union

from cloud_goodput.ml_goodput_measurement.src import checkpoint_badput_calculator
from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_exclusion
from cloud_goodput.ml_goodput_measurement.src import goodput_utils


get_timestamp_from_log_entry = goodput_utils.get_timestamp_from_log_entry
get_extra_time_from_anomalous_steps = (
    goodput_utils.get_extra_time_from_anomalous_steps
)
compute_ideal_step_time = goodput_utils.compute_ideal_step_time
compute_baseline_step_time = goodput_utils.compute_baseline_step_time

BadputType = goodput_utils.BadputType
RestartType = goodput_utils.RestartType
CheckpointLoggerOptions = checkpoint_badput_calculator.CheckpointLoggerOptions
CheckpointBadputCalculator = (
    checkpoint_badput_calculator.CheckpointBadputCalculator
)
GoodputType = goodput_utils.GoodputType
GoodputCache = goodput_cache.GoodputCache
GoodputInfo = goodput_utils.GoodputInfo
MetricType = goodput_utils.MetricType
IntervalMetricType = goodput_utils.IntervalMetricType
StepInfo = goodput_utils.StepInfo
# Data structure to store the type of unproductive time (BadputType) and the
# corresponding time in seconds. If the BadputType is CUSTOM_BADPUT_EVENTS, the
# value is a dictionary of user defined event type and the corresponding time
# in seconds.
UnproductiveTimeDict = dict[
    BadputType, Union[float, dict[str, float]]
]
WorkloadMetricDetails = goodput_utils.WorkloadMetricDetails
IntervalWorkloadMetricDetails = goodput_utils.IntervalWorkloadMetricDetails

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
_CUSTOM_BADPUT_EVENT_TYPE = 'custom_badput_event_type'
_CUSTOM_BADPUT_EVENT_START_TIME = 'custom_badput_event_start_time'
_CUSTOM_BADPUT_EVENT_END_TIME = 'custom_badput_event_end_time'
_STARTUP_COMPILATION_WINDOW = 5

_CLOUD_LOGGING_PAGE_SIZE = 1000000
_CLOUD_LOGGING_DEFAULT_RETENTION = datetime.timedelta(days=7)

logger = logging.getLogger(__name__)


class _CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_name: Name of a specific job.
    logger: The Cloud Logging logger object.
    job_start_time: Start time of the job run.
  """

  def __init__(
      self,
      job_name: str,
      log_name: str,
      max_logs_retention_period: Optional[datetime.timedelta] = None,
      *,
      background_grace_period_s: float = 5.0,
      background_batch_size: int = 100,
      background_max_latency_s: float = 2.0,
  ):
    """_CloudLogger constructor.

    Writes are dispatched asynchronously via `CloudLoggingHandler` /
    `BackgroundThreadTransport`; `write_cloud_logging_entry` returns
    immediately and a daemon thread commits batched entries.

    Args:
      job_name: Name of the job the _CloudLogger is for.
      log_name: Name of the log being written.
      max_logs_retention_period: Maximum retention period for Cloud Logging
        logs.

    Keyword-only:
      background_grace_period_s: Shutdown drain budget. Default 5.0.
      background_batch_size: Entries per `write_entries` RPC. Default 100.
      background_max_latency_s: Max queue dwell after the first entry
        arrives, before committing. Default 2.0.
    """

    import google.cloud.logging  # pylint: disable=g-import-not-at-top
    from google.cloud.logging_v2.handlers import CloudLoggingHandler  # pylint: disable=g-import-not-at-top
    from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport  # pylint: disable=g-import-not-at-top

    self.job_name = job_name
    logging_client = google.cloud.logging.Client()
    self.project_id = logging_client.project
    self.log_name = log_name
    self.logger = logging_client.logger(log_name)
    self.job_start_time = None
    self.retention_period = (
        max_logs_retention_period
        if max_logs_retention_period
        else _CLOUD_LOGGING_DEFAULT_RETENTION
    )

    # Async write path: synchronous `Logger.log_struct` adds ~50-150 ms of
    # gRPC RTT per call, which compounds on the per-step recorder calls.
    # `CloudLoggingHandler` gives us a daemon-thread worker, batched commits,
    # and atexit-safe shutdown.
    transport_factory = functools.partial(
        BackgroundThreadTransport,
        grace_period=background_grace_period_s,
        batch_size=background_batch_size,
        max_latency=background_max_latency_s,
    )
    self._async_handler = CloudLoggingHandler(
        client=logging_client,
        name=log_name,
        transport=transport_factory,  # type: ignore[arg-type]
    )
    # One stdlib logger per _CloudLogger *instance* (not per log_name).
    # `logging.getLogger(name)` returns a process-global singleton, so if
    # two _CloudLogger instances are created with the same log_name in the
    # same process (sequential jobs in a notebook, reusable worker
    # processes, tests), the second `addHandler` call would attach a
    # second CloudLoggingHandler onto the shared logger and every entry
    # would be uploaded twice. Adding `id(self)` makes the name unique
    # per instance; propagate=False keeps dict payloads off the root
    # logger.
    self._async_logger = logging.getLogger(
        f'_ml_goodput_async.{log_name}.{id(self)}'
    )
    self._async_logger.setLevel(logging.INFO)
    self._async_logger.addHandler(self._async_handler)
    self._async_logger.propagate = False

  def write_cloud_logging_entry(self, entry) -> None:
    """Writes an entry asynchronously at INFO level.

    Args:
      entry: JSON-serializable structured log dictionary.
    """
    if entry is None:
      return
    if entry[_JOB_NAME] != self.job_name:
      return
    self._async_logger.info(entry)

  def flush(self) -> None:
    """Block until every entry enqueued so far has been committed."""
    self._async_handler.flush()

  def _get_filter_msg(
      self,
      start_time: Optional[datetime.datetime],
      end_time: Optional[datetime.datetime],
      last_entry_info: Optional[
          tuple[datetime.datetime, str]
      ] = None,  # (timestamp, id)
  ) -> str:
    """Gets the filter message for the Cloud Logging query."""
    filter_entries = []
    if self.project_id:
      full_log_name = f'projects/{self.project_id}/logs/{self.log_name}'
      filter_entries.append(f'logName="{full_log_name}"')

    filter_entries.append('severity=INFO')
    filter_entries.append(f'jsonPayload.job_name="{self.job_name}"')
    # Add a filter to bind an end-time to the query window.
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)
    elif end_time.tzinfo is None:
      end_time = end_time.replace(tzinfo=datetime.timezone.utc)

    filter_entries.append(f'timestamp<="{end_time.isoformat()}"')

    # Determine the effective start time and id for filtering.
    effective_start_timestamp: Optional[datetime.datetime] = None
    effective_start_entry_id: Optional[str] = None

    if last_entry_info:
      effective_start_timestamp, effective_start_entry_id = last_entry_info
    elif start_time is not None:
      effective_start_timestamp = start_time
      if effective_start_timestamp.tzinfo is None:
        effective_start_timestamp = effective_start_timestamp.replace(
            tzinfo=datetime.timezone.utc
        )
    elif self.job_start_time is not None:
      effective_start_timestamp = self.job_start_time - datetime.timedelta(
          seconds=10
      )
      if effective_start_timestamp.tzinfo is None:  # pyrefly: ignore[missing-attribute]
        effective_start_timestamp = effective_start_timestamp.replace(  # pyrefly: ignore[missing-attribute]
            tzinfo=datetime.timezone.utc
        )
    else:
      effective_start_timestamp = end_time - self.retention_period

    if effective_start_timestamp is not None:
      if effective_start_entry_id:
        filter_clause = (
            f'(timestamp > "{effective_start_timestamp.isoformat()}" OR'
            f' (timestamp = "{effective_start_timestamp.isoformat()}" AND'
            f' insertId > "{effective_start_entry_id}"))'
        )
        filter_entries.append(filter_clause)
      else:
        filter_entries.append(
            f'timestamp>"{effective_start_timestamp.isoformat()}"'
        )

    return ' AND '.join(filter_entries)

  def _update_job_start_time(self, entries: list[Any]):
    if self.job_start_time:
      return
    for entry in entries:
      if _JOB_START_TIME in entry and self.job_start_time is None:
        self.job_start_time = datetime.datetime.fromtimestamp(
            entry[_JOB_START_TIME], tz=datetime.timezone.utc
        )
        break

  def read_cloud_logging_entries(
      self,
      start_time: Optional[datetime.datetime] = None,
      end_time: Optional[datetime.datetime] = None,
      last_entry_info: Optional[tuple[datetime.datetime, str]] = None,
  ):
    """Queries Cloud Logging entries for the specific job.

    Reads following a write may see up to `background_max_latency_s`
    (2 s default) of staleness, since writes commit asynchronously.
    Call `GoodputRecorder.flush()` for a strict fence.

    Args:
      start_time: The start time of the query window.
      end_time: The end time of the query window.
      last_entry_info: The timestamp and unique identifier of the last entry
        previously read.

    Returns:
      Filtered entries in ascending order of timestamp.
    """
    import google.cloud.logging  # pylint: disable=g-import-not-at-top

    entries = self.logger.list_entries(
        filter_=self._get_filter_msg(start_time, end_time, last_entry_info),
        order_by=google.cloud.logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entries_info = [
        {
            'payload': entry.payload,
            'timestamp': entry.timestamp,
            'id': entry.insert_id,
        }
        for entry in entries
    ]

    if entries_info and last_entry_info:
      first_entry = entries_info[0]
      last_entry_ts, last_entry_id = last_entry_info
      if (
          first_entry['timestamp'] == last_entry_ts
          and first_entry['id'] == last_entry_id
      ):
        entries_info = entries_info[1:]

    current_last_entry_ts, current_last_entry_id = (
        (entries_info[-1]['timestamp'], entries_info[-1]['id'])
        if entries_info
        else (None, None)
    )

    entry_payload = [entry['payload'] for entry in entries_info]
    self._update_job_start_time(entry_payload)
    return entry_payload, (current_last_entry_ts, current_last_entry_id)


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
      *,
      background_grace_period_s: float = 5.0,
      background_batch_size: int = 100,
      background_max_latency_s: float = 2.0,
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

    Keyword-only:
      background_grace_period_s: Forwarded to the underlying `_CloudLogger`.
      background_batch_size: Forwarded to the underlying `_CloudLogger`.
      background_max_latency_s: Forwarded to the underlying `_CloudLogger`.
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
      self._cloud_logger = _CloudLogger(
          job_name,
          logger_name,
          background_grace_period_s=background_grace_period_s,
          background_batch_size=background_batch_size,
          background_max_latency_s=background_max_latency_s,
      )

  def flush(self) -> None:
    """Block until pending async writes from this recorder have committed.

    Use for strict read-after-write fencing (unit tests, end-of-training
    finalization, hand-off to another process).
    """
    if self._cloud_logger is None:
      return
    self._cloud_logger.flush()

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
      start_time = datetime.datetime.now(datetime.timezone.utc)

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
      start_time = datetime.datetime.now(datetime.timezone.utc)

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
      end_time = datetime.datetime.now(datetime.timezone.utc)

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
      start_time = datetime.datetime.now(datetime.timezone.utc)

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
      end_time = datetime.datetime.now(datetime.timezone.utc)

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
      start_time = datetime.datetime.now(datetime.timezone.utc)

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
      end_time = datetime.datetime.now(datetime.timezone.utc)

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
      start_time = datetime.datetime.now(datetime.timezone.utc)

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
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _DATA_LOADING_END_TIME: end_time.timestamp(),
    })

  def record_custom_badput_event_start_time(
      self,
      start_time: Optional[datetime.datetime] = None,
      custom_badput_event_type: str = 'unknown',
  ):
    """Main recorder function to log the start time of a custom badput event.

    Use this function to record the start time of a custom badput event that
    occurs inside the training loop and utilizes the accelerator resources,
    and blocks training.

    For example, use this API to record the start time of the evaluation
    loop or an SDC check if the the event blocks the training loop.

    Args:
      start_time: Start time of the custom badput event.
      custom_badput_event_type: Type of the custom badput event.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _CUSTOM_BADPUT_EVENT_TYPE: custom_badput_event_type,
        _CUSTOM_BADPUT_EVENT_START_TIME: start_time.timestamp(),
    })

  def record_custom_badput_event_end_time(
      self,
      end_time: Optional[datetime.datetime] = None,
      custom_badput_event_type: str = 'unknown',
  ):
    """Main recorder function to log the end time of a custom badput event.

    Args:
      end_time: End time of the custom badput event.
      custom_badput_event_type: Type of the custom badput event.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now(datetime.timezone.utc)

    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _CUSTOM_BADPUT_EVENT_TYPE: custom_badput_event_type,
        _CUSTOM_BADPUT_EVENT_END_TIME: end_time.timestamp(),
    })


class GoodputCalculator(goodput_exclusion.GoodputExclusion):
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
      max_logs_retention_period: Optional[datetime.timedelta] = None,
      gcs_path: Optional[str] = None,
      cache_dir: str = '/tmp',
  ):
    """GoodputCalculator constructor.

    Args:
      job_name: Name of the job the GoodputCalculator is for.
      logger_name: Name of the log being written.
      cloud_logger: Should never be passed directly by the user.
      using_pathways: Whether or not the job uses Pathways.
      max_logs_retention_period: Optional retention period for query of Cloud
        Logging entries.
      gcs_path: Optional GCS path to sync the cache to.
      cache_dir: Local directory to store the cache files.
    """
    self.job_name = job_name
    self.using_pathways = using_pathways
    self._interval_cache = []
    self._last_interval_entry_info: Optional[
        tuple[datetime.datetime, str]
    ] = None
    self._last_interval_start_time: Optional[datetime.datetime] = None
    self._last_interval_end_time: Optional[datetime.datetime] = None
    if cloud_logger is not None:
      self._cloud_logger = cloud_logger
    else:
      self._cloud_logger = _CloudLogger(
          job_name,
          logger_name,
          max_logs_retention_period=max_logs_retention_period,
      )
    self._goodput_cache = GoodputCache(
        job_name=job_name,
        gcs_path=gcs_path,
        cache_dir=cache_dir,
    )
    self._goodput_cache_lock = threading.Lock()
    self._interval_entries = []
    self._interval_start_time = None
    self._interval_end_time = None
    self._number_of_interruptions = 0
    self._gcm_last_recorded_timestamp = None
    self._last_disruption_time = None  # pyrefly: ignore[bad-assignment]
    self._last_disrupted_step = None  # pyrefly: ignore[bad-assignment]
    self._historical_step_times = {}

  def sync_to_gcs(self):
    """Syncs the underlying cache files to GCS."""
    with self._goodput_cache_lock:
      self._goodput_cache.sync_to_gcs()

  def _get_total_productive_and_unproductive_time(
      self, new_entries: list[dict[str, Any]]
  ) -> tuple[float, UnproductiveTimeDict, int, int]:
    """Helper function to compute the total productive and unproductive time.

    Args:
      new_entries: A list of new log entries to process.

    Returns:
      A tuple of:
        - total productive training time
        - total unproductive time
        - max productive step
        - last recorded step
    """
    # If no new entries are present, return last computed values.
    if not new_entries:
      cached_values = self._get_cached_productive_and_unproductive_time()
      if cached_values is not None:
        return cached_values

    return self._get_current_productive_and_unproductive_time()

  def _get_cached_productive_and_unproductive_time(
      self,
  ) -> tuple[float, UnproductiveTimeDict, int, int] | None:
    """Helper function to retrieve the cached productive training time and unproductive time."""
    with self._goodput_cache_lock:
      goodput_info = self._goodput_cache.get_goodput_info()
      if not self._goodput_cache.is_cache_empty() and goodput_info is not None:
        return (
            goodput_info.total_productive_time,
            goodput_info.total_unproductive_time,
            goodput_info.max_productive_step,
            goodput_info.last_recorded_step,
        )
    return None

  def _accumulate_unproductive_time(
      self,
      segment_unproductive_time: UnproductiveTimeDict,
      total_unproductive_time: UnproductiveTimeDict,
  ):
    """Helper function to accumulate the segment unproductive time.

    Args:
      segment_unproductive_time: A dictionary of unproductive time for a
        segment.
      total_unproductive_time: A dictionary of total unproductive time.

    Returns:
      None. The function updates the total_unproductive_time dictionary.
    """

    for badput_type, unproductive_value in segment_unproductive_time.items():
      if isinstance(unproductive_value, dict):
        if badput_type not in total_unproductive_time:
          total_unproductive_time[badput_type] = dict(unproductive_value)
        else:
          existing_value = total_unproductive_time[badput_type]
          if isinstance(existing_value, dict):
            for sub_type, sub_value in unproductive_value.items():
              existing_value[sub_type] = (
                  existing_value.get(sub_type, 0.0) + sub_value
              )
      else:
        if badput_type in total_unproductive_time:
          existing_value = total_unproductive_time[badput_type]
          if isinstance(existing_value, float):
            total_unproductive_time[badput_type] = (
                existing_value + unproductive_value
            )
        else:
          total_unproductive_time[badput_type] = unproductive_value

  def _extract_custom_sync_intervals(
      self,
      entries: list[dict[str, Any]],
  ) -> list[tuple[float, float, str]]:
    """Extracts custom badput intervals from Cloud Logging entries.

    This helper function scans through a list of Cloud Logging entries to find
    custom badput start and end times, pairing them into intervals.

    Args:
        entries: A list of dictionaries representing Cloud Logging entries. Each
          entry may contain keys indicating the start or end of a custom badput
          event.

    Returns:
        A list of tuples, where each tuple consists of:
            - start_time (float): The timestamp when the sync event started.
            - end_time (float): The timestamp when the sync event ended.
            - sync_type (str): The type of custom sync
            event.
    """
    intervals = []
    active_syncs = {}

    for entry in entries:
      if _CUSTOM_BADPUT_EVENT_START_TIME in entry:
        sync_type = entry[_CUSTOM_BADPUT_EVENT_TYPE].upper()
        active_syncs[sync_type] = entry[_CUSTOM_BADPUT_EVENT_START_TIME]
      elif _CUSTOM_BADPUT_EVENT_END_TIME in entry:
        sync_type = entry[_CUSTOM_BADPUT_EVENT_TYPE].upper()
        if sync_type in active_syncs:
          start_time = active_syncs.pop(sync_type)
          end_time = entry[_CUSTOM_BADPUT_EVENT_END_TIME]
          if start_time < end_time:
            intervals.append((start_time, end_time, sync_type))

    return intervals

  def _get_restart_event(
      self, payload: dict[str, Any]
  ) -> Optional[tuple[float, RestartType]]:
    """Returns `(timestamp, RestartType)` if `payload` marks a job restart.

    Subclasses (such as `ElasticGoodputCalculator`) override this hook to
    recognize additional in-process restart events (e.g. `elastic_wait_start`
    or `elastic_reinit_start`) when `job_start_time` is not emitted.

    Args:
      payload: Dictionary payload of a Cloud Logging entry.

    Returns:
      A tuple of `(restart_timestamp, RestartType)` if the entry indicates a
      restart, or None otherwise.
    """
    if _JOB_START_TIME in payload:
      return float(payload[_JOB_START_TIME]), RestartType.INFRA_RESTART
    return None

  def _extract_checkpoint_save_events(
      self, entries: list[Any]
  ) -> list[tuple[int, bool, Optional[float], float]]:
    """Extracts `(save_step, is_local, save_start_time, save_duration)` tuples.

    Args:
      entries: A list of Cloud Logging entry payloads.

    Returns:
      A list of tuples `(save_step, is_local, save_start_time, save_duration)`
      for all checkpoint save operations with positive blocking duration.
    """
    save_events = []
    for entry in entries:
      if not isinstance(entry, dict):
        continue
      if (
          entry.get('event_type') == 'save'
          and 'step' in entry
          and entry.get('step') is not None
      ):
        save_step = int(entry['step'])
        is_local = not str(entry.get('directory', '')).startswith('gs://')
        raw_start_time = entry.get('checkpoint_manager_blocking_start_time')
        save_start_time = (
            float(raw_start_time) if raw_start_time is not None else None
        )
        save_duration = float(
            entry.get('checkpoint_manager_blocking_duration_secs') or 0.0
        )
        if save_duration > 0.0:
          save_events.append(
              (save_step, is_local, save_start_time, save_duration)
          )
    return save_events

  @staticmethod
  def _get_checkpoint_save_by_type_in_window(
      save_events: list[tuple[int, bool, Optional[float], float]],
      step_num: int,
      window_start_time: float,
      window_end_time: Optional[float] = None,
  ) -> dict[bool, float]:
    """Computes average blocking save duration per locality within a window.

    When multiple workers log a checkpoint save for the same `(step_num,
    is_local)` within the step execution window, their durations are averaged
    (matching `CheckpointBadputCalculator`), while local and persistent saves
    are tracked separately.

    Args:
      save_events: Extracted `(save_step, is_local, save_start_time,
        save_duration)` tuples.
      step_num: The training step number to match.
      window_start_time: Inclusive start timestamp of the step execution window.
      window_end_time: Optional exclusive end timestamp of the step execution
        window.

    Returns:
      A dictionary mapping `is_local` (`bool`) to the average blocking save
      duration (in seconds) for `step_num` within `[window_start_time,
      window_end_time)`.
    """
    durations_by_locality: dict[bool, list[float]] = {}
    for save_step, is_local, save_start_time, save_duration in save_events:
      if save_step != step_num:
        continue
      if save_start_time is None or (
          save_start_time >= window_start_time
          and (window_end_time is None or save_start_time < window_end_time)
      ):
        durations_by_locality.setdefault(is_local, []).append(save_duration)
    return {
        is_local: sum(durations) / len(durations)
        for is_local, durations in durations_by_locality.items()
        if durations
    }

  @classmethod
  def _get_checkpoint_save_duration_in_window(
      cls,
      save_events: list[tuple[int, bool, Optional[float], float]],
      step_num: int,
      window_start_time: float,
      window_end_time: Optional[float] = None,
  ) -> float:
    """Computes synchronous checkpoint save blocking duration within a window.

    Args:
      save_events: Extracted `(save_step, is_local, save_start_time,
        save_duration)` tuples.
      step_num: The training step number to match.
      window_start_time: Inclusive start timestamp of the step execution window.
      window_end_time: Optional exclusive end timestamp of the step execution
        window.

    Returns:
      Total synchronous checkpoint save blocking duration (in seconds) for
      `step_num` that started within `[window_start_time, window_end_time)`.
    """
    return sum(
        cls._get_checkpoint_save_by_type_in_window(
            save_events, step_num, window_start_time, window_end_time
        ).values()
    )

  def _compute_segment_startup_overhead(
      self, step_times: list[float], min_step: int
  ) -> float:
    """Computes XLA compilation/startup extra time for a segment of steps.

    Uses the current segment's step times to establish a steady-state baseline
    when `len(step_times) > 1`, or falls back to prior segments' historical step
    times when a post-restart segment has only 1 completed step. Also updates
    `self._historical_step_times` for the compilation step to exclude the
    startup overhead.

    Args:
      step_times: List of completed step durations (with custom sync and
        checkpoint save blocking times already removed) in the current segment.
      min_step: The starting step number of the current segment.

    Returns:
      The excess startup/compilation duration (`PROGRAM_STARTUP` badput) in
      seconds.
    """
    steps_in_segment = len(step_times)
    if steps_in_segment == 0:
      return 0.0

    # Attribute the maximum excess step time in the startup window to program
    # startup.
    startup_window = min(_STARTUP_COMPILATION_WINDOW, steps_in_segment)
    prior_step_times = [
        duration
        for step_idx, duration in self._historical_step_times.items()
        if step_idx < min_step or step_idx >= min_step + steps_in_segment
    ]
    if steps_in_segment > 1:
      baseline_step_time = compute_baseline_step_time(step_times)
    elif prior_step_times:
      # When a post-restart segment has only 1 completed step before another
      # disruption, use prior segments' historical step times as the baseline so
      # compilation overhead on that single step is not miscounted as goodput.
      baseline_step_time = compute_baseline_step_time(prior_step_times)
    else:
      baseline_step_time = step_times[0]

    compilation_step_offset = max(
        range(startup_window), key=lambda i: step_times[i]
    )
    has_reference_baseline = steps_in_segment > 1 or bool(prior_step_times)
    startup_extra_time = (
        max(0.0, step_times[compilation_step_offset] - baseline_step_time)
        if has_reference_baseline
        else 0.0
    )

    # Adjust the compilation step's historical time to exclude startup overhead.
    compilation_step = min_step + compilation_step_offset
    if (
        startup_extra_time > 0.0
        and compilation_step in self._historical_step_times
    ):
      self._historical_step_times[compilation_step] -= startup_extra_time

    return startup_extra_time

  def _estimate_sequential_restart_step_time(
      self,
      prev_step: int,
      segment_productive_time: float,
      num_completed_steps: int,
  ) -> float:
    """Estimates the productive duration of the tail step before a restart.

    When a job checkpoints at `prev_step`, restarts, and resumes sequentially at
    `prev_step + 1`, `prev_step` has no subsequent step in the closed segment.
    We estimate its execution time from the segment's average productive step
    time (or historical step times if `prev_step` was the only step in the
    segment).

    Args:
      prev_step: The step number of the last step before the sequential restart.
      segment_productive_time: Total productive time of completed steps in the
        closed segment.
      num_completed_steps: Number of completed step intervals in the closed
        segment.

    Returns:
      Estimated productive execution duration (in seconds) for `prev_step`.
    """
    if num_completed_steps > 0:
      return segment_productive_time / num_completed_steps
    if prev_step in self._historical_step_times:
      return self._historical_step_times[prev_step]
    if self._historical_step_times:
      return compute_baseline_step_time(
          list(self._historical_step_times.values())
      )
    return 0.0

  def _compute_salvaged_historical_time(
      self,
      disrupted_segment_min_step: int,
      resume_step: int,
      completed_productive_steps: set[int],
  ) -> float:
    """Computes salvaged productive time across a jump-forward rollback.

    When an earlier run progressed past `resume_step`, a second run rolled back
    to `disrupted_segment_min_step`, and a third run resumed from a snapshot at
    `resume_step`, steps in `[disrupted_segment_min_step, resume_step)` were
    already completed in the earlier run and are salvaged from
    `self._historical_step_times`.

    Args:
      disrupted_segment_min_step: The starting step of the disrupted segment.
      resume_step: The step at which the new run resumed (`curr_step`).
      completed_productive_steps: Set of productive step numbers to update.

    Returns:
      Total salvaged productive time (in seconds) from historical steps.
    """
    salvaged_historical_productive_time = 0.0
    for step_idx in range(disrupted_segment_min_step, resume_step):
      salvaged_historical_productive_time += self._historical_step_times.get(
          step_idx, 0.0
      )
      completed_productive_steps.add(step_idx)
    return salvaged_historical_productive_time

  @staticmethod
  def _compute_remaining_checkpoint_save_badput(
      save_events: list[tuple[int, bool, Optional[float], float]],
      completed_productive_steps: set[int],
      deducted_save_keys: set[tuple[int, bool]],
  ) -> float:
    """Sums checkpoint save durations for completed steps not deducted in-window.

    Serves as a fallback when checkpoint save logs omit start timestamps or use
    synthetic timestamps outside step windows, while ensuring uncompleted tail
    steps never have checkpoint save time subtracted from productive time.
    Tracks `(save_step, is_local)` separately and averages across multiple
    worker occurrences per key to match `CheckpointBadputCalculator`.

    Args:
      save_events: Extracted `(save_step, is_local, save_start_time,
        save_duration)` tuples.
      completed_productive_steps: Set of steps that completed productively.
      deducted_save_keys: Set of `(save_step, is_local)` keys whose checkpoint
        save time was already deducted in-window.

    Returns:
      Remaining checkpoint save blocking time (in seconds) to deduct from total
      productive training time.
    """
    remaining_by_key: dict[tuple[int, bool], list[float]] = {}
    for save_step, is_local, _, save_duration in save_events:
      save_key = (save_step, is_local)
      if (
          save_step in completed_productive_steps
          and save_key not in deducted_save_keys
      ):
        remaining_by_key.setdefault(save_key, []).append(save_duration)
    return sum(
        sum(durations) / len(durations)
        for durations in remaining_by_key.values()
        if durations
    )

  def _get_current_productive_and_unproductive_time(
      self, interval_query: Optional[bool] = False
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
      int,
  ]:
    """Helper function to compute the current productive training time, unproductive time and the last step recorded till now.

    Args:
      interval_query: A boolean value to indicate whether the current query is
        for an interval or not.

    Returns:
      A tuple of the productive training time, the unproductive time
      (dict of BadputType and unproductive time), the last productive step and
      the last recorded step.
    """
    if interval_query:
      entries_to_process = self._interval_entries
    else:
      with self._goodput_cache_lock:
        entries_to_process = list(self._goodput_cache.get_cached_entries())

    save_events = self._extract_checkpoint_save_events(entries_to_process)
    deducted_save_keys: set[tuple[int, bool]] = set()
    completed_productive_steps: set[int] = set()

    def _compute_adjusted_segment_productive_and_unproductive_time(
        step_items: list[tuple[int, float]],
        curr_step: int,
        min_step: int,  # pylint: disable=unused-argument
        custom_sync_intervals: list[tuple[float, float, str]],
    ) -> tuple[
        float,
        list[float],
        float,
        dict[str, float],
        int,
        int,
    ]:
      """Computes adjusted productive and unproductive time for a segment of steps.

      This helper function calculates the total productive time, and the
      breakdown of time lost due to custom badput events and synchronous
      checkpoint saves, as well as wasted progress caused by disruptions.

      Args:
          step_items: A list of tuples, where each tuple contains a step number
            (int) and its start timestamp (float).
          curr_step: The current step number indicating the end of the segment.
          min_step: The minimum step number indicating the start of the segment.
          custom_sync_intervals: A list of (start_time, end_time, sync_type)
            tuples representing custom sync events.

      Returns:
          A tuple containing:
            - total_productive_time (float): Adjusted time excluding custom
                sync and synchronous checkpoint save durations.
            - step_times (list[float]): List of adjusted times for all
                productive steps in the segment.
            - wasted_progress (float): Total unproductive time due to possible
                disruptions.
            - custom_sync_breakdown (dict[str, float]):
                Breakdown of time spent in each custom sync type.
            - steps_in_segment (int): Total number of steps considered in the
                segment.
            - max_productive_step_count (int): Maximum step count in the
                segment that is considered productive.
      """
      total_productive_time = 0.0
      step_times = []
      wasted_progress = 0.0
      custom_sync_breakdown: dict[str, float] = {}

      steps_in_segment = 0
      max_productive_step_count = 0

      for i in range(1, len(step_items)):
        prev_step, prev_time = step_items[i - 1]
        curr_step_num, curr_time = step_items[i]

        raw_delta = curr_time - prev_time

        custom_sync_in_interval = 0.0
        for sync_start, sync_end, sync_type in custom_sync_intervals:
          if prev_time <= sync_start and sync_end <= curr_time:
            sync_duration = sync_end - sync_start
            custom_sync_in_interval += sync_duration
            if curr_step_num <= curr_step:
              custom_sync_breakdown[sync_type] = (
                  custom_sync_breakdown.get(sync_type, 0.0) + sync_duration
              )

        adjusted_delta = max(0.0, raw_delta - custom_sync_in_interval)

        # Deduct synchronous checkpoint save blocking duration that occurred
        # during `prev_step` before computing startup overhead so save time is
        # not misclassified as `PROGRAM_STARTUP` or subtracted twice.
        save_by_type = self._get_checkpoint_save_by_type_in_window(
            save_events, prev_step, prev_time, curr_time
        )
        checkpoint_save_in_step = sum(save_by_type.values())
        if (
            checkpoint_save_in_step > 0.0
            and adjusted_delta >= checkpoint_save_in_step
        ):
          adjusted_delta -= checkpoint_save_in_step
          if curr_step_num <= curr_step and curr_step_num - 1 == prev_step:
            for is_local in save_by_type:
              deducted_save_keys.add((prev_step, is_local))

        if curr_step_num <= curr_step:
          self._historical_step_times[prev_step] = adjusted_delta
          if curr_step_num - 1 != prev_step:
            continue

          total_productive_time += adjusted_delta
          step_times.append(adjusted_delta)
          steps_in_segment += 1
          max_productive_step_count = prev_step
          completed_productive_steps.add(prev_step)

        else:
          # Preserve rolled-back step times for potential jump-forward salvage
          # only if not already recorded by an earlier productive segment.
          if prev_step not in self._historical_step_times:
            self._historical_step_times[prev_step] = adjusted_delta
          if (
              checkpoint_save_in_step > 0.0
              and raw_delta >= checkpoint_save_in_step
          ):
            for is_local in save_by_type:
              deducted_save_keys.add((prev_step, is_local))
          # These steps are after curr_step, they are lost due to disruption.
          wasted_progress += max(0.0, raw_delta - checkpoint_save_in_step)

      return (
          total_productive_time,
          step_times,
          wasted_progress,
          custom_sync_breakdown,
          steps_in_segment,
          max_productive_step_count,
      )

    def _compute_segment_final_metrics(
        total_productive_time: float,
        startup_extra_time: float,
        wasted_progress: float,
        custom_sync_breakdown: dict[str, float],
    ) -> tuple[
        float,
        UnproductiveTimeDict,
    ]:
      """Computes final metrics for a segment, separating productive and unproductive time.

      Args:
          total_productive_time: Total productive time for the segment.
          startup_extra_time: Total excess time attributed to program startup.
          wasted_progress: Total time lost due to step discontinuities.
          custom_sync_breakdown: A dictionary mapping each custom sync type to
            the total badput time it accounted for during the segment.

      Returns:
          A tuple containing:
              - final_productive_time (float)
              - total_segment_unproductive_time (dict)
      """
      final_productive_time = total_productive_time - startup_extra_time
      total_segment_unproductive_time = {
          BadputType.PROGRAM_STARTUP: startup_extra_time,
          BadputType.WASTED_PROGRESS_FROM_DISRUPTION: wasted_progress,
          BadputType.CUSTOM_BADPUT_EVENTS: custom_sync_breakdown,
      }
      return final_productive_time, total_segment_unproductive_time

    def _get_segment_productive_and_unproductive_time(
        step_start_data: dict[int, float],
        curr_step: int,
        entries_to_process: list[Any],
    ) -> tuple[
        float,
        UnproductiveTimeDict,
        int,
    ]:
      if curr_step == 0:
        return 0.0, {}, 0

      step_items = list(step_start_data.items())
      min_step = min(step_start_data.keys())

      # Extract custom sync intervals
      custom_sync_intervals = self._extract_custom_sync_intervals(
          entries_to_process
      )

      # Compute adjusted segment productive and unproductive times
      (
          total_productive_time,
          step_times,
          wasted_progress_from_disruption,
          custom_sync_breakdown,
          steps_in_segment,
          max_productive_step_count,
      ) = _compute_adjusted_segment_productive_and_unproductive_time(
          step_items, curr_step, min_step, custom_sync_intervals
      )

      if steps_in_segment == 0:
        return (
            0.0,
            {
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION: (
                    wasted_progress_from_disruption
                )
            },
            0,
        )

      startup_extra_time = self._compute_segment_startup_overhead(
          step_times, min_step
      )

      # Compute final segment metrics.
      (
          final_adjusted_productive_time,
          total_segment_unproductive_time,
      ) = _compute_segment_final_metrics(
          total_productive_time,
          startup_extra_time,
          wasted_progress_from_disruption,
          custom_sync_breakdown,
      )

      return (
          final_adjusted_productive_time,
          total_segment_unproductive_time,
          max_productive_step_count,
      )

    # Build a deserialized dictionary from cloud logging entries to store step
    # start times. The dictionary maps from step count to start time and will be
    # used to compute each step's productive time by looking for its completion
    # in the next step's start.
    # Note in the instance where progress is lost due to a disruption and the
    # last successful checkpoint did not include all the steps, the last set of
    # records of the step information will be kept and the previous set will be
    # overwritten by design so as to correct for the previously computed
    # additional time that was counted as productive but lost due to a
    # disruption.
    productive_training_time = 0.0
    total_unproductive_time = {}
    step_start_data = {}
    pending_restart_time: Optional[float] = None
    job_end_time = None
    tpu_init_start_time = None
    training_prep_start_time = None
    data_loading_start_time = None
    tpu_initialization_badput = 0.0
    training_prep_badput = 0.0
    data_loading_badput = 0.0
    sync_data_loading = True
    current_sync_data_loading = None

    def _add_badput_to_segment(
        seg_unprod: UnproductiveTimeDict, badput_type: BadputType, amount: float
    ) -> None:
      """Adds `amount` to `badput_type` inside `seg_unprod` if positive."""
      if amount <= 0.0:
        return
      existing = seg_unprod.get(badput_type, 0.0)
      val = existing if isinstance(existing, (int, float)) else 0.0
      seg_unprod[badput_type] = val + amount

    def _handle_segment_restart(
        mode: str,
        curr_step: int,
        prev_step: int,
        restart_ts: Optional[float],
    ) -> None:
      """Closes a disrupted step segment and accumulates its Goodput/Badput.

      Handles three restart resume modes:
        - `jump_forward`: The workload resumed at `curr_step > prev_step + 1`
          using a checkpoint/snapshot from an earlier run. Salvages historical
          step times for `[disrupted_segment_min_step, curr_step)`.
        - `sequential`: A restart occurred between `prev_step` and
          `curr_step == prev_step + 1`. Credits `prev_step` with an estimated
          productive duration so the last step before checkpoint/restart is not
          discarded.
        - `duplicate`: The workload rolled back to a previously executed step
          (`curr_step <= prev_step`). All steps `< curr_step` are preserved as
          productive, while steps `>= curr_step` become wasted progress.

      Unproductive time from the disruption is separated into two buckets:
        1. Wasted training progress (`WASTED_PROGRESS_FROM_DISRUPTION`) and
           infrastructure downtime until the restart begins
           (`INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION`).
        2. Re-initialization overhead (`TPU_INITIALIZATION`, `TRAINING_PREP`,
           `DATA_LOADING_SYNC`, `PROGRAM_STARTUP`, checkpoint restore) after
           the restart begins.

      Args:
        mode: Restart resume mode (`'jump_forward'`, `'sequential'`, or
          `'duplicate'`).
        curr_step: The step number at which training resumed after the restart.
        prev_step: The last recorded step number in the disrupted segment.
        restart_ts: Timestamp of the earliest restart event following
          `prev_step`, if available.
      """
      nonlocal productive_training_time, sync_data_loading, current_sync_data_loading
      self._number_of_interruptions += 1
      self._last_disrupted_step = prev_step
      self._last_disruption_time = step_start_data[prev_step]

      boundary_step = prev_step if mode == 'sequential' else curr_step
      (
          segment_productive_time,
          segment_unproductive_time,
          _,
      ) = _get_segment_productive_and_unproductive_time(
          step_start_data, boundary_step, entries_to_process
      )

      estimated_last_step_time = 0.0
      if mode == 'jump_forward':
        disrupted_segment_min_step = min(step_start_data.keys())
        salvaged_historical_productive_time = (
            self._compute_salvaged_historical_time(
                disrupted_segment_min_step,
                curr_step,
                completed_productive_steps,
            )
        )
        # Adjust wasted progress: we lost the progress in the current segment
        # (`segment_productive_time`) but recovered
        # `salvaged_historical_productive_time` from the earlier run.
        segment_unproductive_time[
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION
        ] = (segment_productive_time - salvaged_historical_productive_time)
        productive_training_time += salvaged_historical_productive_time
      elif mode == 'sequential':
        num_completed_steps = len(step_start_data) - 1
        estimated_last_step_time = self._estimate_sequential_restart_step_time(
            prev_step, segment_productive_time, num_completed_steps
        )
        productive_training_time += (
            segment_productive_time + estimated_last_step_time
        )
        if estimated_last_step_time > 0.0:
          self._historical_step_times[prev_step] = estimated_last_step_time
      else:
        # Duplicate-step rollback: steps >= curr_step were rolled back. If
        # `curr_step` rolled back past the start of the current segment
        # (`disrupted_segment_min_step`), also move previously completed
        # historical steps in `[curr_step, disrupted_segment_min_step)` from
        # productive time to wasted progress.
        disrupted_segment_min_step = min(step_start_data.keys())
        rolled_back_historical_time = 0.0
        for step_idx in range(curr_step, disrupted_segment_min_step):
          if step_idx in completed_productive_steps:
            rolled_back_historical_time += self._historical_step_times.get(
                step_idx, 0.0
            )
        for step_idx in list(completed_productive_steps):
          if step_idx >= curr_step:
            completed_productive_steps.discard(step_idx)
        if rolled_back_historical_time > 0.0:
          productive_training_time = max(
              0.0, productive_training_time - rolled_back_historical_time
          )
          _add_badput_to_segment(
              segment_unproductive_time,
              BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
              rolled_back_historical_time,
          )
        productive_training_time += segment_productive_time

      # When the job restarts, the first data loading event is synchronous.
      sync_data_loading = True
      if current_sync_data_loading is not None:
        _add_badput_to_segment(
            segment_unproductive_time,
            BadputType.DATA_LOADING_SYNC,
            current_sync_data_loading,
        )
        current_sync_data_loading = None

      # Compute infrastructure recovery downtime between the disrupted step
      # start time and the restart timestamp, excluding any credited productive
      # step duration and synchronous checkpoint save time on `prev_step`.
      if (
          restart_ts is not None
          and self._last_disruption_time is not None
          and restart_ts > self._last_disruption_time
      ):
        disrupted_save_by_type = self._get_checkpoint_save_by_type_in_window(
            [e for e in save_events if e[2] is not None],
            prev_step,
            self._last_disruption_time,
            restart_ts,
        )
        disrupted_step_checkpoint_save_time = sum(
            disrupted_save_by_type.values()
        )
        for is_local in disrupted_save_by_type:
          deducted_save_keys.add((prev_step, is_local))
        infrastructure_disruption_badput = max(
            0.0,
            (restart_ts - self._last_disruption_time)
            - estimated_last_step_time
            - disrupted_step_checkpoint_save_time,
        )
        _add_badput_to_segment(
            segment_unproductive_time,
            BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION,
            infrastructure_disruption_badput,
        )

      self._accumulate_unproductive_time(
          segment_unproductive_time, total_unproductive_time
      )

    self._number_of_interruptions = 0
    for payload in entries_to_process:
      restart_event = self._get_restart_event(payload)
      if restart_event is not None:
        restart_ts, restart_type = restart_event
        if step_start_data:
          last_step_ts = step_start_data[list(step_start_data.keys())[-1]]
          if restart_ts > last_step_ts:
            # Keep the earliest restart timestamp after `last_step_ts` (e.g.
            # `elastic_wait_start_time` before `elastic_reinit_start_time`),
            # unless overridden by a full `INFRA_RESTART` (`job_start_time`).
            if (
                pending_restart_time is None
                or pending_restart_time <= last_step_ts
                or restart_type == RestartType.INFRA_RESTART
            ):
              pending_restart_time = restart_ts
            sync_data_loading = True
            current_sync_data_loading = None
        else:
          pending_restart_time = restart_ts

      if _STEP_START_TIME in payload:
        curr_step = int(payload[_STEP_COUNT])
        if not step_start_data:
          step_start_data[curr_step] = payload[_STEP_START_TIME]
          pending_restart_time = None
        else:
          prev_step = list(step_start_data.keys())[-1]
          if curr_step > prev_step + 1:
            _handle_segment_restart(
                'jump_forward', curr_step, prev_step, pending_restart_time
            )
            step_start_data = {curr_step: payload[_STEP_START_TIME]}
            pending_restart_time = None
          elif curr_step <= prev_step:
            _handle_segment_restart(
                'duplicate', curr_step, prev_step, pending_restart_time
            )
            step_start_data = {curr_step: payload[_STEP_START_TIME]}
            pending_restart_time = None
          elif (
              pending_restart_time is not None
              and pending_restart_time > step_start_data[prev_step]
          ):
            _handle_segment_restart(
                'sequential', curr_step, prev_step, pending_restart_time
            )
            step_start_data = {curr_step: payload[_STEP_START_TIME]}
            pending_restart_time = None
          else:
            step_start_data[curr_step] = payload[_STEP_START_TIME]
            pending_restart_time = None

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
        data_loading_end_time = payload[_DATA_LOADING_END_TIME]
        current_sync_data_loading = (
            data_loading_end_time - data_loading_start_time
        )
        data_loading_badput += current_sync_data_loading
        if sync_data_loading:
          # When the job starts or restarts, data loading is synchronous.
          total_unproductive_time[BadputType.DATA_LOADING_SYNC] = (
              total_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0)
              + current_sync_data_loading
          )
          sync_data_loading = False
          current_sync_data_loading = None
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

    # Populate async data loading badput.
    async_data_loading_badput = (
        data_loading_badput
        - total_unproductive_time.get(BadputType.DATA_LOADING_SYNC, 0)
    )
    total_unproductive_time[BadputType.DATA_LOADING_ASYNC] = (
        async_data_loading_badput
    )

    # Populate checkpoint manager save and restore badput.
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME] = (
        checkpoint_manager_save_badput
    )
    total_unproductive_time[BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME] = (
        checkpoint_manager_restore_badput
    )

    if not step_start_data:
      return 0.0, total_unproductive_time, 0, 0

    last_step = max(list(step_start_data.keys()))
    (
        segment_productive_time,
        segment_unproductive_time,
        max_productive_step_count,
    ) = _get_segment_productive_and_unproductive_time(
        step_start_data, last_step, entries_to_process
    )
    productive_training_time += segment_productive_time
    self._accumulate_unproductive_time(
        segment_unproductive_time, total_unproductive_time
    )

    # Only consider the last step productive if the job has completed.
    if job_end_time is not None:
      last_step_delta = job_end_time - step_start_data[last_step]
      last_step_save_by_type = self._get_checkpoint_save_by_type_in_window(
          save_events, last_step, step_start_data[last_step], job_end_time
      )
      last_step_checkpoint_save = sum(last_step_save_by_type.values())
      if (
          last_step_checkpoint_save > 0.0
          and last_step_delta >= last_step_checkpoint_save
      ):
        last_step_delta -= last_step_checkpoint_save
        for is_local in last_step_save_by_type:
          deducted_save_keys.add((last_step, is_local))
      productive_training_time += last_step_delta
      max_productive_step_count = last_step
      completed_productive_steps.add(last_step)

    # Deduct any remaining checkpoint save time that occurred on completed
    # productive steps but whose timestamp was not already deducted in-window.
    remaining_completed_save_badput = (
        self._compute_remaining_checkpoint_save_badput(
            save_events, completed_productive_steps, deducted_save_keys
        )
    )
    if remaining_completed_save_badput > 0.0:
      productive_training_time = max(
          0.0, productive_training_time - remaining_completed_save_badput
      )

    # Return a tuple of the total productive training time, the total
    # unproductive time (dict of BadputType and unproductive time) and the last
    # step recorded.
    return (
        productive_training_time,
        total_unproductive_time,
        max_productive_step_count,
        last_step,
    )

  def _get_total_job_time(self, query_time: datetime.datetime) -> float:
    """Helper function to compute the current job runtime.

    Args:
      query_time: The time at which the query is being made.

    Returns:
      The job's total runtime computed based on the last retrieved logs.
    """
    # Find the job's original start time from the cache.
    with self._goodput_cache_lock:
      start_time = self._goodput_cache.get_job_start_time()
      end_time = self._goodput_cache.get_job_end_time()
      if start_time:
        if not end_time:
          end_time = query_time
        return end_time.timestamp() - start_time.timestamp()

    # De-serialize job start and end times from cloud logging entries. These
    # will be used to compute total runtime of the job.
    job_start_time = None
    job_end_time = None
    with self._goodput_cache_lock:
      cached_entries = list(self._goodput_cache.get_cached_entries())
    for payload in cached_entries:
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
      # current query time to compute total job time.
      return query_time.timestamp() - job_start_time
    # The the job's start time is missing so the total job time cannot be
    # calculated. Caller of this function should raise an error if this happens.
    return 0.0

  def _fetch_new_entries(self, query_time: datetime.datetime) -> list[Any]:
    """Thread-safe helper function to update and return new log entries."""
    new_entries = []
    current_last_entry_info = None
    with self._goodput_cache_lock:
      if not self._goodput_cache.is_cache_empty():
        cached_last_entry_info = self._goodput_cache.get_last_entry_info()
        if cached_last_entry_info:
          cached_last_entry_ts, _ = cached_last_entry_info
          if (
              cached_last_entry_ts is not None
              and query_time <= cached_last_entry_ts
          ):
            return []

          new_entries, current_last_entry_info = (
              self._cloud_logger.read_cloud_logging_entries(
                  start_time=cached_last_entry_ts,
                  end_time=query_time,
                  last_entry_info=cached_last_entry_info,
              )
          )
      else:
        new_entries, current_last_entry_info = (
            self._cloud_logger.read_cloud_logging_entries()
        )

      # Update the cache with the new log entries.
      self._goodput_cache.update_cached_entries(
          new_entries, current_last_entry_info
      )
      return new_entries

  def _get_interval_log_entries(
      self, start_time: datetime.datetime, end_time: datetime.datetime
  ):
    """Helper function to get log entries from an interval window incrementally."""
    if start_time is None or end_time is None:
      raise ValueError(
          'Start and end times are required to get log entries from an interval'
          ' window.'
      )

    # Detect if this is a sequential forward sliding window query
    is_sequential = (
        self._last_interval_end_time is not None
        and self._last_interval_start_time is not None
        and end_time > self._last_interval_end_time
        and start_time >= self._last_interval_start_time
    )

    if not is_sequential:
      # Reset cache for non-sequential (arbitrary/jump) queries
      self._interval_cache = []
      self._last_interval_entry_info = None

    if not self._interval_cache or self._last_interval_entry_info is None:
      # Full query for cold start or non-sequential query
      entries, last_info = self._cloud_logger.read_cloud_logging_entries(
          start_time=start_time,
          end_time=end_time,
      )
      self._interval_cache = entries
      self._last_interval_entry_info = last_info
    else:
      # Incremental query: fetch only delta since last timestamp
      last_ts, _ = self._last_interval_entry_info
      if last_ts and end_time > last_ts:
        new_entries, last_info = self._cloud_logger.read_cloud_logging_entries(
            start_time=last_ts,
            end_time=end_time,
            last_entry_info=self._last_interval_entry_info,
        )
        self._interval_cache.extend(new_entries)
        if last_info and last_info[0] is not None:
          self._last_interval_entry_info = last_info

    # Slide the window in-memory: discard older entries
    self._interval_cache = [
        entry
        for entry in self._interval_cache
        if (entry_ts := get_timestamp_from_log_entry(entry)) is not None
        and entry_ts > start_time
    ]

    self._interval_entries = self._interval_cache
    self._last_interval_start_time = start_time
    self._last_interval_end_time = end_time

    if not self._interval_entries:
      raise ValueError(
          'No log entries found within the interval window between %s and %s.'
          % (start_time, end_time)
      )

  def _sanitize_unproductive_times(
      self,
      unproductive_times: UnproductiveTimeDict,
      max_allowed: float,
  ) -> None:
    """Helper function to sanitize unproductive times."""
    for badput_type, value in unproductive_times.items():
      if isinstance(value, float):
        if value < 0.0 or value > max_allowed:
          logging.warning(
              'Unproductive time for %s could not be computed.', badput_type
          )
          unproductive_times[badput_type] = 0.0
      elif isinstance(value, dict):
        for sub_type, sub_value in value.items():
          if sub_value < 0.0 or sub_value > max_allowed:
            logging.warning(
                'Unproductive time for %s[%s] could not be computed.',
                badput_type,
                sub_type,
            )
            value[sub_type] = 0.0

  def _calculate_total_flat_unproductive_time(
      self,
      unproductive_time_dict: UnproductiveTimeDict,
  ) -> float:
    """Helper function to calculate total flat unproductive time."""
    total = 0.0
    for badput_type, value in unproductive_time_dict.items():
      if badput_type in {BadputType.DATA_LOADING_ASYNC, BadputType.OTHER}:
        continue
      if isinstance(value, float):
        total += value
      elif isinstance(value, dict):
        total += sum(value.values())
    return total

  def _compute_other_unproductive_time(
      self,
      total_job_time: float,
      productive_training_time: float,
      unproductive_time_dict: UnproductiveTimeDict,
  ) -> float:
    """Helper function to compute the "Unknown/Other" unproductive time."""
    other_unproductive_time = (
        total_job_time
        - productive_training_time
        - self._calculate_total_flat_unproductive_time(unproductive_time_dict)
    )
    return max(0.0, other_unproductive_time)

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
      self,
      include_badput_breakdown=False,
      configured_ideal_step_time: Optional[float] = None,
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
  ]:
    """Method to get the cumulative Goodput and Badput breakdown of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job.

    This method also returns the Badput breakdown of the job if
    `include_badput_breakdown` is set to True.

    Additionally, this method returns the last step recorded for the job. This is
    primarily used for improving monitoring and observability of the job's
    overall Goodput as a function of number of executed steps.

    Args:
      include_badput_breakdown: Whether or not to return the badput breakdown.
        If False, returns {} for the badput breakdown.
      configured_ideal_step_time: The configured ideal step time for the job. If
        not set, the an ideal step time will be computed under the hood.

    Returns:
      A tuple of the job's Goodput, optionally the Badput breakdown and the last
      productive step recorded for the job.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training time is invalid.
    """
    query_time = datetime.datetime.now(datetime.timezone.utc)

    # Update the logs used to compute Goodput.
    new_entries = self._fetch_new_entries(query_time)

    total_job_time = self._get_total_job_time(query_time)
    # No calculations can be made if total job time is zero. This can happen if
    # logs for the job are not present, sent to an invalid location or contain
    # bad data. Raise a ValueError if this happens.
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    (
        productive_training_time,
        total_unproductive_time,
        max_productive_step,
        last_recorded_step,
    ) = self._get_total_productive_and_unproductive_time(new_entries)
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )

    # Sanitize the unproductive times.
    self._sanitize_unproductive_times(total_unproductive_time, total_job_time)

    # Compute the "Unknown/Other" unproductive time.
    total_unproductive_time[BadputType.OTHER] = (
        self._compute_other_unproductive_time(
            total_job_time, productive_training_time, total_unproductive_time
        )
    )

    # Compute the job Goodput and Badput breakdown.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    job_badput_breakdown = (
        self._get_job_badput_breakdown(total_unproductive_time, total_job_time)
        if include_badput_breakdown
        else {}
    )

    # Update the Goodput cache with new information.
    with self._goodput_cache_lock:
      self._goodput_cache.update_goodput_info(
          GoodputInfo(
              total_productive_time=productive_training_time,
              total_elapsed_time=total_job_time,
              total_unproductive_time=total_unproductive_time,
              max_productive_step=max_productive_step,
              last_recorded_step=last_recorded_step,
              last_updated_timestamp=datetime.datetime.now(
                  datetime.timezone.utc
              ),
              number_of_disruptions=self._number_of_interruptions,
          )
      )

    # Compute and store step information.
    try:
      self._compute_step_info_and_update_cache(
          new_entries, configured_ideal_step_time
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.info('Failed to compute step information: %s', e)

    return job_goodput, job_badput_breakdown, max_productive_step

  def get_job_goodput_interval(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
      float,
      int,
  ]:
    """Method to get the Goodput and Badput breakdown of the job within an interval window.

    If the application is interested in retrieving the Goodput of the job within
    a specific window of time, this method provides the metrics computed between
    the start and end of this window.

    Additionally, this method returns the last step recorded for the job. This is
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
        - The last productive step for the job within the interval window.
        - The total job time within the interval window.
        - The number of disruptions within the interval window.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
      ValueError if productive training or unproductive time is invalid.
    """
    # Get the logs for the interval and validate the interval window.
    self._get_interval_log_entries(interval_start, interval_end)
    return self._compute_job_goodput_interval_metrics(
        interval_start, interval_end
    )

  def _compute_job_goodput_interval_metrics(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> tuple[
      float,
      UnproductiveTimeDict,
      int,
      float,
      int,
  ]:
    """Computes Goodput and Badput metrics from the currently loaded self._interval_entries.

    Callers are responsible for populating self._interval_entries (e.g. via
    _get_interval_log_entries) with entries bounded to
    [interval_start, interval_end] before calling this method.
    """
    total_job_time = self._get_total_job_time_from_interval(
        interval_start, interval_end
    )

    (
        productive_training_time,
        total_unproductive_time,
        max_productive_step,
        _,
    ) = self._get_current_productive_and_unproductive_time(interval_query=True)
    if (
        productive_training_time < 0.0
        or productive_training_time > total_job_time
    ):
      raise ValueError(
          'Productive training time is invalid. Please fix the logging entries.'
      )

    # Sanitize unproductive times
    self._sanitize_unproductive_times(total_unproductive_time, total_job_time)

    # Compute the "Unknown/Other" unproductive time
    total_unproductive_time[BadputType.OTHER] = (
        self._compute_other_unproductive_time(
            total_job_time, productive_training_time, total_unproductive_time
        )
    )

    # Compute the job Goodput and Badput breakdown.
    job_goodput = (float(productive_training_time) / total_job_time) * 100
    job_badput_breakdown = self._get_job_badput_breakdown(
        total_unproductive_time, total_job_time
    )

    return (
        job_goodput,
        job_badput_breakdown,
        max_productive_step,
        total_job_time,
        self._number_of_interruptions,
    )

  def _get_step_times(self, entries: list[Any]) -> dict[int, float]:
    """Computes per-step execution durations from log entries.

    Excludes synchronous checkpoint save blocking durations and breaks step
    pairings across job or elastic restarts so recovery downtime does not
    pollute `ideal_step_time` or `step_time_deviation`.

    Args:
      entries: List of raw or `(timestamp, payload)` cached log entries.

    Returns:
      Dictionary mapping step count (`int`) to execution duration (`float`).
    """
    normalized_entries = []
    for item in entries:
      if (
          isinstance(item, tuple)
          and len(item) == 2
          and isinstance(item[1], dict)
      ):
        normalized_entries.append(item[1])
      else:
        normalized_entries.append(item)

    save_events = self._extract_checkpoint_save_events(normalized_entries)
    step_times = {}
    previous_step_start_time = None
    previous_step_count = None

    for payload in normalized_entries:
      if not isinstance(payload, dict):
        continue

      # When a restart occurs after `previous_step_start_time`, do not pair
      # `previous_step_count` directly with the post-restart step timestamp.
      restart_event = self._get_restart_event(payload)
      if restart_event is not None and previous_step_start_time is not None:
        restart_ts, _ = restart_event
        if restart_ts > previous_step_start_time:
          if (
              previous_step_count is not None
              and previous_step_count in self._historical_step_times
          ):
            step_times[previous_step_count] = self._historical_step_times[
                previous_step_count
            ]
          previous_step_start_time = None
          previous_step_count = None

      if _STEP_START_TIME in payload:
        step_start_time = payload[_STEP_START_TIME]
        step_count = int(payload[_STEP_COUNT])
        if (
            previous_step_start_time is not None
            and previous_step_count is not None
            and step_count == previous_step_count + 1
        ):
          step_delta = step_start_time - previous_step_start_time
          checkpoint_save_in_step = (
              self._get_checkpoint_save_duration_in_window(
                  save_events,
                  previous_step_count,
                  previous_step_start_time,
                  step_start_time,
              )
          )
          if (
              checkpoint_save_in_step > 0.0
              and step_delta >= checkpoint_save_in_step
          ):
            step_delta -= checkpoint_save_in_step
          if previous_step_count in self._historical_step_times:
            step_delta = min(
                step_delta, self._historical_step_times[previous_step_count]
            )
          step_times[previous_step_count] = step_delta
        previous_step_count = step_count
        previous_step_start_time = step_start_time
    return step_times

  def _contains_step_entries(self, entries: list[Any]) -> bool:
    return any(_STEP_START_TIME in entry for entry in entries)

  def _compute_step_info_and_update_cache(
      self,
      new_entries: list[Any],
      configured_ideal_step_time: Optional[float] = None,
  ) -> dict[int, float]:
    """Method to compute the step time deviation and update the cache."""
    with self._goodput_cache_lock:
      step_info = self._goodput_cache.get_step_info()

    if (
        not self._contains_step_entries(new_entries)
        and step_info
        and step_info.step_deviations
    ):
      return step_info.step_deviations

    with self._goodput_cache_lock:
      process_entries = self._goodput_cache.get_cached_entries()

    step_times = self._get_step_times(process_entries)

    if not step_times:
      raise ValueError(
          'No step times available and no previous step deviations found.'
      )

    # Compute ideal step time.
    ideal_step_time = (
        configured_ideal_step_time
        if configured_ideal_step_time is not None
        else compute_ideal_step_time(list(step_times.values()))
    )
    if not ideal_step_time:
      raise ValueError(
          'No ideal step time available and no previous step deviations found.'
      )

    # Compute step deviation.
    step_deviations = {
        step_count: abs(step_time - ideal_step_time)
        for step_count, step_time in step_times.items()
    }
    # Update the step information in the cache.
    with self._goodput_cache_lock:
      self._goodput_cache.update_step_info(
          StepInfo(
              ideal_step_time=ideal_step_time,
              step_deviations=step_deviations,
          )
      )
    return step_deviations

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
    query_time = datetime.datetime.now(datetime.timezone.utc)
    new_entries = self._fetch_new_entries(query_time)
    return self._compute_step_info_and_update_cache(
        new_entries, configured_ideal_step_time
    )

  def _get_job_badput_breakdown(
      self, total_unproductive_time, total_job_time
  ) -> UnproductiveTimeDict:
    """Method to get the the Badput breakdown as percentage of total job time.

    This method provides a granular breakdown of the known components of Badput.

    Args:
      total_unproductive_time: A dictionary of computed unproductive time of
        each BadputType.
      total_job_time: The total job time.

    Returns:
      A dictionary of badput components and their percentage breakdown within
      total job time.
    """
    badput_breakdown: dict[
        BadputType, float | dict[str, float]
    ] = {}
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

    # Only synchronous data loading is badput.
    # Sync data loading is accumulated after start and reset of the job and is
    # blocking.
    sync_data_loading_badput = total_unproductive_time.get(
        BadputType.DATA_LOADING_SYNC, 0.0
    )
    # Async data loading is accumulated overlapping with training and is
    # non-blocking, therefore is not unproductive time.
    async_data_loading_badput = total_unproductive_time.get(
        BadputType.DATA_LOADING_ASYNC, 0.0
    )
    badput_breakdown[BadputType.DATA_LOADING_SYNC] = (
        (sync_data_loading_badput / total_job_time) * 100
        if 0 < sync_data_loading_badput < total_job_time
        else 0.0
    )
    badput_breakdown[BadputType.DATA_LOADING_ASYNC] = (
        (async_data_loading_badput / total_job_time) * 100
        if 0 < async_data_loading_badput < total_job_time
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

    # Infrastructure recovery from disruption badput.
    infrastructure_recovery_from_disruption_badput = (
        total_unproductive_time.get(
            BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION, 0.0
        )
    )
    badput_breakdown[BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION] = (
        (infrastructure_recovery_from_disruption_badput / total_job_time) * 100
        if 0 < infrastructure_recovery_from_disruption_badput < total_job_time
        else 0.0
    )

    # Custom events badput.
    badput_breakdown[BadputType.CUSTOM_BADPUT_EVENTS] = {}
    custom_events_badput = total_unproductive_time.get(
        BadputType.CUSTOM_BADPUT_EVENTS, {}
    )

    if isinstance(custom_events_badput, dict):
      nested_breakdown = {}
      for (
          custom_badput_type,
          custom_events_badput_value,
      ) in custom_events_badput.items():
        nested_breakdown[custom_badput_type] = (
            (custom_events_badput_value / total_job_time) * 100
            if 0 < custom_events_badput_value < total_job_time
            else 0.0
        )
      badput_breakdown[BadputType.CUSTOM_BADPUT_EVENTS] = (
          nested_breakdown
      )

    # Populate the 'Other/Unknown' badput bucket.
    other_badput = total_unproductive_time.get(BadputType.OTHER, 0.0)
    badput_breakdown[BadputType.OTHER] = (
        (other_badput / total_job_time) * 100
        if 0 < other_badput < total_job_time
        else 0.0
    )

    return badput_breakdown

  def get_job_goodput_details(self) -> WorkloadMetricDetails:
    """Method to get workload metrics details."""
    with self._goodput_cache_lock:
      goodput_info = self._goodput_cache.get_goodput_info()
      if goodput_info is None:
        logger.warning(
            'Goodput information unavailable and will not be uploaded to GCM'
        )
        return {
            MetricType.GOODPUT_TIME.value: {},
            MetricType.BADPUT_TIME.value: {},
            MetricType.MAX_PRODUCTIVE_STEP.value: 0,
            MetricType.TOTAL_ELAPSED_TIME.value: 0.0,
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.STEP_TIME_DEVIATION.value: {},
            MetricType.IDEAL_STEP_TIME.value: 0.0,
            MetricType.TOTAL_EXCLUDED_TIME.value: 0.0,
        }

      (
          productive_training_time,
          total_unproductive_time,
          cache_last_updated_timestamp,
          max_productive_step,
          total_elapsed_time,
          number_of_disruptions,
      ) = (
          goodput_info.total_productive_time,
          goodput_info.total_unproductive_time,
          goodput_info.last_updated_timestamp,
          goodput_info.max_productive_step,
          goodput_info.total_elapsed_time,
          goodput_info.number_of_disruptions,
      )

      if (
          self._gcm_last_recorded_timestamp
          is not None  # Ignore the first entry.
          and self._gcm_last_recorded_timestamp >= cache_last_updated_timestamp
      ):
        logger.warning(
            'No new data, metrics may be stale until the next poll. Cache'
            ' Timestamp: %s, GCM Timestamp: %s',
            cache_last_updated_timestamp,
            self._gcm_last_recorded_timestamp,
        )

      self._gcm_last_recorded_timestamp = datetime.datetime.now(
          datetime.timezone.utc
      )
      # Fetch the step deviation from the cache.
      step_info = self._goodput_cache.get_step_info()
      step_time_deviation = (
          step_info.step_deviations
          if step_info and step_info.step_deviations
          else {}
      )
      ideal_step_time = (
          step_info.ideal_step_time
          if step_info and step_info.ideal_step_time
          else 0.0
      )

      total_productive_time = {GoodputType.TOTAL: productive_training_time}

      return {
          MetricType.GOODPUT_TIME.value: total_productive_time,
          MetricType.BADPUT_TIME.value: total_unproductive_time,
          MetricType.MAX_PRODUCTIVE_STEP.value: max_productive_step,
          MetricType.TOTAL_ELAPSED_TIME.value: total_elapsed_time,
          MetricType.DISRUPTION_COUNT.value: number_of_disruptions,
          MetricType.STEP_TIME_DEVIATION.value: step_time_deviation,
          MetricType.IDEAL_STEP_TIME.value: ideal_step_time,
          MetricType.TOTAL_EXCLUDED_TIME.value: 0.0,
      }

  def get_interval_metric_details(
      self, interval_start: datetime.datetime, interval_end: datetime.datetime
  ) -> IntervalWorkloadMetricDetails:
    """Method to get interval metric details."""
    try:
      (
          interval_goodput,
          interval_badput_breakdown,
          _,
          _,
          _,
      ) = self.get_job_goodput_interval(interval_start, interval_end)
      return {
          IntervalMetricType.INTERVAL_GOODPUT.value: {
              GoodputType.TOTAL: interval_goodput
          },
          IntervalMetricType.INTERVAL_BADPUT.value: interval_badput_breakdown,
          IntervalMetricType.INTERVAL_SIZE.value: (int)(
              (interval_end - interval_start).total_seconds()
          ),
      }

    except ValueError as e:
      logger.warning('Failed to get interval metric details: %s', e)
      return {
          IntervalMetricType.INTERVAL_GOODPUT.value: {},
          IntervalMetricType.INTERVAL_BADPUT.value: {},
          IntervalMetricType.INTERVAL_SIZE.value: (int)(
              (interval_end - interval_start).total_seconds()
          ),
      }
