"""Checkpoint Badput Calculator class."""

import dataclasses
from typing import Dict, List, Optional

import google.cloud.logging as google_cloud_logging


_JOB_NAME = 'checkpoint_job'
_LOGGER_NAME = 'checkpoint_logger'

_STEP = 'step'
_BLOCKING_SAVE_START_TIME = 'checkpoint_start_time'
_BLOCKING_SAVE_END_TIME = 'checkpoint_end_time'
_EVENT_TYPE = 'event_type'
_SYNCHRONOUS = 'synchronous'
_PREEMPTION = 'reached_preemption'
_PREEMPTION_RECEIVED_AT = 'preemption_received_at'
_WAIT_FOR_PREV_START_TIME = 'wait_for_prev_start_time'
_WAIT_FOR_PREV_END_TIME = 'wait_for_prev_end_time'


# Total checkpoint save time.
_TOTAL_SAVE_START_TIME = 'start_time'
_TOTAL_SAVE_END_TIME = 'end_time'


_CLOUD_LOGGING_PAGE_SIZE = 10000


@dataclasses.dataclass
class CheckpointLoggerOptions:
  """Checkpoint logger options."""
  job_name: str = _JOB_NAME
  logger_name: str = _LOGGER_NAME
  client: Optional[google_cloud_logging.Client] = None


class CheckpointBadputCalculator:
  """Checkpoint Badput Calculator class."""

  def __init__(
      self, options: CheckpointLoggerOptions = CheckpointLoggerOptions()
  ):
    self._options = options

    if options.client is None:
      self.logging_client = google_cloud_logging.Client()
    else:
      self.logging_client = options.client
    self._logger = self.logging_client.logger(options.logger_name)
    self.entries = []

  def read_entries(self) -> List[Dict[str, str]]:
    """Queries Cloud Logging entries for the specific job.

    Returns:
      Filtered entries in ascending order of timestamp.

    """

    filter_entries = [
        'severity=INFO',
        f'jsonPayload.job_name="{self._options.job_name}"',
        'jsonPayload.event_type=save',
    ]
    filter_entries = ' AND '.join(filter_entries)
    entries = self._logger.list_entries(
        filter_=filter_entries,
        order_by=google_cloud_logging.ASCENDING,
        page_size=_CLOUD_LOGGING_PAGE_SIZE,
    )
    entry_payload = [entry.payload for entry in entries]
    return entry_payload

  def calculate_blocking_checkpoint_time(self) -> float:
    """Gets blocking checkpoint time duration for total checkpoint."""
    if not self.entries:
      self.entries = self.read_entries()

    total_blocking_checkpoint_time = 0.0
    for step in self.entries:
      if step[_PREEMPTION]:
        total_blocking_checkpoint_time += float(
            step[_TOTAL_SAVE_END_TIME]
        ) - float(step[_WAIT_FOR_PREV_END_TIME])
      else:
        total_blocking_checkpoint_time += float(
            step[_BLOCKING_SAVE_END_TIME]
        ) - float(step[_BLOCKING_SAVE_START_TIME])
    return total_blocking_checkpoint_time
