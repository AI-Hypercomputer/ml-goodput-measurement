"""Goodput Utility Classes and Helpers."""

import datetime
import enum
from typing import Any, Optional

_TIME_ENTRY = 'time'


class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING = 4
  UNPRODUCTIVE_CHECKPOINT_SAVE_TIME = 5
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME = 6
  WASTED_PROGRESS_FROM_DISRUPTION = 7
  OTHER = 8


class GoodputInfo:
  """Goodput Information."""

  def __init__(
      self,
      total_productive_time: float = 0.0,
      total_elapsed_time_since_start: float = 0.0,
      total_unproductive_time: Optional[dict[BadputType, float]] = {},
      last_recorded_step: int = 0,
  ):
    self.total_productive_time = total_productive_time
    self.total_elapsed_time_since_start = total_elapsed_time_since_start
    self.total_unproductive_time = total_unproductive_time
    self.last_recorded_step = last_recorded_step


def get_timestamp_from_log_entry(
    entry: dict[str, Any],
) -> datetime.datetime | None:
  """Helper function to get the timestamp from a log entry."""
  timestamp_posix_time = [
      entry_value
      for entry_label, entry_value in entry.items()
      if _TIME_ENTRY in entry_label
  ]
  if timestamp_posix_time:
    return datetime.datetime.fromtimestamp(
        timestamp_posix_time[0], datetime.timezone.utc
    )
  return None
