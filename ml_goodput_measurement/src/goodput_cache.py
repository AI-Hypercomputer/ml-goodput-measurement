"""Goodput Cache implementations."""

import datetime
from typing import Any

from ml_goodput_measurement.src import goodput_utils


StepInfo = goodput_utils.StepInfo
GoodputInfo = goodput_utils.GoodputInfo
_TIME_ENTRY = 'time'
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'


class GoodputCache:
  """Goodput Cache."""

  def __init__(self):
    self._cached_entries = []
    self._goodput_info = None
    self._last_entry_timestamp = None
    self._job_start_time = None
    self._job_end_time = None
    self._step_info = None

  def update_step_info(self, step_info: StepInfo):
    """Updates the step information."""
    self._step_info = step_info

  def update_cached_entries(self, entries: list[Any]):
    """Updated the cached entries."""
    self._cached_entries.extend(entries)
    self.update_last_entry_timestamp()
    self.update_job_start_time()
    self.update_job_end_time()

  def update_last_entry_timestamp(self):
    """Helper function to store the timestamp of the last entry in the cache."""
    if self._cached_entries:
      last_entry = self._cached_entries[-1]
      last_entry_posix_time = [
          entry_value
          for entry_label, entry_value in last_entry.items()
          if _TIME_ENTRY in entry_label
      ]
      if last_entry_posix_time:
        self._last_entry_timestamp = datetime.datetime.fromtimestamp(
            last_entry_posix_time[0]
        )

  def update_job_start_time(self):
    """Updates the job start time."""
    # If the job start time is not set, try to find it in the cached entries.
    if self._job_start_time is None and self._cached_entries:
      for entry in self._cached_entries:
        if _JOB_START_TIME in entry:
          self._job_start_time = datetime.datetime.fromtimestamp(
              entry[_JOB_START_TIME]
          )
          break

  def update_job_end_time(self):
    """Updates the job end time."""
    # Overwrite the latest job end time if cached entries contain the job end
    # time.
    if self._cached_entries:
      for entry in self._cached_entries:
        if _JOB_END_TIME in entry:
          self._job_end_time = datetime.datetime.fromtimestamp(
              entry[_JOB_END_TIME]
          )

  def update_goodput_info(self, goodput_info: GoodputInfo):
    """Updates the last computed Goodput information."""
    self._goodput_info = goodput_info

  def get_goodput_info(self):
    """Returns the last computed Goodput information."""
    return self._goodput_info

  def clear_cache(self):
    """Clears the cache."""
    self._cached_entries = []
    self._goodput_info = None
    self._last_entry_timestamp = None

  def is_cache_empty(self) -> bool:
    """Checks if the cache is empty."""
    return not self._cached_entries
