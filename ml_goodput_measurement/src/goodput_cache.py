"""Goodput Cache implementations."""

import datetime
from typing import Any, Dict, Optional

from cloud_goodput.ml_goodput_measurement.src.goodput_utils import BadputType, GoodputInfo

_TIME_ENTRY = 'time'


class GoodputCache:
  """Goodput Cache."""

  def __init__(self):
    self._cached_entries = []
    self._goodput_info = None
    self._last_entry_timestamp = None

  def update_cached_entries(self, entries: list[Any]):
    """Updated the cached entries."""
    self._cached_entries = entries
    self.update_last_entry_timestamp()

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

  def update_goodput_info(self, goodput_info: GoodputInfo):
    """Updates the last computed Goodput information."""
    self._goodput_info = goodput_info

  def clear_cache(self):
    """Clears the cache."""
    self._cached_entries = []
    self._goodput_info = None
    self._last_entry_timestamp = None

  def is_cache_empty(self) -> bool:
    """Checks if the cache is empty."""
    return not self._cached_entries

