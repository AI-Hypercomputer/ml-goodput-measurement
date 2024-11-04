"""Goodput Utility Classes and Helpers."""

import datetime
import enum
from typing import Optional


class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING = 4
  UNPRODUCTIVE_CHECKPOINTING = 5
  WASTED_PROGRESS_FROM_DISRUPTION = 6
  OTHER = 7


class GoodputInfo:
  """Goodput Information."""

  def __init__(
      self,
      total_productive_time: float = 0.0,
      total_elapsed_time_since_start: float = 0.0,
      total_unproductive_time: Optional[dict[BadputType, float]] = None,
      last_recorded_step: int = 0,
  ):
    self.total_productive_time = total_productive_time
    self.total_elapsed_time_since_start = total_elapsed_time_since_start
    self.total_unproductive_time = total_unproductive_time
    self.last_recorded_step = last_recorded_step
