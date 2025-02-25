"""Goodput Utility Classes and Helpers."""

import datetime
import enum
import subprocess
from typing import Any, Optional

import numpy as np
from scipy import stats


_TIME_ENTRY = 'time'

# Productive time is not broken down by activities yet. As such, we only have
# one type of Goodput which contributes to the total productive time.
class GoodputType(enum.Enum):
  """The type of Goodput."""

  TOTAL = 1


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


class StepInfo:
  """Step Information."""

  def __init__(
      self,
      ideal_step_time: float,
      step_deviations: dict[int, float],
  ):
    self.ideal_step_time = ideal_step_time
    self.step_deviations = step_deviations


def compute_ideal_step_time(
    step_times: list[float], previous_ideal_step_time: Optional[float]
) -> float:
  """Helper function to compute the ideal step time."""
  # Filter out the normal step times from the step times dictionary.
  mad = stats.median_abs_deviation(step_times)
  med = np.median(step_times)
  normal_step_times = []
  for step_time in step_times:
    if step_time <= (med + mad * 3):
      normal_step_times.append(step_time)
  mean_normal_step_time = np.mean(normal_step_times)
  if previous_ideal_step_time is not None:
    return np.mean([mean_normal_step_time, previous_ideal_step_time])
  return mean_normal_step_time


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


def get_extra_time_from_anomalous_steps(step_times: list[Any]) -> float:
  anomalous_step_times, normal_step_times = get_anomalous_and_normal_step_times(
      step_times
  )
  normal_step_mean = np.mean(normal_step_times)
  return sum(anomalous_step_times) - (
      len(anomalous_step_times) * normal_step_mean
  )


def get_timestamp_from_log_entry(
    entry: dict[str, Any],
) -> Optional[datetime.datetime]:
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


def check_gcloud_auth(project_id: str):
  """Checks if gcloud is authenticated and set to the correct project."""
  try:
    # Check if gcloud is authenticated
    process = subprocess.run(
        ['gcloud', 'auth', 'list'],
        capture_output=True,
        text=True,
        check=True
    )
    if 'ACTIVE' not in process.stdout:
      raise ValueError(
          'gcloud is not authenticated. Please run `gcloud auth login`.'
      )

    # Check if the project is set correctly
    process = subprocess.run(
        ['gcloud', 'config', 'get-value', 'project'],
        capture_output=True,
        text=True,
        check=True,
    )
    if process.stdout.strip() != project_id:
      raise ValueError(
          'gcloud is not set to the correct project. Please run `gcloud config'
          f' set project {project_id}`.'
      )

  except Exception as e:
    raise ValueError('Error checking gcloud authentication.') from e
