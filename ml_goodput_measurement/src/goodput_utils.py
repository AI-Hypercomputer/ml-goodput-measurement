"""Goodput Utility Classes and Helpers."""

import dataclasses
import datetime
import enum
import logging
from typing import Any, Optional, TypedDict, Union

import numpy as np
import requests
from scipy import stats
from urllib3.util import retry


Retry = retry.Retry
_TIME_ENTRY = 'time'
_METADATA_SERVER_URL = 'http://metadata.google.internal/computeMetadata/v1/'
_METADATA_HEADERS = {'Metadata-Flavor': 'Google'}

MACHINE_TYPE_TO_ACCELERATOR_TYPE_MAPPING = {
    'ct6e': 'TPU-v6e',
    'ct5p': 'TPU-v5p',
    'ct5lp': 'TPU-v5e',
    'ct5l': 'TPU-v5e',
    'ct4p': 'TPU-v4p',
    'ct3p': 'TPU-v3',
    'ct3': 'TPU-v3',
    'tpu-v2': 'TPU-v2',
    'tpu': 'TPU',
    'a3-edgegpu': 'NVIDIA-H100',
    'a3-highgpu': 'NVIDIA-H100',
    'a3-megagpu': 'NVIDIA-H100',
    'a3-ultragpu': 'NVIDIA-H200',
    'a2': 'NVIDIA-A100',
    'gpu': 'GPU',
}


@dataclasses.dataclass
class GCPOptions:
  project_id: Optional[str] = None
  location: Optional[str] = None
  replica_id: str = '0'
  acc_type: Optional[str] = None
  enable_gcp_goodput_metrics: bool = True
  enable_gcp_step_deviation_metrics: bool = True


# Cumulative metric types for upload and monitoring.
class MetricType(enum.Enum):
  """The type of CUMULATIVE Metric."""
  GOODPUT_TIME = 'goodput_time'
  BADPUT_TIME = 'badput_time'
  MAX_PRODUCTIVE_STEP = 'max_productive_step'
  TOTAL_ELAPSED_TIME = 'total_elapsed_time'
  DISRUPTION_COUNT = 'disruption_count'
  STEP_TIME_DEVIATION = 'step_time_deviation'
  IDEAL_STEP_TIME = 'ideal_step_time'
  TOTAL_EXCLUDED_TIME = 'total_excluded_time'


# Interval metric types for upload and monitoring.
class IntervalMetricType(enum.Enum):
  """The type of INTERVAL Metric."""

  INTERVAL_GOODPUT = 'interval_goodput'
  INTERVAL_BADPUT = 'interval_badput'
  INTERVAL_SIZE = 'interval_size'


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
  DATA_LOADING_SYNC = 4
  DATA_LOADING_ASYNC = 5
  UNPRODUCTIVE_CHECKPOINT_SAVE_TIME = 6
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME = 7
  WASTED_PROGRESS_FROM_DISRUPTION = 8
  INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION = 9
  CUSTOM_BADPUT_EVENTS = 10
  OTHER = 11


class WorkloadMetricDetails(TypedDict):
  goodput_time: dict[GoodputType, float]
  badput_time: dict[BadputType, float | dict[str, float]]
  max_productive_step: int
  total_elapsed_time: float
  disruption_count: int
  step_time_deviation: dict[int, float]
  ideal_step_time: float
  total_excluded_time: float


class IntervalWorkloadMetricDetails(TypedDict):
  interval_goodput: dict[GoodputType, float]
  interval_badput: dict[BadputType, float | dict[str, float]]
  interval_size: int  # Unit: seconds.


ACTIVITY_EXCLUSION_LIST = [
    # DATA_LOADING_ASYNC is not a non-productive activity as it is not
    # blocking. Hence, we exclude it from calculating Goodput.
    'DATA_LOADING_ASYNC',
]


class MonitoringWindowType(enum.Enum):
  """The type of Monitoring Window."""

  CUMULATIVE = 'cumulative'
  INTERVAL = 'interval'


_DEFAULT_RECENT_WINDOW_SIZE = 100
_DEFAULT_BASELINE_WINDOW_SIZE = 1000
_DEFAULT_SPIKE_PERCENTILE = 90


class GoodputInfo:
  """Goodput Information."""

  def __init__(
      self,
      total_productive_time: float = 0.0,
      total_elapsed_time: float = 0.0,
      total_unproductive_time: Optional[
          dict[BadputType, Union[float, dict[str, float]]]
      ] = None,
      max_productive_step: int = 0,
      last_recorded_step: int = 0,
      last_updated_timestamp: datetime.datetime = datetime.datetime.now(
          datetime.timezone.utc
      ),
      number_of_disruptions: int = 0,
      total_excluded_time: float = 0.0,
  ):
    self.total_productive_time = total_productive_time
    self.total_elapsed_time = total_elapsed_time

    # We cannot use {} as the default argument directly because it's a mutable
    # default argument.  Mutable default arguments are shared between all
    # instances of the class.  If one instance modifies the default
    # dictionary, it will affect all other instances.  Instead, we use
    # None as a sentinel value and create a new dictionary inside the
    # __init__ method if no dictionary is provided. This ensures each
    # instance gets its own dictionary.
    self.total_unproductive_time = (
        total_unproductive_time or {}
    )
    self.max_productive_step = max_productive_step
    self.last_recorded_step = last_recorded_step
    self.last_updated_timestamp = last_updated_timestamp
    self.number_of_disruptions = number_of_disruptions
    self.total_excluded_time = total_excluded_time


class StepInfo:
  """Step Information."""

  def __init__(
      self,
      ideal_step_time: float,
      step_deviations: dict[int, float],
  ):
    self.ideal_step_time = ideal_step_time
    self.step_deviations = step_deviations


def compute_percentile(values: list[float], percentile: float) -> float:
  """Computes the specified percentile value from a list of floats."""
  if not values:
    return 0.0

  sorted_values = sorted(values)
  index = (len(sorted_values) - 1) * (percentile / 100.0)
  lower_index = int(index)
  upper_index = min(lower_index + 1, len(sorted_values) - 1)

  return sorted_values[lower_index] + (
      sorted_values[upper_index] - sorted_values[lower_index]
  ) * (index - lower_index)


def compute_step_deviation_from_baseline(
    step_time_deviation: dict[int, float],
    mode: MonitoringWindowType = MonitoringWindowType.CUMULATIVE,
    recent_window_size: int = _DEFAULT_RECENT_WINDOW_SIZE,
    baseline_window_size: int = _DEFAULT_BASELINE_WINDOW_SIZE,
    spike_percentile: int = _DEFAULT_SPIKE_PERCENTILE,
) -> float:
  """Computes a spike-sensitive step time deviation metric.

  Args:
    step_time_deviation: Ordered dict (step count -> step deviation in seconds).
    mode: 'cumulative' to compare against a historical baseline; 'interval' to
      reflect short-term spikes only.
    recent_window_size: Number of recent steps to consider for interval mode.
    baseline_window_size: Number of older steps for cumulative baseline.
    spike_percentile: Percentile to use for recent deviation sensitivity.

  Returns:
    The step deviation from the baseline.
  """
  if not step_time_deviation:
    return 0.0

  deviations = [abs(deviation) for deviation in step_time_deviation.values()]
  total_steps = len(deviations)

  if total_steps < _DEFAULT_RECENT_WINDOW_SIZE:
    return np.mean(deviations)

  if mode == MonitoringWindowType.INTERVAL:
    recent_deviations = deviations[-recent_window_size:]
    return compute_percentile(recent_deviations, spike_percentile)

  elif mode == MonitoringWindowType.CUMULATIVE:
    if total_steps < (recent_window_size + baseline_window_size):
      recent_deviations = deviations[-recent_window_size:]
      return compute_percentile(recent_deviations, spike_percentile)

    recent_deviations = deviations[-recent_window_size:]
    baseline_deviations = deviations[
        -(recent_window_size + baseline_window_size) : -recent_window_size
    ]

    if not baseline_deviations:
      return compute_percentile(recent_deviations, spike_percentile)

    baseline_median = np.median(baseline_deviations)
    spike_value = compute_percentile(recent_deviations, spike_percentile)
    return spike_value - baseline_median

  else:
    raise ValueError('Unsupported MonitoringWindowType mode: {mode}')


def compute_ideal_step_time(step_times: list[float]) -> Optional[float]:
  """Helper function to compute the ideal step time."""
  # Filter out step times that may be less than 1 second.
  step_times = [step_time for step_time in step_times if step_time >= 1.0]
  if not step_times:
    return None
  # Compute the median absolute deviation (MAD) and median of the step times
  mad = stats.median_abs_deviation(step_times)
  med = np.median(step_times)

  # Normalize the step times to the median + 3 * MAD.
  normal_step_times = [
      step_time for step_time in step_times if step_time <= (med + mad * 3)
  ]
  return np.mean(normal_step_times) if normal_step_times else None


def get_anomalous_and_normal_step_times(
    step_times: list[Any],
) -> tuple[list[Any], list[Any]]:
  """Helper function to get anomalous and normal step times."""
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


def get_gcp_metadata(category: str, attribute: str, timeout=5, retries=3):
  """Fetch the specified attribute from GCP metadata server.

  Args:
    category (str): The high-level metadata category (ex: 'instance',
      'project').
    attribute (str): The attribute to fetch under this category (ex: 'id',
      'zone').
    timeout (int): Timeout for the request in seconds.
    retries (int): Number of retry attempts for transient failures.

  Returns:
    str: The metadata value as a string, or None if the request fails.
  """
  target_url = f'{_METADATA_SERVER_URL}{category}/{attribute}'

  session = requests.Session()
  retry_strategy = Retry(
      total=retries,
      backoff_factor=0.5,
      # Retry on the following status codes
      status_forcelist=[429, 500, 502, 503, 504],
  )
  adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
  session.mount('http://', adapter)

  try:
    response = session.get(
        target_url, headers=_METADATA_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    return response.text
  except requests.exceptions.RequestException as e:
    logging.warning(
        'Failed to retrieve metadata for %s/%s: %s', category, attribute, e
    )
    return None


def get_gcp_project_id():
  """Returns the project id of the current GCP project."""
  return get_gcp_metadata('project', 'project-id')


def get_node_zone():
  """Returns the zone of the GCE instance."""
  zone_path = get_gcp_metadata('instance', 'zone')
  # example zone_path: "projects/123456789/zones/us-central1-a"
  return zone_path.rsplit('/', 1)[-1] if zone_path else None


def get_accelerator_type():
  """Retrieves the accelerator type from GCP metadata.

  For GKE TPU VMs, it extracts the type from the 'machine-type' metadata.

  Returns:
    str: The accelerator type, or 'UNKNOWN' if not found.
  """
  machine_type_url = get_gcp_metadata('instance', 'machine-type')
  # example machine_type_url: "projects/123456789/machineTypes/a3-highgpu-8g"
  machine_type_name = (
      machine_type_url.split('/')[-1] if machine_type_url else None
  )

  if not machine_type_name:
    return 'UNKNOWN'

  for (
      prefix,
      accelerator_type,
  ) in MACHINE_TYPE_TO_ACCELERATOR_TYPE_MAPPING.items():
    if prefix.lower() in machine_type_name.lower():
      return accelerator_type

  return 'UNKNOWN'
