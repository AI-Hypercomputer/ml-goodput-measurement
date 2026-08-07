"""Goodput Cache implementations backed by local file and GCS."""

import datetime
import json
import os
from typing import Any, Optional
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from google.cloud import storage

StepInfo = goodput_utils.StepInfo
GoodputInfo = goodput_utils.GoodputInfo
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_STEP_START_TIME = 'step_start_time'


class GoodputCache:
  """Goodput Cache backed by local file and GCS."""

  def __init__(
      self,
      job_name: str,
      cache_dir: str = '/tmp',
      gcs_path: Optional[str] = None,
  ):
    self._job_name = job_name
    self._cache_dir = cache_dir
    self._local_timeline_path = os.path.join(
        cache_dir, f'{job_name}_timeline.jsonl'
    )
    self._local_metadata_path = os.path.join(
        cache_dir, f'{job_name}_metadata.json'
    )
    self._gcs_path = gcs_path

    self._last_entry_info = None
    self._job_start_time = None
    self._job_end_time = None
    self._step_info = None
    self._goodput_info = None

    self._gcs_bucket_name = None
    self._gcs_timeline_blob_name = None
    self._gcs_metadata_blob_name = None

    if gcs_path and gcs_path.startswith('gs://'):
      path_parts = gcs_path[5:].split('/', 1)
      self._gcs_bucket_name = path_parts[0]
      if len(path_parts) > 1:
        self._gcs_timeline_blob_name = os.path.join(
            path_parts[1], f'{job_name}_timeline.jsonl'
        )
        self._gcs_metadata_blob_name = os.path.join(
            path_parts[1], f'{job_name}_metadata.json'
        )

    # Try to restore from GCS on startup (Cold Start Recovery)
    self._restore_from_gcs()

  def _restore_from_gcs(self):
    """Restores the timeline and metadata files from GCS if they exist."""
    if not self._gcs_bucket_name:
      return
    try:
      client = storage.Client()
      bucket = client.bucket(self._gcs_bucket_name)

      # Restore timeline file
      if self._gcs_timeline_blob_name:
        blob = bucket.blob(self._gcs_timeline_blob_name)
        if blob.exists():
          os.makedirs(self._cache_dir, exist_ok=True)
          blob.download_to_filename(self._local_timeline_path)
          self._initialize_times_from_file()

      # Restore metadata file (cursor)
      if self._gcs_metadata_blob_name:
        blob = bucket.blob(self._gcs_metadata_blob_name)
        if blob.exists():
          blob.download_to_filename(self._local_metadata_path)
          self._initialize_metadata_from_file()
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Log warning but don't fail, we can start with empty cache
      print(f'Warning: Failed to restore cache from GCS: {e}')

  def _initialize_times_from_file(self):
    """Initializes job start and end times from the local timeline file."""
    if not os.path.exists(self._local_timeline_path):
      return
    try:
      with open(self._local_timeline_path, 'r') as f:
        for line in f:
          if line.strip():
            entry = json.loads(line)
            if self._job_start_time is None and _JOB_START_TIME in entry:
              self._job_start_time = datetime.datetime.fromtimestamp(
                  entry[_JOB_START_TIME], tz=datetime.timezone.utc
              )
            if _JOB_END_TIME in entry:
              self._job_end_time = datetime.datetime.fromtimestamp(
                  entry[_JOB_END_TIME], tz=datetime.timezone.utc
              )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error reading times from file: {e}')

  def _initialize_metadata_from_file(self):
    """Initializes metadata (last entry info) from the local metadata file."""
    if not os.path.exists(self._local_metadata_path):
      return
    try:
      with open(self._local_metadata_path, 'r') as f:
        data = json.load(f)
        last_entry_ts_str = data.get('last_entry_timestamp')
        last_entry_id = data.get('last_entry_id')
        if last_entry_ts_str:
          ts_str = last_entry_ts_str.replace('Z', '+00:00')
          last_entry_ts = datetime.datetime.fromisoformat(ts_str)
          self._last_entry_info = (last_entry_ts, last_entry_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error reading metadata file: {e}')

  def update_step_info(self, step_info: StepInfo):
    """Updates the step information."""
    self._step_info = step_info

  def update_cached_entries(
      self, entries: list[Any], last_entry_info: tuple[datetime.datetime, str]
  ):
    """Updates the cached entries by appending to local file."""
    if entries:
      os.makedirs(self._cache_dir, exist_ok=True)
      with open(self._local_timeline_path, 'a') as f:
        for entry in entries:
          f.write(json.dumps(entry) + '\n')
      self._update_times_from_new_entries(entries)

    if last_entry_info and last_entry_info[0] is not None:
      self.update_last_entry_info(last_entry_info)

  def _update_times_from_new_entries(self, new_entries):
    for entry in new_entries:
      if self._job_start_time is None and _JOB_START_TIME in entry:
        self._job_start_time = datetime.datetime.fromtimestamp(
            entry[_JOB_START_TIME], tz=datetime.timezone.utc
        )
      if _JOB_END_TIME in entry:
        self._job_end_time = datetime.datetime.fromtimestamp(
            entry[_JOB_END_TIME], tz=datetime.timezone.utc
        )

  def update_last_entry_info(
      self, last_entry_info: tuple[datetime.datetime, str]
  ):
    """Updates the last entry's timestamp and unique identifier."""
    self._last_entry_info = last_entry_info
    try:
      os.makedirs(self._cache_dir, exist_ok=True)
      with open(self._local_metadata_path, 'w') as f:
        json.dump(
            {
                'last_entry_timestamp': last_entry_info[0].isoformat(),
                'last_entry_id': last_entry_info[1],
            },
            f,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error saving metadata: {e}')

  def update_goodput_info(self, goodput_info: GoodputInfo):
    """Updates the last computed Goodput information."""
    self._goodput_info = goodput_info

  def get_cached_entries(self) -> list[Any]:
    """Loads all entries from disk into memory."""
    entries = []
    if os.path.exists(self._local_timeline_path):
      try:
        with open(self._local_timeline_path, 'r') as f:
          for line in f:
            if line.strip():
              entries.append(json.loads(line))
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'Error reading timeline file: {e}')
    return entries

  def get_step_entries(self) -> list[Any]:
    """Loads and filters step entries from disk."""
    step_entries = []
    if os.path.exists(self._local_timeline_path):
      try:
        with open(self._local_timeline_path, 'r') as f:
          for line in f:
            if line.strip():
              entry = json.loads(line)
              if _STEP_START_TIME in entry:
                step_entries.append(entry)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'Error reading step entries from file: {e}')
    return step_entries

  def get_goodput_info(self) -> Optional[GoodputInfo]:
    """Returns the last computed Goodput information."""
    return self._goodput_info

  def get_job_start_time(self) -> Optional[datetime.datetime]:
    """Returns the job start time."""
    return self._job_start_time

  def get_job_end_time(self) -> Optional[datetime.datetime]:
    """Returns the job end time."""
    return self._job_end_time

  def get_last_entry_info(self) -> Optional[tuple[datetime.datetime, str]]:
    """Returns the last entry info (timestamp and unique identifier)."""
    return self._last_entry_info

  def get_step_info(self) -> Optional[StepInfo]:
    """Returns the step information."""
    return self._step_info

  def sync_to_gcs(self):
    """Uploads local timeline and metadata files to GCS."""
    if not self._gcs_bucket_name:
      return
    try:
      client = storage.Client()
      bucket = client.bucket(self._gcs_bucket_name)

      if self._gcs_timeline_blob_name and os.path.exists(
          self._local_timeline_path
      ):
        blob = bucket.blob(self._gcs_timeline_blob_name)
        blob.upload_from_filename(self._local_timeline_path)

      if self._gcs_metadata_blob_name and os.path.exists(
          self._local_metadata_path
      ):
        blob = bucket.blob(self._gcs_metadata_blob_name)
        blob.upload_from_filename(self._local_metadata_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error syncing to GCS: {e}')

  def clear_cache(self):
    """Clears the local cache files."""
    try:
      if os.path.exists(self._local_timeline_path):
        os.remove(self._local_timeline_path)
      if os.path.exists(self._local_metadata_path):
        os.remove(self._local_metadata_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error clearing cache files: {e}')
    self._last_entry_info = None
    self._goodput_info = None
    self._job_start_time = None
    self._job_end_time = None
    self._step_info = None

  def is_cache_empty(self) -> bool:
    """Checks if the cache is empty."""
    return (
        not os.path.exists(self._local_timeline_path)
        or os.path.getsize(self._local_timeline_path) == 0
    )
