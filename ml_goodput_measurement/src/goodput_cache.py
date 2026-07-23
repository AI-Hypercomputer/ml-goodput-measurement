"""Goodput Cache implementations."""

import datetime
import json
import logging
import os
from typing import Any

from cloud_goodput.ml_goodput_measurement.src import goodput_utils


StepInfo = goodput_utils.StepInfo
GoodputInfo = goodput_utils.GoodputInfo
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_STEP_START_TIME = 'step_start_time'

logger = logging.getLogger(__name__)


class GoodputCache:
  """Goodput Cache with optional file-backed persistent storage.

  By default, entries are appended to a JSONL file under cache_dir (/tmp)
  and the Cloud Logging cursor is persisted in a companion JSON sidecar.
  On process restart the cursor enables incremental reads instead of a full
  Cloud Logging replay.

  Set cache_dir=None for the original in-memory behavior.
  GCS sync (gcs_cache_path) is opt-in and enables cross-restart recovery even
  when the local node is replaced.
  """

  def __init__(
      self,
      job_name: str = '',
      cache_dir: str | None = '/tmp',
      gcs_cache_path: str | None = None,
      cache_key: str = '',
  ):
    self._job_name = job_name
    self._gcs_cache_path = gcs_cache_path
    self._use_file_cache = False
    self._cached_entries: list[Any] = []  # used only when _use_file_cache=False
    self._step_entries: list[Any] = []    # used only when _use_file_cache=False
    self._local_timeline_path: str | None = None
    self._local_cursor_path: str | None = None

    if cache_dir and job_name:
      self._try_init_file_cache(cache_dir, cache_key)

    # Small memory-bound metadata always kept in-memory.
    self._last_entry_info: tuple[datetime.datetime, str] | None = None
    self._job_start_time: datetime.datetime | None = None
    self._job_end_time: datetime.datetime | None = None
    self._step_info: StepInfo | None = None
    self._goodput_info: GoodputInfo | None = None

    if self._use_file_cache:
      self._load_cursor_from_file()

  def _try_init_file_cache(self, cache_dir: str, cache_key: str) -> None:
    """Set up file-backed storage; falls back to in-memory on any OS error."""
    try:
      os.makedirs(cache_dir, exist_ok=True)
      suffix = f'_{cache_key}' if cache_key else ''
      self._local_timeline_path = os.path.join(
          cache_dir, f'{self._job_name}{suffix}_timeline.jsonl'
      )
      self._local_cursor_path = os.path.join(
          cache_dir, f'{self._job_name}{suffix}_cursor.json'
      )
      with open(self._local_timeline_path, 'a'):
        pass
      self._use_file_cache = True
    except OSError as e:
      logger.warning(
          'Cannot use file cache in %s: %s. Falling back to in-memory.',
          cache_dir,
          e,
      )

  def _load_cursor_from_file(self) -> None:
    """Reload last_entry_info from the cursor sidecar."""
    if (
        not self._local_cursor_path
        or not os.path.exists(self._local_cursor_path)
    ):
      return
    try:
      with open(self._local_cursor_path, 'r') as f:
        data = json.load(f)
      ts = datetime.datetime.fromisoformat(data['last_entry_timestamp'])
      self._last_entry_info = (ts, data['last_entry_id'])
    except (OSError, KeyError, ValueError) as e:
      logger.warning(
          'Failed to load cursor from %s: %s', self._local_cursor_path, e
      )

  def update_step_info(self, step_info: StepInfo):
    """Updates the step information."""
    self._step_info = step_info

  def update_cached_entries(
      self, entries: list[Any], last_entry_info: tuple[datetime.datetime, str]
  ):
    """Persists new log entries and advances the cursor."""
    if entries:
      if self._use_file_cache:
        try:
          with open(self._local_timeline_path, 'a') as f:
            for entry in entries:
              f.write(json.dumps(entry) + '\n')
        except OSError as e:
          logger.warning(
              'Failed to append to timeline file: %s. Falling back to in-memory.', e
          )
          self._use_file_cache = False
          self._cached_entries.extend(entries)
          self._step_entries.extend(
              [e for e in entries if _STEP_START_TIME in e]
          )
      else:
        self._cached_entries.extend(entries)
        self._step_entries.extend([e for e in entries if _STEP_START_TIME in e])

      self._update_times_from_entries(entries)

    if last_entry_info and last_entry_info[0] is not None:
      self.update_last_entry_info(last_entry_info)

  def _update_times_from_entries(self, entries: list[Any]) -> None:
    """Update in-memory job start/end times from a batch of new entries."""
    for entry in entries:
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
    """Updates the cursor and atomically persists it to the sidecar file."""
    self._last_entry_info = last_entry_info
    if self._use_file_cache:
      try:
        tmp_path = self._local_cursor_path + '.tmp'
        with open(tmp_path, 'w') as f:
          json.dump(
              {
                  'last_entry_timestamp': last_entry_info[0].isoformat(),
                  'last_entry_id': last_entry_info[1],
              },
              f,
          )
        os.replace(tmp_path, self._local_cursor_path)
      except OSError as e:
        logger.warning('Failed to persist cursor: %s', e)

  def update_goodput_info(self, goodput_info: GoodputInfo):
    """Updates the last computed Goodput information."""
    self._goodput_info = goodput_info

  def get_cached_entries(self) -> list[Any]:
    """Returns all cached entries (reads JSONL file in file-backed mode)."""
    if self._use_file_cache:
      return self._read_entries_from_file()
    return self._cached_entries

  def _read_entries_from_file(self) -> list[Any]:
    entries: list[Any] = []
    if not self._local_timeline_path:
      return entries
    try:
      with open(self._local_timeline_path, 'r') as f:
        for line in f:
          line = line.strip()
          if line:
            entries.append(json.loads(line))
    except FileNotFoundError:
      pass
    except (OSError, json.JSONDecodeError) as e:
      logger.warning('Error reading timeline file: %s', e)
    return entries

  def get_step_entries(self) -> list[Any]:
    """Returns step entries (filters JSONL file in file-backed mode)."""
    if self._use_file_cache:
      return [e for e in self._read_entries_from_file() if _STEP_START_TIME in e]
    return self._step_entries

  def get_goodput_info(self) -> GoodputInfo | None:
    """Returns the last computed Goodput information."""
    return self._goodput_info

  def get_job_start_time(self) -> datetime.datetime | None:
    """Returns the job start time."""
    return self._job_start_time

  def get_job_end_time(self) -> datetime.datetime | None:
    """Returns the job end time."""
    return self._job_end_time

  def get_last_entry_info(self) -> tuple[datetime.datetime, str] | None:
    """Returns the cursor (last Cloud Logging entry timestamp and insertId)."""
    return self._last_entry_info

  def get_step_info(self) -> StepInfo | None:
    """Returns the step information."""
    return self._step_info

  def clear_cache(self):
    """Clears entries and cursor while preserving job metadata."""
    if self._use_file_cache:
      for path in (self._local_timeline_path, self._local_cursor_path):
        if path and os.path.exists(path):
          try:
            os.remove(path)
          except OSError as e:
            logger.warning('Failed to remove cache file %s: %s', path, e)
    else:
      self._cached_entries = []
      self._step_entries = []
    self._last_entry_info = None
    self._goodput_info = None
    # Preserves _job_start_time, _job_end_time, _step_info (original behavior).

  def is_cache_empty(self) -> bool:
    """Returns True if no entries have been persisted yet."""
    if self._use_file_cache:
      try:
        return (
            not self._local_timeline_path
            or not os.path.exists(self._local_timeline_path)
            or os.path.getsize(self._local_timeline_path) == 0
        )
      except OSError:
        return True
    return not self._cached_entries

  def sync_to_gcs(self) -> None:
    """Upload local timeline and cursor files to GCS (no-op if not configured)."""
    if not self._use_file_cache or not self._gcs_cache_path:
      return
    try:
      from google.cloud import storage  # pylint: disable=g-import-not-at-top
      client = storage.Client()
      bucket_name, blob_prefix = self._parse_gcs_path()
      bucket = client.bucket(bucket_name)
      for local_path in (self._local_timeline_path, self._local_cursor_path):
        if local_path and os.path.exists(local_path):
          blob_name = '/'.join(
              filter(None, [blob_prefix, os.path.basename(local_path)])
          )
          bucket.blob(blob_name).upload_from_filename(local_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning('Failed to sync cache to GCS: %s', e)

  def restore_from_gcs(self) -> None:
    """Download timeline and cursor from GCS into the local cache dir."""
    if not self._use_file_cache or not self._gcs_cache_path:
      return
    try:
      from google.cloud import storage  # pylint: disable=g-import-not-at-top
      client = storage.Client()
      bucket_name, blob_prefix = self._parse_gcs_path()
      bucket = client.bucket(bucket_name)
      for local_path in (self._local_timeline_path, self._local_cursor_path):
        if not local_path:
          continue
        blob_name = '/'.join(
            filter(None, [blob_prefix, os.path.basename(local_path)])
        )
        blob = bucket.blob(blob_name)
        if blob.exists():
          blob.download_to_filename(local_path)
      self._load_cursor_from_file()
      self._update_times_from_entries(self._read_entries_from_file())
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning('Failed to restore cache from GCS: %s', e)

  def _parse_gcs_path(self) -> tuple[str, str]:
    path = self._gcs_cache_path.rstrip('/')
    if path.startswith('gs://'):
      path = path[5:]
    parts = path.split('/', 1)
    return parts[0], (parts[1] if len(parts) > 1 else '')
