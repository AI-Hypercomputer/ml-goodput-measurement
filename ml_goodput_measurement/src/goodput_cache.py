"""Goodput Cache implementations backed by local file and GCS."""

import datetime
import json
import logging
import os
from typing import Any, Optional
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from google.cloud import storage

StepInfo = goodput_utils.StepInfo
GoodputInfo = goodput_utils.GoodputInfo
_JOB_START_TIME = 'job_start_time'
_JOB_END_TIME = 'job_end_time'
_STEP_START_TIME = 'step_start_time'
_LAST_ENTRY_TIMESTAMP = 'last_entry_timestamp'
_LAST_ENTRY_ID = 'last_entry_id'
_RECENT_ENTRY_IDS = 'recent_entry_ids'
_RECENT_ENTRY_IDS_SINCE = 'recent_entry_ids_since'

logger = logging.getLogger(__name__)


def _parse_timestamp(value: str) -> datetime.datetime:
  """Parses an ISO 8601 timestamp as written by `datetime.isoformat`."""
  return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _chronological(entries: list[Any]) -> list[Any]:
  """Sorts entries by their own timestamps.

  The calculator replays the timeline as a sequential state machine, so an
  entry that was read late (see `_CloudLogger.read_new_entries`) has to move to
  its chronological position. The sort is stable; an entry without a timestamp
  keeps its position by reusing the previous entry's key. Orbax save and
  restore entries sort by their start time instead of their log time, which is
  harmless because the calculator extracts checkpoint events separately
  (`_extract_checkpoint_save_events`, `_extract_restore_intervals`,
  `CheckpointBadputCalculator`) and matches them to steps by time window.

  Args:
    entries: Entry payloads in the order they were appended to the timeline.

  Returns:
    The same entries in chronological order.
  """
  keys = []
  previous_key = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
  for entry in entries:
    timestamp = (
        goodput_utils.get_timestamp_from_log_entry(entry)
        if isinstance(entry, dict)
        else None
    )
    if timestamp is not None:
      previous_key = timestamp
    keys.append(previous_key)
  return [
      entry for _, entry in sorted(zip(keys, entries), key=lambda pair: pair[0])
  ]


class GoodputCache:
  """Goodput Cache backed by local file and GCS."""

  def __init__(
      self,
      job_name: str,
      cache_dir: str = '/tmp',
      gcs_path: Optional[str] = None,
      read_overlap_seconds: float = 300.0,
  ):
    """Initializes the cache, resuming from GCS or local files if present.

    Args:
      job_name: Name of the job the cache is for.
      cache_dir: Local directory holding the timeline and metadata files.
      gcs_path: Optional GCS path the cache files are restored from and synced
        to.
      read_overlap_seconds: How far before the cursor the calculator re-reads
        Cloud Logging; insertIds of entries older than that are forgotten.
    """
    self._job_name = job_name
    self._cache_dir = cache_dir
    self._local_timeline_path = os.path.join(
        cache_dir, f'{job_name}_timeline.jsonl'
    )
    self._local_metadata_path = os.path.join(
        cache_dir, f'{job_name}_metadata.json'
    )
    self._gcs_path = gcs_path
    self._read_overlap_seconds = read_overlap_seconds

    self._last_entry_info = None
    # insertId -> log timestamp of recently appended entries, and the timestamp
    # from which on every appended entry's id is known.
    self._recent_entry_ids: dict[str, datetime.datetime] = {}
    self._recent_entry_ids_since: Optional[datetime.datetime] = None
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
            path_parts[1], 'goodput_cache', f'{job_name}_timeline.jsonl'
        )
        self._gcs_metadata_blob_name = os.path.join(
            path_parts[1], 'goodput_cache', f'{job_name}_metadata.json'
        )

    # Try to restore from GCS on startup (Cold Start Recovery)
    self._restore_from_gcs()
    # Resume from a local cache left by an earlier process (e.g. a restart on
    # the same host); without the cursor, new entries would never be fetched.
    self._initialize_times_from_file()
    self._initialize_metadata_from_file()
    if self._last_entry_info is None and not self.is_cache_empty():
      logger.warning(
          'The cached timeline for job %s has no read cursor; discarding it so'
          ' that all entries are read again from Cloud Logging.',
          job_name,
      )
      self.clear_cache()

  def _restore_from_gcs(self):
    """Restores the timeline and metadata files from GCS if they exist."""
    if not self._gcs_bucket_name:
      return
    try:
      os.makedirs(self._cache_dir, exist_ok=True)
      client = storage.Client()
      bucket = client.bucket(self._gcs_bucket_name)

      # Restore timeline file
      if self._gcs_timeline_blob_name:
        blob = bucket.blob(self._gcs_timeline_blob_name)
        if blob.exists():
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
    """Initializes the cursor and recent entry ids from the metadata file."""
    if not os.path.exists(self._local_metadata_path):
      return
    try:
      with open(self._local_metadata_path, 'r') as f:
        data = json.load(f)
      last_entry_ts_str = data.get(_LAST_ENTRY_TIMESTAMP)
      if not last_entry_ts_str:
        return
      last_entry_ts = _parse_timestamp(last_entry_ts_str)
      self._last_entry_info = (last_entry_ts, data.get(_LAST_ENTRY_ID))
      self._recent_entry_ids = {
          insert_id: _parse_timestamp(timestamp)
          for insert_id, timestamp in data.get(_RECENT_ENTRY_IDS, [])
      }
      # Metadata written before ids were tracked knows nothing below the
      # cursor, so no overlapping read may start before it.
      ids_since = data.get(_RECENT_ENTRY_IDS_SINCE)
      self._recent_entry_ids_since = (
          _parse_timestamp(ids_since) if ids_since else last_entry_ts
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error reading metadata file: {e}')

  def update_step_info(self, step_info: StepInfo):
    """Updates the step information."""
    self._step_info = step_info

  def update_cached_entries(
      self,
      entries: list[Any],
      last_entry_info: Optional[
          tuple[Optional[datetime.datetime], Optional[str]]
      ],
      entry_ids: Optional[list[tuple[str, datetime.datetime]]] = None,
  ):
    """Appends entries to the local timeline and advances the cursor.

    Args:
      entries: Entry payloads to append, in the order they were read.
      last_entry_info: `(timestamp, insert_id)` of the newest entry read so
        far; ignored when the timestamp is None.
      entry_ids: `(insert_id, timestamp)` of each appended entry, remembered so
        that a later overlapping read does not append the entry again. None
        when the ids are unknown, in which case overlapping reads may not start
        before this cursor.
    """
    if entries:
      os.makedirs(self._cache_dir, exist_ok=True)
      with open(self._local_timeline_path, 'a') as f:
        for entry in entries:
          f.write(json.dumps(entry) + '\n')
      self._update_times_from_new_entries(entries)
    for insert_id, timestamp in entry_ids or []:
      self._recent_entry_ids[insert_id] = timestamp

    cursor_ts, cursor_id = last_entry_info or (None, None)
    if cursor_ts is None:
      return
    if isinstance(cursor_ts, datetime.datetime):
      self._prune_recent_entry_ids(
          cursor_ts, ids_known=entry_ids is not None or not entries
      )
    self.update_last_entry_info((cursor_ts, cursor_id))

  def _prune_recent_entry_ids(
      self, cursor_ts: datetime.datetime, ids_known: bool
  ):
    """Forgets ids that no overlapping read starting at `cursor_ts` returns."""
    window_start = cursor_ts - datetime.timedelta(
        seconds=self._read_overlap_seconds
    )
    ids_since = window_start if ids_known else cursor_ts
    if self._recent_entry_ids_since is not None:
      ids_since = max(ids_since, self._recent_entry_ids_since)
    self._recent_entry_ids_since = ids_since
    self._recent_entry_ids = {
        insert_id: timestamp
        for insert_id, timestamp in self._recent_entry_ids.items()
        if timestamp >= ids_since
    }

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
      self, last_entry_info: tuple[datetime.datetime, Optional[str]]
  ):
    """Updates the last entry's timestamp and unique identifier."""
    self._last_entry_info = last_entry_info
    try:
      metadata = {
          _LAST_ENTRY_TIMESTAMP: last_entry_info[0].isoformat(),
          _LAST_ENTRY_ID: last_entry_info[1],
          _RECENT_ENTRY_IDS: [
              [insert_id, timestamp.isoformat()]
              for insert_id, timestamp in self._recent_entry_ids.items()
          ],
      }
      if self._recent_entry_ids_since is not None:
        metadata[_RECENT_ENTRY_IDS_SINCE] = (
            self._recent_entry_ids_since.isoformat()
        )
      # Serialize before truncating the file so a failure cannot corrupt it.
      serialized = json.dumps(metadata)
      os.makedirs(self._cache_dir, exist_ok=True)
      with open(self._local_metadata_path, 'w') as f:
        f.write(serialized)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error saving metadata: {e}')

  def update_goodput_info(self, goodput_info: GoodputInfo):
    """Updates the last computed Goodput information."""
    self._goodput_info = goodput_info

  def get_cached_entries(self) -> list[Any]:
    """Loads all entries from disk into memory, in chronological order."""
    entries = []
    if os.path.exists(self._local_timeline_path):
      try:
        with open(self._local_timeline_path, 'r') as f:
          for line in f:
            if line.strip():
              entries.append(json.loads(line))
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'Error reading timeline file: {e}')
    return _chronological(entries)

  def get_step_entries(self) -> list[Any]:
    """Loads and filters step entries from disk, in chronological order."""
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
    return _chronological(step_entries)

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

  def get_recent_entry_ids(self) -> dict[str, datetime.datetime]:
    """Returns the insertIds of recently appended entries by log timestamp."""
    return self._recent_entry_ids

  def get_read_overlap_seconds(self) -> float:
    """Returns how far before the cursor a read may re-check for late entries.

    Capped at the window whose insertIds are known, so entries appended before
    ids were tracked (an older metadata file, or an overlap that grew) are not
    appended a second time. Zero when there is no cursor.
    """
    if self._last_entry_info is None or self._recent_entry_ids_since is None:
      return 0.0
    cursor_ts = self._last_entry_info[0]
    if not isinstance(cursor_ts, datetime.datetime):
      return 0.0
    known_window = (cursor_ts - self._recent_entry_ids_since).total_seconds()
    return max(0.0, min(self._read_overlap_seconds, known_window))

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
    self._recent_entry_ids = {}
    self._recent_entry_ids_since = None
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
