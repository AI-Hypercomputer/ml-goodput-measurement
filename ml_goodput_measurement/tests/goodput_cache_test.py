"""Tests to unit test GoodputCache class."""

import datetime
import json
import os
import tempfile
from unittest import mock

from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_utils

from google3.testing.pybase import googletest

BadputType = goodput_utils.BadputType
GoodputInfo = goodput_utils.GoodputInfo


class GoodputCacheInMemoryTest(googletest.TestCase):
  """Tests for in-memory mode (cache_dir=None) — preserves original behavior."""

  def setUp(self):
    super().setUp()
    self.goodput_cache = goodput_cache.GoodputCache(
        job_name='test-job', cache_dir=None
    )

  def test_update_cached_entries(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries, (3, 'mock_entry-3'))
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(self.goodput_cache.get_cached_entries(), mock_entries)

  def test_update_goodput_info(self):
    goodput_info = GoodputInfo(
        total_productive_time=100,
        total_elapsed_time=200,
        total_unproductive_time={
            BadputType.TPU_INITIALIZATION: 10,
            BadputType.TRAINING_PREP: 10,
            BadputType.DATA_LOADING_SYNC: 30,
            BadputType.PROGRAM_STARTUP: 10,
            BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME: 20,
            BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME: 10,
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
            BadputType.OTHER: 10,
        },
        max_productive_step=3,
        last_recorded_step=3,
        number_of_disruptions=1,
    )
    self.goodput_cache.update_goodput_info(goodput_info)
    self.assertEqual(self.goodput_cache._goodput_info, goodput_info)

  def test_clear_cache(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries, (3, 'mock_entry-3'))
    self.goodput_cache.update_goodput_info(
        GoodputInfo(
            total_productive_time=100,
            total_elapsed_time=200,
            total_unproductive_time={
                BadputType.TPU_INITIALIZATION: 10,
                BadputType.TRAINING_PREP: 10,
                BadputType.DATA_LOADING_SYNC: 30,
                BadputType.PROGRAM_STARTUP: 10,
                BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME: 20,
                BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME: 10,
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
                BadputType.OTHER: 10,
            },
            max_productive_step=3,
            last_recorded_step=3,
            number_of_disruptions=1,
        )
    )
    self.goodput_cache.clear_cache()
    self.assertEqual(self.goodput_cache.get_cached_entries(), [])
    self.assertIsNone(self.goodput_cache._goodput_info)

  def test_is_cache_empty(self):
    self.assertTrue(self.goodput_cache.is_cache_empty())
    self.goodput_cache.update_cached_entries(
        [
            {'time': 1, 'step': 1},
            {'time': 2, 'step': 2},
            {'time': 3, 'step': 3},
        ],
        (3, 'mock_entry-3'),
    )
    self.assertFalse(self.goodput_cache.is_cache_empty())

  def test_get_step_info(self):
    step_info = goodput_utils.StepInfo(
        step_deviations={1: 1.0, 2: 2.0},
        ideal_step_time=1.0,
    )
    self.goodput_cache.update_step_info(step_info)
    self.assertEqual(self.goodput_cache._step_info, step_info)

  def test_update_job_start_time(self):
    self.assertIsNone(self.goodput_cache._job_start_time)
    self.goodput_cache.update_cached_entries(
        [
            {'step_start_time': 2, 'step': 1},
            {'step_start_time': 3, 'step': 2},
            {'job_end_time': 4},
        ],
        (3, 'mock_entry-3'),
    )
    self.assertIsNone(self.goodput_cache._job_start_time)
    self.goodput_cache.update_cached_entries(
        [
            {'job_start_time': 1},
            {'job_start_time': 9},
            {'step_start_time': 2, 'step': 1},
            {'step_start_time': 3, 'step': 2},
            {'job_end_time': 4},
        ],
        (3, 'mock_entry-3'),
    )
    self.assertEqual(
        self.goodput_cache._job_start_time,
        datetime.datetime.fromtimestamp(1, tz=datetime.timezone.utc),
    )

  def test_update_job_end_time(self):
    self.assertIsNone(self.goodput_cache._job_end_time)
    self.goodput_cache.update_cached_entries(
        [
            {'job_end_time': 1},
            {'job_end_time': 2},
            {'job_end_time': 3},
        ],
        (3, 'mock_entry-3'),
    )
    self.assertEqual(
        self.goodput_cache._job_end_time,
        datetime.datetime.fromtimestamp(3, tz=datetime.timezone.utc),
    )


class GoodputCacheFileBackedTest(googletest.TestCase):
  """Tests for file-backed mode (default)."""

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.mkdtemp()
    self.goodput_cache = goodput_cache.GoodputCache(
        job_name='test-job',
        cache_dir=self._tmpdir,
        cache_key='test',
    )

  def _make_cache(self, **kwargs):
    """Create a fresh GoodputCache pointing at the same tmpdir."""
    return goodput_cache.GoodputCache(
        job_name='test-job',
        cache_dir=self._tmpdir,
        cache_key='test',
        **kwargs,
    )

  def test_file_backed_mode_enabled(self):
    self.assertTrue(self.goodput_cache._use_file_cache)
    self.assertIsNotNone(self.goodput_cache._local_timeline_path)

  def test_entries_written_to_file(self):
    entries = [{'time': 1, 'step': 1}, {'time': 2, 'step': 2}]
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries(entries, (ts, 'id-2'))
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(self.goodput_cache.get_cached_entries(), entries)

  def test_entries_persist_across_instances(self):
    entries = [{'time': 1, 'step': 1}, {'time': 2, 'step': 2}]
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries(entries, (ts, 'id-2'))

    # New instance pointing at same directory reads the same data.
    cache2 = self._make_cache()
    self.assertFalse(cache2.is_cache_empty())
    self.assertEqual(cache2.get_cached_entries(), entries)

  def test_cursor_persisted_and_reloaded(self):
    ts = datetime.datetime(2024, 6, 15, 12, 0, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries([{'step': 1}], (ts, 'entry-abc'))
    self.assertEqual(self.goodput_cache.get_last_entry_info(), (ts, 'entry-abc'))

    cache2 = self._make_cache()
    self.assertEqual(cache2.get_last_entry_info(), (ts, 'entry-abc'))

  def test_cursor_written_atomically(self):
    ts = datetime.datetime(2024, 6, 15, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_last_entry_info((ts, 'id-1'))
    # .tmp file should not remain after a successful write.
    self.assertFalse(
        os.path.exists(self.goodput_cache._local_cursor_path + '.tmp')
    )
    self.assertTrue(os.path.exists(self.goodput_cache._local_cursor_path))

  def test_clear_cache_removes_files_and_cursor(self):
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries([{'step': 1}], (ts, 'id-1'))
    self.goodput_cache.clear_cache()

    self.assertTrue(self.goodput_cache.is_cache_empty())
    self.assertIsNone(self.goodput_cache.get_last_entry_info())
    self.assertFalse(os.path.exists(self.goodput_cache._local_timeline_path))

  def test_clear_cache_preserves_job_metadata(self):
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries(
        [{'job_start_time': 1000}], (ts, 'id-1')
    )
    expected_start = datetime.datetime.fromtimestamp(
        1000, tz=datetime.timezone.utc
    )
    self.goodput_cache.clear_cache()
    # job_start_time must survive clear_cache.
    self.assertEqual(self.goodput_cache._job_start_time, expected_start)

  def test_get_step_entries_filters_from_file(self):
    entries = [
        {'step_start_time': 1.0, 'step': 1},
        {'job_start_time': 0.0},
        {'step_start_time': 2.0, 'step': 2},
    ]
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    self.goodput_cache.update_cached_entries(entries, (ts, 'id-3'))
    step_entries = self.goodput_cache.get_step_entries()
    self.assertLen(step_entries, 2)
    self.assertIn('step_start_time', step_entries[0])

  def test_fallback_to_in_memory_on_bad_cache_dir(self):
    cache = goodput_cache.GoodputCache(
        job_name='test-job',
        cache_dir='/nonexistent/path/that/cannot/be/created/xyz',
    )
    self.assertFalse(cache._use_file_cache)
    entries = [{'step': 1}]
    ts = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    cache.update_cached_entries(entries, (ts, 'id-1'))
    self.assertEqual(cache.get_cached_entries(), entries)

  def test_cache_key_produces_distinct_files(self):
    cache_a = goodput_cache.GoodputCache(
        job_name='myjob', cache_dir=self._tmpdir, cache_key='goodput'
    )
    cache_b = goodput_cache.GoodputCache(
        job_name='myjob', cache_dir=self._tmpdir, cache_key='step_dev'
    )
    self.assertNotEqual(
        cache_a._local_timeline_path, cache_b._local_timeline_path
    )

  def test_gcs_restore_called_on_cold_start(self):
    with mock.patch.object(
        self.goodput_cache, 'restore_from_gcs'
    ) as mock_restore:
      # Simulate what _fetch_new_entries does on a true cold start.
      self.assertTrue(self.goodput_cache.is_cache_empty())
      self.assertIsNone(self.goodput_cache.get_last_entry_info())
      self.goodput_cache.restore_from_gcs()
      mock_restore.assert_called_once()


if __name__ == '__main__':
  googletest.main()
