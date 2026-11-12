"""Tests to unit test GoodputCache class."""

import datetime
import json
import os
from typing import Any

from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_utils

from google3.testing.pybase import googletest

BadputType = goodput_utils.BadputType
GoodputInfo = goodput_utils.GoodputInfo

_CURSOR_TIME = datetime.datetime(
    2025, 1, 1, 0, 5, 0, tzinfo=datetime.timezone.utc
)


def _step_entry(step: int, timestamp: datetime.datetime) -> dict[str, float]:
  return {'step_count': step, 'step_start_time': timestamp.timestamp()}


def _write_timeline(
    cache_dir: str, job_name: str, entries: list[dict[str, Any]]
):
  with open(os.path.join(cache_dir, f'{job_name}_timeline.jsonl'), 'w') as f:
    for entry in entries:
      f.write(json.dumps(entry) + '\n')


class GoodputCacheTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = f'test-cache-run-{self.id().split(".")[-1]}'
    self.goodput_cache = goodput_cache.GoodputCache(job_name=self.job_name)

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

  def test_new_cache_resumes_from_local_files(self):
    cache_dir = self.create_tempdir().full_path
    cache = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=cache_dir
    )
    last_entry_ts = datetime.datetime(2025, 1, 1, 0, 0, 2, tzinfo=datetime.timezone.utc)
    cache.update_cached_entries(
        [
            {'job_start_time': 1.0, 'job_name': self.job_name},
            {'job_end_time': 2.0, 'job_name': self.job_name},
        ],
        (last_entry_ts, 'entry-2'),
    )

    resumed = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=cache_dir
    )

    self.assertEqual(resumed.get_job_start_time().timestamp(), 1.0)
    self.assertEqual(resumed.get_job_end_time().timestamp(), 2.0)
    self.assertEqual(resumed.get_last_entry_info(), (last_entry_ts, 'entry-2'))
    self.assertLen(resumed.get_cached_entries(), 2)

  def test_new_cache_without_local_files_starts_empty(self):
    cache = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=self.create_tempdir().full_path
    )
    self.assertTrue(cache.is_cache_empty())
    self.assertIsNone(cache.get_job_start_time())
    self.assertIsNone(cache.get_job_end_time())
    self.assertIsNone(cache.get_last_entry_info())

  def test_recent_entry_ids_are_pruned_to_overlap(self):
    cache = goodput_cache.GoodputCache(
        job_name=self.job_name,
        cache_dir=self.create_tempdir().full_path,
        read_overlap_seconds=100.0,
    )
    old_time = _CURSOR_TIME - datetime.timedelta(seconds=150)
    recent_time = _CURSOR_TIME - datetime.timedelta(seconds=50)

    cache.update_cached_entries(
        [
            _step_entry(0, old_time),
            _step_entry(1, recent_time),
            _step_entry(2, _CURSOR_TIME),
        ],
        (_CURSOR_TIME, 'id-2'),
        [('id-0', old_time), ('id-1', recent_time), ('id-2', _CURSOR_TIME)],
    )

    self.assertEqual(
        cache.get_recent_entry_ids(),
        {'id-1': recent_time, 'id-2': _CURSOR_TIME},
    )
    self.assertEqual(cache.get_read_overlap_seconds(), 100.0)
    with open(cache._local_metadata_path) as f:
      metadata = json.load(f)
    self.assertEqual(metadata['last_entry_id'], 'id-2')
    self.assertEqual(
        metadata['recent_entry_ids'],
        [
            ['id-1', recent_time.isoformat()],
            ['id-2', _CURSOR_TIME.isoformat()],
        ],
    )

  def test_metadata_round_trip(self):
    cache_dir = self.create_tempdir().full_path
    cache = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=cache_dir, read_overlap_seconds=100.0
    )
    earlier_time = _CURSOR_TIME - datetime.timedelta(seconds=30)
    cache.update_cached_entries(
        [_step_entry(0, earlier_time), _step_entry(1, _CURSOR_TIME)],
        (_CURSOR_TIME, 'id-1'),
        [('id-0', earlier_time), ('id-1', _CURSOR_TIME)],
    )

    resumed = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=cache_dir, read_overlap_seconds=100.0
    )

    self.assertEqual(resumed.get_last_entry_info(), (_CURSOR_TIME, 'id-1'))
    self.assertEqual(
        resumed.get_recent_entry_ids(),
        {'id-0': earlier_time, 'id-1': _CURSOR_TIME},
    )
    self.assertEqual(resumed.get_read_overlap_seconds(), 100.0)

  def test_metadata_without_ids_caps_overlap(self):
    cache_dir = self.create_tempdir().full_path
    _write_timeline(cache_dir, self.job_name, [_step_entry(0, _CURSOR_TIME)])
    metadata_path = os.path.join(cache_dir, f'{self.job_name}_metadata.json')
    with open(metadata_path, 'w') as f:
      json.dump(
          {
              'last_entry_timestamp': _CURSOR_TIME.isoformat(),
              'last_entry_id': 'id-0',
          },
          f,
      )

    cache = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=cache_dir, read_overlap_seconds=100.0
    )

    # Nothing below the cursor may be re-read: its ids are unknown.
    self.assertEqual(cache.get_last_entry_info(), (_CURSOR_TIME, 'id-0'))
    self.assertEqual(cache.get_recent_entry_ids(), {})
    self.assertEqual(cache.get_read_overlap_seconds(), 0.0)

    # The overlap grows with the cursor as ids become known.
    later_time = _CURSOR_TIME + datetime.timedelta(seconds=40)
    cache.update_cached_entries(
        [_step_entry(1, later_time)],
        (later_time, 'id-1'),
        [('id-1', later_time)],
    )
    self.assertEqual(cache.get_read_overlap_seconds(), 40.0)

    latest_time = _CURSOR_TIME + datetime.timedelta(seconds=400)
    cache.update_cached_entries(
        [_step_entry(2, latest_time)],
        (latest_time, 'id-2'),
        [('id-2', latest_time)],
    )
    self.assertEqual(cache.get_read_overlap_seconds(), 100.0)
    self.assertEqual(cache.get_recent_entry_ids(), {'id-2': latest_time})

  def test_cached_entries_are_sorted_by_timestamp(self):
    cache = goodput_cache.GoodputCache(
        job_name=self.job_name, cache_dir=self.create_tempdir().full_path
    )
    base_time = _CURSOR_TIME.timestamp()
    entries = [
        {'step_count': 0, 'step_start_time': base_time},
        {'step_count': 2, 'step_start_time': base_time + 6},
        {'no_timestamp': 'keeps its position after step 2'},
        {'step_count': 1, 'step_start_time': base_time + 3},
        {
            'event_type': 'save',
            'step': 0,
            'checkpoint_manager_blocking_start_time': base_time + 1,
            'checkpoint_manager_blocking_duration_secs': 1.0,
        },
    ]
    cache.update_cached_entries(entries, (_CURSOR_TIME, 'id-4'), [])

    self.assertEqual(
        cache.get_cached_entries(),
        [entries[0], entries[4], entries[3], entries[1], entries[2]],
    )
    self.assertEqual(
        cache.get_step_entries(), [entries[0], entries[3], entries[1]]
    )

  def test_timeline_without_metadata_is_cleared(self):
    cache_dir = self.create_tempdir().full_path
    _write_timeline(cache_dir, self.job_name, [_step_entry(0, _CURSOR_TIME)])

    with self.assertLogs(level='WARNING') as logs:
      cache = goodput_cache.GoodputCache(
          job_name=self.job_name, cache_dir=cache_dir
      )

    self.assertTrue(cache.is_cache_empty())
    self.assertIsNone(cache.get_last_entry_info())
    self.assertIn('no read cursor', logs.output[0])

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


if __name__ == '__main__':
  googletest.main()
