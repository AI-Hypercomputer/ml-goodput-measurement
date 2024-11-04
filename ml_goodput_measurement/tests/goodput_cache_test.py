"""Tests to unit test GoodputCache class."""

import datetime
from unittest import mock

from cloud_goodput.ml_goodput_measurement.src import goodput_cache
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src.goodput_utils import BadputType, GoodputInfo

from google3.testing.pybase import googletest


class GoodputCacheTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.goodput_cache = goodput_cache.GoodputCache()

  def test_update_cached_entries(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries)
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(self.goodput_cache._cached_entries, mock_entries)

  def test_update_goodput_info(self):
    goodput_info = GoodputInfo(
        total_productive_time=100,
        total_elapsed_time_since_start=200,
        total_unproductive_time={
            BadputType.TPU_INITIALIZATION: 10,
            BadputType.TRAINING_PREP: 10,
            BadputType.DATA_LOADING: 30,
            BadputType.PROGRAM_STARTUP: 10,
            BadputType.UNPRODUCTIVE_CHECKPOINTING: 20,
            BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
            BadputType.OTHER: 10,
        },
        last_recorded_step=3,
    )
    self.goodput_cache.update_goodput_info(goodput_info)
    self.assertEqual(self.goodput_cache._goodput_info, goodput_info)

  def test_clear_cache(self):
    mock_entries = [
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ]
    self.goodput_cache.update_cached_entries(mock_entries)
    self.goodput_cache.update_goodput_info(
        GoodputInfo(
            total_productive_time=100,
            total_elapsed_time_since_start=200,
            total_unproductive_time={
                BadputType.TPU_INITIALIZATION: 10,
                BadputType.TRAINING_PREP: 10,
                BadputType.DATA_LOADING: 30,
                BadputType.PROGRAM_STARTUP: 10,
                BadputType.UNPRODUCTIVE_CHECKPOINTING: 20,
                BadputType.WASTED_PROGRESS_FROM_DISRUPTION: 10,
                BadputType.OTHER: 10,
            },
            last_recorded_step=3,
        )
    )
    self.goodput_cache.clear_cache()
    self.assertEqual(self.goodput_cache._cached_entries, [])
    self.assertIsNone(self.goodput_cache._goodput_info)
    self.assertIsNone(self.goodput_cache._last_entry_timestamp)

  def test_is_cache_empty(self):
    self.assertTrue(self.goodput_cache.is_cache_empty())
    self.goodput_cache.update_cached_entries([
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ])
    self.assertFalse(self.goodput_cache.is_cache_empty())

  def test_get_last_entry_timestamp(self):
    self.assertIsNone(self.goodput_cache._last_entry_timestamp)
    self.goodput_cache.update_cached_entries([
        {'time': 1, 'step': 1},
        {'time': 2, 'step': 2},
        {'time': 3, 'step': 3},
    ])
    self.assertFalse(self.goodput_cache.is_cache_empty())
    self.assertEqual(
        self.goodput_cache._last_entry_timestamp,
        datetime.datetime.fromtimestamp(3),
    )


if __name__ == '__main__':
  googletest.main()
