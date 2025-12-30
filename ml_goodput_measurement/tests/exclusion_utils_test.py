"""Tests for exclusion_utils."""

import datetime
from typing import Tuple

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import exclusion_utils

TimeRange = Tuple[datetime.datetime, datetime.datetime]


class ExclusionUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.anchor_time = datetime.datetime(
        2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

  def _make_interval(
      self, start_offset_sec: int, end_offset_sec: int
  ) -> TimeRange:
    """Helper to create datetime intervals relative to test anchor time."""
    return (
        self.anchor_time + datetime.timedelta(seconds=start_offset_sec),
        self.anchor_time + datetime.timedelta(seconds=end_offset_sec),
    )

  def test_normalize_intervals_basic_merge(self):
    # Case: Two overlapping intervals.
    intervals = [
        self._make_interval(0, 30),
        self._make_interval(15, 45),
    ]

    normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 1)
    self.assertEqual(normalized[0][0], self.anchor_time.timestamp())
    self.assertEqual(
        normalized[0][1],
        (self.anchor_time + datetime.timedelta(seconds=45)).timestamp(),
    )

  def test_normalize_intervals_nested(self):
    # Case: One interval completely inside another.
    intervals = [
        self._make_interval(0, 60),
        self._make_interval(10, 20),
    ]

    normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 1)
    self.assertEqual(
        normalized[0][1],
        (self.anchor_time + datetime.timedelta(seconds=60)).timestamp(),
    )

  def test_normalize_intervals_touching(self):
    # Case: Intervals that touch boundaries.
    intervals = [
        self._make_interval(0, 10),
        self._make_interval(10, 20),
    ]

    normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 1)
    self.assertEqual(
        normalized[0][1],
        (self.anchor_time + datetime.timedelta(seconds=20)).timestamp(),
    )

  def test_normalize_intervals_disjoint(self):
    # Case: Disjoint intervals.
    intervals = [
        self._make_interval(0, 10),
        self._make_interval(20, 30),
    ]

    normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 2)
    self.assertEqual(normalized[0][1], self.anchor_time.timestamp() + 10.0)
    self.assertEqual(normalized[1][0], self.anchor_time.timestamp() + 20.0)

  def test_normalize_intervals_unordered(self):
    # Case: Intervals out of order.
    intervals = [
        self._make_interval(20, 30),
        self._make_interval(0, 10),
    ]

    normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 2)
    self.assertEqual(normalized[0][0], self.anchor_time.timestamp())

  def test_normalize_intervals_invalid(self):
    # Case: Mixed invalid scenarios.
    naive_dt = datetime.datetime(2025, 1, 1, 12, 0, 0)
    intervals = [
        (
            self.anchor_time + datetime.timedelta(seconds=10),
            self.anchor_time,
        ),  # End < Start.
        (self.anchor_time, self.anchor_time),  # Invalid: Zero length.
        (
            naive_dt,
            naive_dt + datetime.timedelta(seconds=10),
        ),  # Non-UTC.
        self._make_interval(0, 5),  # 0s to 5s.
    ]

    with self.assertLogs(level='WARNING'):
      normalized = exclusion_utils.normalize_intervals(intervals)

    self.assertLen(normalized, 1)
    self.assertEqual(normalized[0][1], self.anchor_time.timestamp() + 5.0)

  def test_normalize_intervals_empty(self):
    self.assertEmpty(exclusion_utils.normalize_intervals([]))

  def test_normalize_intervals_all_invalid(self):
    # Case: No intervals are valid.
    intervals = [
        (self.anchor_time, self.anchor_time),
        (
            self.anchor_time,
            self.anchor_time - datetime.timedelta(seconds=1),
        ),
    ]

    with self.assertLogs(level='WARNING'):
      normalized = exclusion_utils.normalize_intervals(intervals)
    self.assertEmpty(normalized)

  def test_get_exclusion_overlap_full_coverage(self):
    # Range: 100-200, Exclusion: 50-250.
    exclusions = [(50.0, 250.0)]

    overlap = exclusion_utils.get_exclusion_overlap(100.0, 200.0, exclusions)
    self.assertEqual(overlap, 100.0)

  def test_get_exclusion_overlap_partial_start(self):
    # Range: 100-200, Exclusion: 50-150.
    exclusions = [(50.0, 150.0)]

    overlap = exclusion_utils.get_exclusion_overlap(100.0, 200.0, exclusions)
    self.assertEqual(overlap, 50.0)

  def test_get_exclusion_overlap_partial_end(self):
    # Range: 100-200, Exclusion: 150-250.
    exclusions = [(150.0, 250.0)]

    overlap = exclusion_utils.get_exclusion_overlap(100.0, 200.0, exclusions)
    self.assertEqual(overlap, 50.0)

  def test_get_exclusion_overlap_multiple_disjoint(self):
    # Range: 0-100, Exclusions: [10-20] and [80-90].
    exclusions = [(10.0, 20.0), (80.0, 90.0)]

    overlap = exclusion_utils.get_exclusion_overlap(0.0, 100.0, exclusions)
    self.assertEqual(overlap, 20.0)  # 10s + 10s

  def test_get_exclusion_overlap_no_overlap(self):
    # Range: 0-100, Exclusion: 200-300.
    exclusions = [(200.0, 300.0)]

    overlap = exclusion_utils.get_exclusion_overlap(0.0, 100.0, exclusions)
    self.assertEqual(overlap, 0.0)

  def test_get_exclusion_overlap_float_precision(self):
    # Test with sub-second timestamps.
    start = 100.0
    end = 100.1
    exclusions = [(100.05, 200.0)]

    overlap = exclusion_utils.get_exclusion_overlap(start, end, exclusions)
    self.assertAlmostEqual(overlap, 0.05)


if __name__ == '__main__':
  absltest.main()
