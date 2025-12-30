"""Utility module for handling exclusion intervals in Goodput calculations."""

import datetime
import logging
from typing import List, Tuple


def normalize_intervals(
    intervals: List[Tuple[datetime.datetime, datetime.datetime]],
) -> List[Tuple[float, float]]:
  """Sorts, merges overlapping intervals, and converts to Unix timestamps.

  Args:
    intervals: A list of (start, end) datetime objects (UTC).

  Returns:
    A list of (start_timestamp, end_timestamp) floats. The list is sorted by
    start time and contains no overlapping intervals. Invalid intervals
    are filtered out.
  """
  if not intervals:
    return []

  valid_intervals = []
  for start, end in intervals:
    if (
        start.tzinfo != datetime.timezone.utc
        or end.tzinfo != datetime.timezone.utc
    ):
      logging.warning(
          "Skipping interval [%s , %s]: Timezone is not UTC",
          start,
          end,
      )
      continue

    if start >= end:
      logging.warning(
          "Skipping interval [%s , %s]: Start time is after or equal to End"
          " time.",
          start,
          end,
      )
      continue

    valid_intervals.append((start, end))

  if not valid_intervals:
    return []

  sorted_intervals = sorted(valid_intervals)
  merged_dt = [sorted_intervals[0]]
  for current in sorted_intervals[1:]:
    last_start, last_end = merged_dt[-1]
    current_start, current_end = current
    if current_start <= last_end:
      merged_dt[-1] = (last_start, max(last_end, current_end))
    else:
      merged_dt.append(current)

  return [(s.timestamp(), e.timestamp()) for s, e in merged_dt]


def get_exclusion_overlap(
    start: float, end: float, exclusions: List[Tuple[float, float]]
) -> float:
  """Calculates the total overlap duration between a range and exclusion intervals.

  Args:
    start: The start timestamp of the range.
    end: The end timestamp of the range.
    exclusions: A list of normalized (start, end) float timestamps.

  Returns:
    The total duration in seconds that the [start, end] range overlaps with
    any interval in the exclusions list.
  """
  total_overlap = 0.0
  for exclusion_start, exclusion_end in exclusions:
    overlap_start = max(start, exclusion_start)
    overlap_end = min(end, exclusion_end)

    if overlap_start < overlap_end:
      total_overlap += overlap_end - overlap_start

  return total_overlap
