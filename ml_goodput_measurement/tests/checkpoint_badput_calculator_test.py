"""Tests for checkpoint badput calculator."""

import dataclasses
from typing import Optional

from absl.testing import absltest
from cloud_tpu_goodput.ml_goodput_measurement.src import checkpoint_badput_calculator
import google.cloud.logging as google_cloud_logging
import mock


_JOB_NAME = 'checkpoint_job'
_LOGGER_NAME = 'checkpoint_logger'


@dataclasses.dataclass
class MockStepStatistics:
  """Attributes.

  Attributes:
    step: The step number.
    event_type: The event type.
    start_time: The start time of the event.
    end_time: The end time of the event.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    wait_for_prev_end_time: The end time of waiting for previous checkpoint.
    checkpoint_start_time: The start time of checkpointing.
    checkpoint_end_time: The end time of checkpointing.
    get_old_steps_start_time: The start time of getting old steps.
    get_old_steps_end_time: The end time of getting old steps.
    synchronous: Whether the event is synchronous.
  """

  step: Optional[int] = None
  event_type: Optional[str] = None
  start_time: Optional[float] = None
  end_time: Optional[float] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_end_time: Optional[float] = None
  checkpoint_start_time: Optional[float] = None
  checkpoint_end_time: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_end_time: Optional[float] = None
  synchronous: Optional[bool] = False


class CheckpointBadputCalculatorTest(absltest.TestCase):

  def setUp(self):
    """Setup for the test."""
    super().setUp()
    mock_gcloud_client = mock.create_autospec(google_cloud_logging.Client)
    options = checkpoint_badput_calculator.CheckpointLoggerOptions(
        job_name=_JOB_NAME,
        logger_name=_LOGGER_NAME,
        client=mock_gcloud_client,
    )
    self.checkpoint_badput_calculator = (
        checkpoint_badput_calculator.CheckpointBadputCalculator(options)
    )
    for i in range(1, 5):
      self.checkpoint_badput_calculator.entries.append(
          dataclasses.asdict(
              MockStepStatistics(
                  step=i,
                  event_type='save',
                  start_time=i * 10.0,
                  wait_for_prev_start_time=i * 10.0,
                  wait_for_prev_end_time=i * 10.0 + 2,
                  checkpoint_start_time=i * 10.0 + 2,
                  checkpoint_end_time=i * 10.0 + 3,
                  end_time=i * 10.0 + 4,
                  reached_preemption=False,
                  preemption_received_at=None,
                  synchronous=False,
              )
          )
      )

  def test_checkpoint_badput_calculator(self):
    total_time = (
        self.checkpoint_badput_calculator.calculate_blocking_checkpoint_time()
    )
    expected_total_time = len(self.checkpoint_badput_calculator.entries)
    self.assertEqual(total_time, expected_total_time)

  def test_checkpoint_badput_calculator_preemption(self):
    i = len(self.checkpoint_badput_calculator.entries) + 1
    preemption_entry = dataclasses.asdict(
        MockStepStatistics(
            step=i,
            event_type='save',
            start_time=i * 10.0,
            wait_for_prev_start_time=i * 10.0,
            wait_for_prev_end_time=i * 10.0 + 2,
            checkpoint_start_time=i * 10.0 + 2,
            checkpoint_end_time=i * 10.0 + 3,
            end_time=i * 10.0 + 4,
            reached_preemption=True,
            preemption_received_at=i * 10.0,
            synchronous=True,
        )
    )
    self.checkpoint_badput_calculator.entries.append(preemption_entry)
    expected_total_time = (i - 1) + (
        preemption_entry['wait_for_prev_end_time']
        - preemption_entry['start_time']
    )
    total_time = (
        self.checkpoint_badput_calculator.calculate_blocking_checkpoint_time()
    )
    print(total_time, expected_total_time)
    self.assertEqual(total_time, expected_total_time)


if __name__ == '__main__':
  absltest.main()
