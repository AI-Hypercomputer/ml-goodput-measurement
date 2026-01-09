"""Goodput Post-Processing Analysis Script.

Instructions

Environment Setup:
  - Ensure Python 3.9+ is installed.
  - Install the Goodput library:
    $ pip install ml-goodput-measurement>=0.0.16

Usage:
  $ python exclusion_query.py

Exclusion JSON Format (UTC):
  [
    {
      "start_time_utc": "2025-09-24T18:00:00Z",
      "end_time_utc": "2025-09-24T18:30:00Z"
    }
  ]
"""

import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from ml_goodput_measurement import goodput


def get_project_id() -> Optional[str]:
  """Retrieves the project ID from the environment or gcloud config."""
  if os.environ.get("PROJECT_ID"):
    return os.environ.get("PROJECT_ID")

  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
      project = result.stdout.strip()
      if project and project != "(unset)":
        return project
  except FileNotFoundError:
    pass

  return None


def load_intervals_from_json(
    file_path: str,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
  """Loads exclusion intervals from a JSON file and returns UTC datetimes."""
  intervals = []
  try:
    with open(file_path, "r") as f:
      data = json.load(f)

    for entry in data:
      start_str = entry["start_time_utc"].replace("Z", "+00:00")
      end_str = entry["end_time_utc"].replace("Z", "+00:00")

      try:
        start_dt = datetime.datetime.fromisoformat(start_str)
        end_dt = datetime.datetime.fromisoformat(end_str)
      except ValueError as error:
        print(f"Skipping malformed date string in {entry}: {error}")
        continue

      if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
      if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)

      if end_dt <= start_dt:
        print(f"Skipping invalid interval [{start_str}, {end_str}]")
        continue
      intervals.append((start_dt, end_dt))

    return intervals
  except Exception as error:  # pylint: disable=broad-except
    print(f"Error loading JSON file: {error}")
    sys.exit(1)


def analyze_run(
    project_id: str,
    run_name: str,
    exclusions: List[Tuple[datetime.datetime, datetime.datetime]],
):
  """Analyzes a single run with optional exclusion intervals."""

  os.environ["PROJECT_ID"] = project_id
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

  logger_name = f"goodput_{run_name}"

  print(f"Analyzing Run: {run_name} in project {project_id}")

  if exclusions:
    exclusions.sort(key=lambda x: x[0])

  try:
    approx_start_time = datetime.timedelta(days=5)
    calculator = goodput.GoodputCalculator(
        job_name=run_name,
        logger_name=logger_name,
        max_logs_retention_period=approx_start_time,
    )

    if exclusions:
      gp, breakdown, step, excl_sec = (
          calculator.get_job_goodput_with_exclusions(
              exclusion_intervals=exclusions, include_badput_breakdown=True
          )
      )
      print_report(run_name, gp, breakdown, step, excl_sec)
    else:
      gp, breakdown, step = calculator.get_job_goodput(
          include_badput_breakdown=True
      )
      print_report(run_name, gp, breakdown, step)

  except Exception as error:  # pylint: disable=broad-except
    print(f"\n[Analysis Error] {error}")
    import traceback  # pylint: disable=g-import-not-at-top
    print("Stack Trace:")
    traceback.print_exc()


def print_report(
    run_name: str,
    gp: float,
    breakdown: Dict[Any, Any],
    step: int,
    excl_sec: Optional[float] = None,
):
  """Prints a summary of the analysis."""
  print(f"Run Name: {run_name}")
  print(f"Maximum Productive Step: {step}")
  print(f"Goodput: {gp:.2f}%")

  if excl_sec is not None:
    print(f"Total Time Excluded: {excl_sec:.2f} seconds")

  print("\nBadput Breakdown:")

  categories = [
      (goodput.BadputType.TPU_INITIALIZATION, "Device Initialization"),
      (goodput.BadputType.TRAINING_PREP, "Training Preparation"),
      (goodput.BadputType.PROGRAM_STARTUP, "Program Startup"),
      (goodput.BadputType.DATA_LOADING_SYNC, "Data Loading"),
      (
          goodput.BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
          "Disruption: Wasted Progress",
      ),
      (
          goodput.BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION,
          "Disruption: Infra Recovery",
      ),
      (goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME, "Checkpoint Save"),
      (
          goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME,
          "Checkpoint Restore",
      ),
      (goodput.BadputType.OTHER, "Other/Unknown"),
  ]

  for badput_type, label in categories:
    val = breakdown.get(badput_type, 0)
    if isinstance(val, dict):
      val = sum(val.values())

    if isinstance(val, (float, int)) and val > 0:
      print(f"  {label:<20}: {val:.2f}%")


def main():
  print("ML Goodput Post-Processing Analysis Script")

  default_project = get_project_id()

  if default_project:
    user_input = input(f"Enter GCP Project ID [{default_project}]: ").strip()
    project_id = user_input if user_input else default_project
  else:
    while True:
      project_id = input("Enter GCP Project ID: ").strip()
      if project_id:
        break

  while True:
    run_name = input("Enter Run Name: ").strip()
    if run_name:
      break

  while True:
    json_path = input(
        "Enter path to intervals JSON file (Enter to skip): "
    ).strip()
    if not json_path:
      exclusions = []
      break

    if os.path.exists(json_path):
      exclusions = load_intervals_from_json(json_path)
      break
    else:
      print(f"Error: File '{json_path}' not found.")

  analyze_run(project_id, run_name, exclusions)


if __name__ == "__main__":
  main()
