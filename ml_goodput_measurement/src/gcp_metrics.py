# gcp_metrics.py

import logging
import time
from typing import Dict
from enum import Enum
from google.cloud import monitoring_v3
from google.api_core.exceptions import GoogleAPIError


class ValueType(Enum):
    """Enum for metric value types."""

    BOOL = "bool_value"
    INT = "int64_value"
    DOUBLE = "double_value"
    STRING = "string_value"
    DISTRIBUTION = "distribution_value"  # Add other types as needed


class GCPMetrics:
    """A generic class to send multiple metrics to GCP Cloud Monitoring in a batch with dynamic resources.
    """

    def __init__(self, project_id: str):
        """Initializes the GCPMetrics.
        """
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"

    def create_time_series(
        self,
        metric_type: str,
        value,
        value_type: ValueType,
        metric_labels: Dict[str, str],
        resource_type: str,
        resource_labels: Dict[str, str],
        seconds: int,
        nanos: int,
    ) -> monitoring_v3.TimeSeries:
        """Creates a TimeSeries object for a single metric with dynamic resources.
        """
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        series.resource.type = resource_type
        series.resource.labels.update(resource_labels)
        series.metric.labels.update(metric_labels)

        point = monitoring_v3.Point(
            interval=monitoring_v3.TimeInterval(
                end_time={"seconds": seconds, "nanos": nanos}
            ),
            value=monitoring_v3.TypedValue(**{value_type.value: value}),
        )
        series.points.append(point)

        return series

    def send_metrics(self, metrics: list[Dict[str, any]]):
        """
        Sends multiple metrics to GCP Monitoring in a batch with dynamic
        resources.

        Args:
            metrics: A list of dictionaries, where each dictionary represents
                     a metric. Each dictionary should have the following keys:
                     - 'metric_type': str
                     - 'value': The metric value.
                     - 'value_type': ValueType (e.g., ValueType.INT,
                       ValueType.DOUBLE)
                     - 'metric_labels': dict (optional)
                     - 'resource_type': str
                     - 'resource_labels': dict
        """
        try:
          now = time.time()
          seconds = int(now)
          nanos = int((now - seconds) * 10**9)

          time_series_list = []
          for metric in metrics:
              metric_labels = metric.get("metric_labels", {})
              series = self.create_time_series(
                  metric["metric_type"],
                  metric["value"],
                  metric["value_type"],
                  metric_labels,
                  metric["resource_type"],
                  metric["resource_labels"],
                  seconds,
                  nanos,
              )
              time_series_list.append(series)

          self.client.create_time_series(
              name=self.project_name, time_series=time_series_list
          )
          logging.info("Sent %d metrics to GCP Monitoring.", len(metrics))

        except GoogleAPIError as e:
          logging.error("Failed to send metrics: %s", e)


# Example Usage
if __name__ == "__main__":
    project_id = "GCP-PROJECT-ID"  # Replace with your project ID.

    metrics_sender = GCPMetrics(project_id)

    metrics_to_send = [
        {
            "metric_type": "compute.googleapis.com/workload/goodput_time",
            "value": 123.45,
            "value_type": ValueType.DOUBLE,
            "metric_labels": {"goodput_source": "TOTAL", "accelerator_type": "tpu-v5p"},
            "resource_type": "compute.googleapis.com/Workload",
            "resource_labels": {
                "location": "us-central1",
                "workload_id": "test-workload",
                "replica_id": "0",
            },
        },
        {
            "metric_type": "compute.googleapis.com/workload/badput_time",
            "value": 23.00,
            "value_type": ValueType.DOUBLE,
            "metric_labels": {
                "badput_source": "TPU_INITIALIZATION",
                "accelerator_type": "tpu-v5p",
            },
            "resource_type": "compute.googleapis.com/Workload",
            "resource_labels": {
                "location": "us-central1",
                "workload_id": "test-workload",
                "replica_id": "0",
            },
        },
        {
            "metric_type": "compute.googleapis.com/workload/badput_time",
            "value": 12.23,
            "value_type": ValueType.DOUBLE,
            "metric_labels": {
                "badput_source": "TRAINING_PREP",
                "accelerator_type": "tpu-v5p",
            },
            "resource_type": "compute.googleapis.com/Workload",
            "resource_labels": {
                "location": "us-central1",
                "workload_id": "test-workload",
                "replica_id": "0",
            },
        },
    ]

    metrics_sender.send_metrics(metrics_to_send)
