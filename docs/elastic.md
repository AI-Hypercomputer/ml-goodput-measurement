# Elastic Goodput

For elastic training workloads (which support autoscaling and dynamic slice
adjustments), the library provides elastic-aware components under
`ml_goodput_measurement.goodput_elastic` and
`ml_goodput_measurement.monitoring_elastic`.

## New Badput Buckets & Slice Efficiency

By adopting the Elastic APIs, the following metrics will be automatically
calculated and uploaded to your dashboards:

*   **Elastic Badput Breakdown**: Separates dynamic node provisioning
    disruptions into distinct categories:
    *   `ELASTIC_SCALE_UP`: Measures the time spent waiting for new slices to
        rejoin the workload (from scale-up detection until slices are fully
        provisioned). For Replica Resize, this captures the scale-up idle wait
        time prior to the next checkpoint.
    *   `ELASTIC_SLICE_DOWN`: Measures the wait time for replacement slices
        during a disruption. For Pause Resume mode, this captures the full
        "slice availability wait time" when active stepping is blocked.
    *   `ELASTIC_REINITIALIZATION`: Measures the time spent on JAX compilation
        and network state resharding/initialization once slices are available
        and a restart/resize is triggered.
*   **Slice Efficiency**: Monitors `stepping_slice_efficiency` and
    `available_slice_efficiency` to track in_use_capacity / required_capacity.

## Elastic Components

*   `ElasticGoodputRecorder`: Inherits from `GoodputRecorder`. Exposes new APIs
    like `record_elastic_slice_counts`, `record_elastic_wait_start_time`, and
    `record_elastic_reinit_start_time`.
*   `ElasticGoodputCalculator`: Integrates the collected elastic metrics
    seamlessly into the broader Goodput calculations.
*   `ElasticGoodputMonitor`: Upgraded monitor that delegates to
    `ElasticGoodputCalculator` internally.

## Instrumenting Elastic Goodput

Using Pathways, you can dynamically poll your environment to identify your slice
counts and properly inject them into your `ElasticGoodputRecorder`.

### 1. Instantiating the Elastic Monitor

Swap the traditional `GoodputMonitor` for the `ElasticGoodputMonitor`, making
sure to pass `include_slice_efficiency=True`:

```python
from ml_goodput_measurement import monitoring_elastic

gcp_options = monitoring_elastic.GCPOptions(enable_gcp_goodput_metrics=True)

goodput_monitor = monitoring_elastic.ElasticGoodputMonitor(
    job_name=job_name,
    logger_name=logger_name,
    tensorboard_dir=tensorboard_dir,
    include_badput_breakdown=True,
    include_slice_efficiency=True,
    gcp_options=gcp_options,
)
goodput_monitor.start_goodput_uploader()
```

### 2. Instantiating the Elastic Recorder & Recording Slice States

Before tracking training steps, create an `ElasticGoodputRecorder` only on the
lead host (rank 0). When your workflow undergoes an elastic event (e.g., node
loss or scaling up), you can fetch the current active/total slices directly
using the Pathways properties:

```python
from ml_goodput_measurement import goodput_elastic
import pathwaysutils

recorder = goodput_elastic.ElasticGoodputRecorder(job_name, logger_name, is_lead_host)

def record_slice_state(recorder, active_slices_override=None):
    if not recorder or not pathwaysutils.is_pathways_backend_used():
        return

    available_slices = len(pathwaysutils.elastic.get_active_slice_indices())
    active_slices = active_slices_override if active_slices_override is not None else len(elastic_manager.active_slice_indices)
    total_slices = len(pathwaysutils.elastic.get_slice_to_devices(jax.devices()))

    recorder.record_elastic_slice_counts(
        available_slices=available_slices,
        active_slices=active_slices,
        total_slices=total_slices,
    )

# Example: When a scale-up or slice-down is triggered by your coordinator,
# track the incoming wait time and push a 0-active bounding state to the recorder:
def record_elastic_scale():
    recorder.record_elastic_wait_start_time(event_type="elastic_scale_up")
    record_slice_state(recorder, active_slices_override=0)
```

See also: [Monitoring](monitoring.md), [Badput types](badput.md).
