# Goodput timeline and run comparison

The goodput timeline renders the goodput logs of a workload, as written by
`ml-goodput-measurement`, as an HTML page: Goodput, disruptions and the
breakdown of each recovery, on a timeline of the workload's runs. `compare`
puts several workloads side by side.

## Install

```
python -m venv goodput-timeline
source goodput-timeline/bin/activate
pip install "ml-goodput-measurement[timeline]"
```

The tool reads the logs from Cloud Logging with Application Default Credentials
(for example, after `gcloud auth application-default login`), so the account
needs read access to the logs of the workload's project.

The commands below run the tool with
`python -m ml_goodput_measurement.timeline`. It creates `outputs/compare` and
`outputs/timeline` in the current directory when they are missing, and caches
the logs that it reads from Cloud Logging in `outputs/cache` (see "Log cache"
at the end).

## Confirm Goodput logs exist for the workload and project of interest.

```
export PROJECT=my-gcp-project
python -m ml_goodput_measurement.timeline --command=list --project_id=$PROJECT
```

## Run export.

`--start_time` and `--end_time` are optional, but without a start time Cloud
Logging only returns the last 24 hours of logs. The logs are cached, so running
it again, e.g. for a narrower time range, does not read Cloud Logging again;
`--refresh_cache` reads them again, e.g. for a job that is still running.

```
export PROJECT=my-gcp-project
export WORKLOAD=resize-4x1k-2
export START=2026-01-23T15:45:06-08:00
export END=2026-01-26T06:25:46-08:00

python -m ml_goodput_measurement.timeline --command=export \
--project_id=$PROJECT \
--workload_id=$WORKLOAD \
--start_time=$START \
--end_time=$END \
--output_path="outputs/timeline/${PROJECT}_${WORKLOAD}_timeline.html"
```

### Or from a local JSON file of log entries.

`--entries_path` renders the timeline from a JSON list of goodput log payloads,
or from a cache file (see "Log cache"), without querying Cloud Logging. The
workload id defaults to the file name.

```
python -m ml_goodput_measurement.timeline --command=export \
--entries_path=outputs/cache/${WORKLOAD}.json \
--output_path=outputs/timeline/${WORKLOAD}_timeline.html
```

### What the timeline page shows.

*   **Workload Metrics & Recovery Summary**: Goodput (preserved productive
    time / total job time, from the ml-goodput-measurement library),
    disruptions and MTBI, mean MTTR and its breakdown into the 10 recovery
    sub-phases.
*   **Workload Overview**: a Gantt chart of all runs (Stepping, Startup /
    Recovery, Interruption) with the sub-events overlaid, and below it, on the
    same time axis:
    *   **Step time vs. baseline**: the time of every step while stepping. The
        dashed line is the steady-state step time (the lower median of all
        step times, also used by the JIT compile overhead detection and the
        comparison table). For elastic workloads, it only counts the steps
        with all slices active, and a step with fewer active slices is
        expected to take the lower median of the steps that ran with the same
        slice counts (dotted line). That time is measured, not scaled from the
        baseline, because fewer slices keep the step time in some workloads
        (fixed per-device batch) and lengthen it in others (fixed global
        batch); with fewer than 5 such steps it is unknown and the step is not
        marked. Steps slower than 1.15× their expected time are slow: a
        diamond when a checkpoint save ran during the step, a square when it
        has JIT compile overhead after a (re)start, and red otherwise. Steps
        above the chart range are drawn at the top as triangles; hover for the
        value.
    *   **Elastic slices**: the active, available and total slice counts that
        elastic Pathways workloads report (`elastic_slice_counts`); hover an
        interval for the counts and how long they held. McJAX and non-elastic
        workloads do not log slice counts and have no such panel.
*   **One section per run**: the run's recovery time breakdown and timeline.
    In the timeline, Stepping and Wasted Progress are containers: Productive
    Steps and Lost Progress are the parts of them outside the JIT compile
    overhead of the first steps after a (re)start, so the three never overlap.

## Compare runs.

`compare` puts several workloads side by side, for example the same model run
with different recovery strategies. It writes one HTML page with:

*   a comparison table, one row per workload: Goodput, incidents (MTBI), MTTR,
    Time to Recover, Lost Progress, each phase group's share of MTTR and the
    main bottleneck, plus the change vs. the baseline of the workload's group
    (see "Groups and baselines" below);
*   for each group of workloads, a two-panel chart: (A) mean recovery time per
    incident, stacked by phase group, and (B) each phase group's share of MTTR.
    Under each chart, collapsed tables break the recovery down by phase group
    and by all 10 recovery sub-phases.

The workloads, with their time ranges, models, scales and baselines, are listed
in a JSON config (`--compare_config`). `compare` does not use `--start_time`
and `--end_time`.

**We strongly recommend setting `model` and `shape` for every workload in the
config**: the table groups the workloads by model and scale. The tool also
tries to read them from the workload's JobSet (see "Model and scale" below),
but that may only work for MaxText workloads: it needs a JobSet named after
the workload, and the model comes from MaxText's `model_name` argument.
Without `model` and `shape`, other workloads can end up in the wrong group.

### Command examples.

Compare the workloads of a config. `--compare_csv_path` also writes the table
as a CSV file:

```
python -m ml_goodput_measurement.timeline \
  --command=compare \
  --compare_config=comparison_grouped.json \
  --output_path=outputs/compare/comparison.html \
  --compare_csv_path=outputs/compare/comparison.csv
```

When `--output_path` or `--compare_csv_path` is a directory, or `--output_path`
is not given, the file is named `comparison_<yyyymmddhhmm>.html` or `.csv`
after the current time, so every run keeps its output:

```
python -m ml_goodput_measurement.timeline \
  --command=compare \
  --compare_config=comparison_grouped.json \
  --output_path=outputs/compare/ \
  --compare_csv_path=outputs/compare/
```

Other flags:

*   `--nocompare_jobset_lookup`: no JobSet lookup, e.g. for workloads that are
    not MaxText JobSets; the config's `model` and `shape` are used as they are.
*   `--compare_parallel_fetches=1`: fetch the logs one workload at a time, e.g.
    after Cloud Logging quota errors.
*   `--compare_title="Recovery modes on TPU7x"`: replace the config's title.
*   `--refresh_cache`: read the logs and JobSets from Cloud Logging again, e.g.
    for jobs that were still running (see "Log cache").

### Config examples.

One model and scale. Without `"baseline": true`, the baseline is the first
workload with recovery incidents:

```
{
  "title": "Recovery modes: Llama-3.1-405B on 4x1k TPU7x",
  "project_id": "my-gcp-project",
  "workloads": [
    {"workload_id": "restart-4x1k-1", "label": "Full JobSet restart",
     "model": "Llama-3.1-405B", "shape": "4x1k",
     "start_time": "2026-01-21T18:00:00-08:00",
     "end_time": "2026-01-22T19:00:00-08:00"},
    {"workload_id": "resize-4x1k-1", "label": "Elastic Replica Resize",
     "model": "Llama-3.1-405B", "shape": "4x1k",
     "start_time": "2026-01-22T18:00:00-08:00",
     "end_time": "2026-01-23T19:00:00-08:00",
     "note": "Goodput and MTBI cover a shorter run."}
  ]
}
```

Several models and scales, each group with its own baseline. `groups` sets the
order of the groups and a description for each; a group name is the model and
the scale joined by " · ":

```
{
  "title": "Pathways elasticity vs. McJAX baselines on TPU7x",
  "project_id": "my-gcp-project",
  "groups": [
    {"name": "Llama-3.1-405B · 2x1k", "description": "TPU7x, 2,048 chips"},
    {"name": "Llama-3.1-8B · 2x64", "description": "TPU7x, 128 chips"}
  ],
  "workloads": [
    {"workload_id": "snapshot-2x1k-1", "label": "Elastic Snapshotting",
     "model": "Llama-3.1-405B", "shape": "2x1k",
     "start_time": "2026-01-17T19:00:00-08:00",
     "end_time": "2026-01-18T20:00:00-08:00"},
    {"workload_id": "mcjax-2x1k-1", "label": "McJAX",
     "model": "Llama-3.1-405B", "shape": "2x1k", "baseline": true,
     "start_time": "2026-01-20T12:00:00-08:00",
     "end_time": "2026-01-20T20:30:00-08:00"},
    {"workload_id": "resize-2x64-1", "label": "Elastic Replica Resize",
     "model": "Llama-3.1-8B", "shape": "2x64",
     "start_time": "2026-01-19T08:00:00-08:00",
     "end_time": "2026-01-19T20:00:00-08:00"},
    {"workload_id": "mcjax-2x64-1", "label": "McJAX",
     "model": "Llama-3.1-8B", "shape": "2x64", "baseline": true,
     "start_time": "2026-01-19T12:00:00-08:00",
     "end_time": "2026-01-20T09:00:00-08:00"}
  ]
}
```

Logs from files instead of Cloud Logging: `entries_path` (JSON list of log
payloads, e.g. a copy of a cache file; a relative path is relative to the
config file). The cache itself does not need it (see "Log cache"). A file is
used as it is: its entries are not filtered by the time range, and its
workload is not looked up, so set its `model` and `shape`. Files in
`outputs/cache` are replaced when the logs are read again, so copy the ones to
keep, e.g. to `saved_logs/`:

```
{
  "title": "Recovery modes from saved logs",
  "workloads": [
    {"workload_id": "snapshot-2x1k-1", "label": "Elastic Snapshotting",
     "model": "Llama-3.1-405B", "shape": "2x1k",
     "entries_path": "saved_logs/snapshot-2x1k-1.json"},
    {"workload_id": "snapshot-2x1k-2", "label": "Elastic Snapshotting",
     "model": "Llama-3.1-405B", "shape": "2x1k",
     "entries_path": "saved_logs/snapshot-2x1k-2.json"}
  ]
}
```

### Config keys.

Workload keys:

*   `workload_id` (required). A workload can also be given as just its id.
*   `label`: the recovery mode, shown in the table.
*   `model` and `shape`: the model and scale, e.g. "Llama-3.1-405B" and "2x1k".
    They decide the group of the workload.
*   `start_time` and `end_time`, e.g. "2026-01-21T18:00:00-08:00". Always set
    a start time: without one, Cloud Logging only returns the last 24 hours of
    logs. A narrow time range per workload is also much faster.
*   `baseline`: true for the baseline of its group (see "Groups and baselines"
    below).
*   `note` (footnote) and `timeline_url` (link of the workload id in the table,
    e.g. to its timeline page).
*   `project_id`, if it is not the top-level one or `--project_id`.
*   `entries_path`: read the logs from a file, see above.
*   `group`: a group name to use instead of the model and scale, e.g. to put
    two scales in one group. Not needed otherwise.

Top-level keys: `workloads`, `title`, `subtitle`, `description` (paragraph
above the table), `groups` (see "Groups and baselines" below), and
`project_id`, `start_time`, `end_time`, `model` and `group` as defaults for all
workloads. Misspelled keys are reported as errors.

The logs of up to 4 workloads are fetched at the same time
(`--compare_parallel_fetches`; use 1 to fetch them one by one), so a comparison
takes about as long as its slowest workload. A read that hits the Cloud Logging
quota is retried a few times, from the page where it stopped. A workload whose
logs cannot be fetched (for example, the query timed out) or that has no logs
in its time range is skipped with a message, and the other workloads are still
compared.

### Model and scale.

The Model column shows the config's `model`. Unless `--nocompare_jobset_lookup`
is set, `compare` also looks up the GKE audit log entry that created the
workload's JobSet (`<workload_id>` or `pw-<workload_id>`, up to 2 days before
its first goodput event) and reads from it:

*   Scale: slices × chips per slice, e.g. 2x1k. Slices are the `replicas` of
    the TPU jobs, not the active slices in the goodput log, which can be fewer.
    Chips per slice are pods (`parallelism`) × `google.com/tpu` chips per pod.
    The JobSet's scale is used over `shape` and the workload id.
*   Model: MaxText's `model_name=...` in the container args, e.g.
    llama3.1-405b. It is only shown if the config has no `model`.

Where the config or the workload id disagrees with the JobSet, a warning is
printed. The lookup takes about 30 seconds per workload the first time; the
result is cached with the logs (see "Log cache"). Workloads read from
`entries_path` are not looked up.

If the JobSet is not found (for example, the workload does not run as a JobSet,
or its JobSet has another name or was created more than 2 days before the first
goodput event), a message is printed and the workload is still compared. Its
model is then the config's `model` and its scale is `shape`, else parsed from
the workload id (e.g. 4x1k). Unknown ones are shown as "—", and the group only
has the known parts, e.g. "Llama-3.1-405B" without a scale. The cache also
remembers a JobSet that was not found; `--refresh_cache` looks it up again.

### Groups and baselines.

The table has a Model column and groups its rows by model and scale, e.g.
"Llama-3.1-405B · 2×1k" (a workload's `group` replaces this). Each group starts
with a row that shows its name, its description from `groups`, the scale range
of its workloads (unless the name already has it) and their step time range.
`groups` also sets the order of the groups (names match regardless of case,
spaces and x vs. ×); groups that it does not list follow in config order, and
workloads with neither a model nor a scale come last under "Other".

Each group has one baseline, and different groups can have different ones:

*   `"baseline": true` on one workload sets it. More than one in a group is
    an error.
*   Otherwise the baseline is the group's first workload with recovery
    incidents, else its first workload.
*   A `"baseline": true` workload whose logs cannot be loaded is skipped, and
    its group uses the default above. One without Goodput or recovery
    incidents stays the baseline; the values that it lacks are not compared,
    and a note at the top of its group says so.

The other workloads of the group show their change vs. the baseline: Goodput in
percentage points, and MTTR, Time to Recover and Lost Progress in percent. A
recovery time at least 5% lower than the baseline's is green and one at least
5% higher is red; the Mean MTTR cell and the "N× faster/slower MTTR" text follow
the same rule. The baseline itself is not colored. Goodput is green from 85%,
yellow from 60% and red below.

A workload whose goodput log has no training step events (for example, only
checkpoint events) shows Goodput as n/a, with a footnote.

## Log cache.

`export` and `compare` read the goodput logs of a workload from Cloud Logging
once and cache them in `outputs/cache` in the current directory
(`--entries_cache_dir`; `--entries_cache_dir=` turns the cache off), in
`<workload_id>.json`, and its JobSet in `<workload_id>.jobset.json`. Both
commands use the same files. A later run reads a cache file instead of Cloud
Logging when the requested time range is within the cached one (a request
without a start or end time is within any), and only keeps the entries in the
requested range; otherwise it reads the logs again and replaces the file.
Workloads without logs in their time range are cached too. Cache files written
by earlier versions of the tool are only used for the same time range.

The cache is not refreshed automatically: logs that a running job wrote after
its logs were read are not in the cache, and a note says so when such a cache
is used. `--refresh_cache` reads the logs and JobSets again and updates the
cache.

Logs are read 1,000 entries per request, and Cloud Logging allows 60 read
requests per minute per project; a read that hits this quota continues from the
page where it stopped. The entries that the Cloud Logging client library adds
to the goodput logs (`logging.googleapis.com/diagnostic`) are not read.
