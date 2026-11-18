import argparse
import datetime
import os
import sys
from typing import Optional

from absl import app
from absl import flags
from cloud_goodput.ml_goodput_measurement.timeline import builder
from cloud_goodput.ml_goodput_measurement.timeline import compare
from cloud_goodput.ml_goodput_measurement.timeline import event_parser
from cloud_goodput.ml_goodput_measurement.timeline import log_fetcher
from cloud_goodput.ml_goodput_measurement.timeline import renderer

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'command',
    'export',
    ['export', 'list', 'compare'],
    'Command to run: export, list or compare.',
)
flags.DEFINE_string(
    'project_id',
    None,
    'GCP project ID. Defaults to the GOOGLE_CLOUD_PROJECT environment'
    ' variable.',
)
flags.DEFINE_string('workload_id', None, 'Workload ID or run name.')
flags.DEFINE_string('start_time', None, 'ISO 8601 UTC start time.')
flags.DEFINE_string('end_time', None, 'ISO 8601 UTC end time.')
flags.DEFINE_string(
    'entries_path',
    None,
    'export: optional JSON file with the goodput log payloads (a list, or a'
    ' file from --entries_cache_dir) to render instead of fetching them from'
    ' Cloud Logging.',
)
flags.DEFINE_string(
    'output_path',
    'goodput_timeline.html',
    'Path to output HTML. compare: a directory, or no value, writes'
    ' comparison_<yyyymmddhhmm>.html in it, or in the current directory.',
)
flags.DEFINE_string(
    'compare_config',
    None,
    'compare: JSON file that lists the workloads to compare and their time'
    ' ranges (see README).',
)
flags.DEFINE_string(
    'compare_title', None, 'compare: chart title; overrides the config title.'
)
flags.DEFINE_string(
    'compare_csv_path',
    None,
    'compare: optional CSV output with one row per workload; a directory'
    ' writes comparison_<yyyymmddhhmm>.csv in it.',
)
flags.DEFINE_string(
    'entries_cache_dir',
    os.path.join('outputs', 'cache'),
    'Directory where export and compare cache the logs and JobSets read from'
    ' Cloud Logging, one file per workload, and reuse them while the cached'
    ' time range covers the requested one. A relative path is relative to the'
    ' current directory. An empty value (--entries_cache_dir=) turns the cache'
    ' off.',
)
flags.DEFINE_bool(
    'refresh_cache',
    False,
    'Read the logs and JobSets from Cloud Logging even if they are cached,'
    ' e.g. for a job that was still running, and update the cache.',
)
flags.DEFINE_integer(
    'compare_parallel_fetches',
    4,
    'compare: number of workloads whose logs are fetched at the same time.'
    ' Use 1 to fetch them one by one.',
    lower_bound=1,
)
flags.DEFINE_bool(
    'compare_jobset_lookup',
    True,
    'compare: read the model, slice count and chips per slice of each'
    ' workload from the GKE audit log entry that created its JobSet. Takes'
    ' about 30s per workload the first time; the result is cached.',
)


def _compare() -> None:
  """Summarizes several workloads and writes the comparison page."""
  if not FLAGS.compare_config:
    raise app.UsageError('--compare_config must be specified for compare.')
  for name in ('start_time', 'end_time'):
    if FLAGS[name].present:
      raise app.UsageError(
          f'--{name} is not used by compare; set start_time and end_time in'
          ' the --compare_config file, at the top level or per workload.'
      )
  config = compare.load_config(FLAGS.compare_config)

  specs = config.workloads
  project_id = _project_id(required=False)
  no_project = [
      s.workload_id
      for s in specs
      if not s.entries_path and not (s.project_id or project_id)
  ]
  if no_project:
    raise app.UsageError(
        f'No GCP project for {", ".join(no_project)}: set project_id in the'
        ' --compare_config file, --project_id or GOOGLE_CLOUD_PROJECT.'
    )
  print(
      f'Loading the logs of {len(specs)} workloads, up to'
      f' {FLAGS.compare_parallel_fetches} at a time...'
  )
  loaded = compare.load_workloads(
      specs,
      project_id or '',
      cache_dir=_cache_dir(),
      max_parallel=FLAGS.compare_parallel_fetches,
      jobset_lookup=FLAGS.compare_jobset_lookup,
      refresh=FLAGS.refresh_cache,
  )
  # Metrics are computed one workload at a time, in the config order.
  summaries = [
      compare.summarize_workload(w.entries, w.spec, jobset=w.jobset)
      for w in loaded
      if not w.error
  ]

  skipped = [w for w in loaded if w.error]
  if skipped:
    print(f'\nSkipped {len(skipped)} of {len(loaded)} workloads:')
    for w in skipped:
      print(f'  {compare.display_name(w.spec)}: {w.error}')
      if w.spec.baseline:
        print('    Its group falls back to the default baseline.')
  if not summaries:
    print('No workload could be loaded; nothing to compare.')
    sys.exit(1)
  try:
    # Workloads without a "group" are only grouped now, by model and scale.
    compare.check_baselines(summaries)
  except ValueError as e:
    print(f'Error in {FLAGS.compare_config}: {e}')
    sys.exit(1)

  # A directory, or no path, gets a file named after the current time.
  name = f'comparison_{datetime.datetime.now():%Y%m%d%H%M}'
  output_path = compare.output_file(
      FLAGS.output_path if FLAGS['output_path'].present else None,
      f'{name}.html',
  )
  compare.to_comparison_html_file(
      summaries,
      output_path,
      title=FLAGS.compare_title or config.title,
      subtitle=config.subtitle,
      description=config.description,
      groups=config.groups,
  )
  print()
  print(compare.format_text_table(summaries))
  print()
  print(f'Comparison written to {output_path}')
  if FLAGS.compare_csv_path:
    csv_path = compare.output_file(FLAGS.compare_csv_path, f'{name}.csv')
    compare.to_csv_file(summaries, csv_path, groups=config.groups)
    print(f'CSV written to {csv_path}')


def _project_id(required: bool) -> Optional[str]:
  """--project_id, else the GOOGLE_CLOUD_PROJECT environment variable."""
  project_id = FLAGS.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
  if required and not project_id:
    raise app.UsageError(
        '--project_id or the GOOGLE_CLOUD_PROJECT environment variable must be'
        ' set to read logs from Cloud Logging.'
    )
  return project_id


def _working_dir() -> str:
  """The directory the command was run from, also under `bazel run`."""
  return os.environ.get('BUILD_WORKING_DIRECTORY') or os.getcwd()


def _cache_dir() -> Optional[str]:
  """--entries_cache_dir relative to the working directory; None if empty."""
  if not FLAGS.entries_cache_dir:
    return None
  return os.path.join(_working_dir(), FLAGS.entries_cache_dir)


def _ensure_output_dirs() -> None:
  """Creates outputs/compare and outputs/timeline in the working directory.

  Version control keeps no empty directories, so the directories used by the
  README commands are created here. Under `bazel run`, the working directory is
  the one the command was run from.
  """
  cwd = _working_dir()
  for name in ('compare', 'timeline'):
    os.makedirs(os.path.join(cwd, 'outputs', name), exist_ok=True)


def main(argv: list[str]) -> None:
  _ensure_output_dirs()
  if FLAGS.command == 'list':
    project_id = _project_id(required=True)
    workloads = log_fetcher.list_workloads(project_id)
    if not workloads:
      print('No workloads with goodput logs found.')
      return
    print(
        f'Found {len(workloads)} workload(s) with goodput logs in'
        f' {project_id}:'
    )
    for w in workloads:
      print(f' {w}')
  elif FLAGS.command == 'compare':
    _compare()
  elif FLAGS.command == 'export':
    workload_id = FLAGS.workload_id or 'workload'
    if FLAGS.entries_path:
      workload_id = FLAGS.workload_id or os.path.splitext(
          os.path.basename(FLAGS.entries_path)
      )[0]
      print(f'Reading logs for {workload_id!r} from {FLAGS.entries_path}')
      entries = compare.read_entries_file(FLAGS.entries_path)
    else:
      if not FLAGS.workload_id:
        raise app.UsageError(
            '--workload_id or --entries_path must be specified for export.'
        )
      print(f'Loading logs for {FLAGS.workload_id!r}')
      entries = compare.load_entries(
          compare.WorkloadSpec(
              workload_id=FLAGS.workload_id,
              start_time=FLAGS.start_time,
              end_time=FLAGS.end_time,
          ),
          _project_id(required=True),
          cache_dir=_cache_dir(),
          refresh=FLAGS.refresh_cache,
      )
    if not entries:
      print(
          'No log entries found. Check project_id, workload_id, and time range.'
      )
      sys.exit(1)
    print(f'  {len(entries)} entries retrieved.')

    parsed = event_parser.parse_entries(entries)
    segments = builder.build_timeline(parsed)
    renderer.to_html_file(
        segments, workload_id, FLAGS.output_path, entries=entries
    )
    print(f'Timeline written to {FLAGS.output_path}')


if __name__ == '__main__':
  app.run(main)
