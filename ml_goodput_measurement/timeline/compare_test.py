import contextlib
import copy
import csv
import dataclasses
import datetime
import html
import io
import json
import os
import re
import threading
from unittest import mock

from absl.testing import absltest

from cloud_goodput.ml_goodput_measurement.timeline import compare
from cloud_goodput.ml_goodput_measurement.timeline import metrics

# One run with steps 0-2, then a single run, no interruption.
_SINGLE_RUN = [
    {'job_start_time': 100.0},
    {'step_count': 0, 'step_start_time': 110.0},
    {'step_count': 1, 'step_start_time': 120.0},
    {'step_count': 2, 'step_start_time': 130.0},
    {'job_end_time': 140.0},
]

# A goodput log with only checkpoint saves: no job or step events.
_CHECKPOINT_ONLY = [
    {
        'checkpoint_manager_blocking_start_time': 100.0 + 30.0 * i,
        'checkpoint_manager_blocking_duration_secs': 2.0,
        'time_between_consecutive_saves_sec': 30.0,
    }
    for i in range(3)
]


def _recovery_entries(scale: float = 1.0) -> list[dict[str, object]]:
  """Two runs and one elastic recovery; all durations scale with `scale`.

  With scale=1 the recovery phases are (seconds): fault detection 2, slice
  wait 10, cluster init 3, TPU init 5, data iterator init 4, checkpoint
  restore 6, AOT compile 8, first batch data loading 2, JIT compile 5 and
  lost progress 20, i.e. downtime 40 and MTTR 65.

  Args:
    scale: Factor applied to all durations.

  Returns:
    Goodput log payloads.
  """

  def t(x: float) -> float:
    return 100.0 + (x - 100.0) * scale

  return [
      {'job_start_time': t(100.0)},
      {'step_count': 0, 'step_start_time': t(110.0)},
      {'step_count': 1, 'step_start_time': t(120.0)},
      {'step_count': 2, 'step_start_time': t(130.0)},
      {
          'elastic_wait_start_time': t(142.0),
          'elastic_wait_event_type': 'elastic_slice_down',
      },
      {
          'elastic_wait_end_time': t(152.0),
          'elastic_wait_event_type': 'elastic_slice_down',
      },
      {'elastic_reinit_start_time': t(152.0)},
      {'tpu_init_start_time': t(155.0)},
      {'tpu_init_end_time': t(160.0)},
      {'training_prep_start_time': t(160.0)},
      {
          'checkpoint_manager_start_time': t(164.0),
          'checkpoint_manager_duration_secs': 6.0 * scale,
          'event_type': 'restore',
          'step': 1,
          'directory': 'gs://bucket/checkpoints',
      },
      {'training_prep_end_time': t(170.0)},
      {'elastic_reinit_end_time': t(178.0)},
      {'data_loading_start_time': t(178.0)},
      {'data_loading_end_time': t(180.0)},
      # Resumes at step 1 (steps 1 and 2 are repeated).
      {'step_count': 1, 'step_start_time': t(180.0)},
      {'step_count': 2, 'step_start_time': t(195.0)},
      {'step_count': 3, 'step_start_time': t(205.0)},
      {'job_end_time': t(215.0)},
  ]


_EXPECTED_GROUP_MEANS = {
    'infra': 12.0,  # 2 + 10
    'init': 14.0,  # 3 + 5 + 4 + 2
    'restore': 6.0,
    'xla': 13.0,  # 8 + 5
    'lost': 20.0,
}

# The default group of a Llama-405B workload with 4x1k in its id.
_GROUP = 'Llama-405B \u00b7 4\u00d71k'


def _container(name, args=(), tpu=None):
  container = {'name': name, 'args': list(args)}
  if tpu is not None:
    container['resources'] = {'limits': {'google.com/tpu': tpu}}
  return container


def _replicated_job(name, replicas, parallelism, containers, accelerator=''):
  pod_spec = {'containers': containers}
  if accelerator:
    pod_spec['nodeSelector'] = {
        'cloud.google.com/gke-tpu-accelerator': accelerator
    }
  return {
      'name': name,
      'replicas': replicas,
      'template': {
          'spec': {
              'parallelism': parallelism,
              'template': {'spec': pod_spec},
          }
      },
  }


def _jobset_create(name, *jobs):
  """The protoPayload of a jobsets.create audit log entry."""
  return {
      'methodName': 'io.x-k8s.jobset.v1alpha2.jobsets.create',
      'resourceName': (
          f'jobset.x-k8s.io/v1alpha2/namespaces/default/jobsets/{name}'
      ),
      'request': {
          'metadata': {'name': name},
          'spec': {'replicatedJobs': list(jobs)},
      },
  }


# Pathways: model_name is in the head job's args, the TPU chips are in the
# worker job (2 slices of 256 pods with 4 chips). Numbers are floats and the
# TPU limit a string, as in the audit log.
_PATHWAYS_CREATE = _jobset_create(
    'pw-restart-2x1k-1',
    _replicated_job(
        'pathways-head',
        1.0,
        1.0,
        [
            _container(
                'jax-tpu',
                ['python3 -m train base.yml model_name=llama3.1-405b steps=9'],
            )
        ],
    ),
    _replicated_job(
        'worker-job0',
        2.0,
        256.0,
        [_container('pathways-worker', ['--server_port=29001'], tpu='4')],
        accelerator='tpu7x',
    ),
)


class PhaseGroupTest(absltest.TestCase):

  def test_groups_cover_every_phase_once(self):
    phases = [p for g in compare.PHASE_GROUPS for p in g.phases]
    self.assertCountEqual(phases, metrics.RECOVERY_PHASE_ORDER)
    keys = [g.key for g in compare.PHASE_GROUPS]
    self.assertLen(set(keys), len(keys))


class SpecTest(absltest.TestCase):

  def test_parse_shape(self):
    self.assertEqual(compare.parse_shape('restart-4x1k-0110-2'), '4\u00d71k')
    self.assertEqual(compare.parse_shape('user-2x1k-0110-01'), '2\u00d71k')
    self.assertEqual(compare.parse_shape('snapshot-3x4-1'), '3\u00d74')
    self.assertEqual(compare.parse_shape('long_snake_case_run_id'), '')
    self.assertEqual(compare.parse_shape('box2x1kx'), '')

  def test_config_from_dict(self):
    config = compare.config_from_dict(
        {
            'title': 'Recovery anatomy',
            'description': 'Three recovery modes.',
            'project_id': 'proj',
            'start_time': '2026-01-01T00:00:00Z',
            'groups': [
                {'name': 'Fuji-70B', 'description': '2× v5p-128'},
                'Fuji-7B',
            ],
            'workloads': [
                'wl-a',
                {
                    'workload_id': 'wl-b',
                    'label': 'B',
                    'group': 'Fuji-70B',
                    'timeline_url': 'wl-b.html',
                    'start_time': '2026-01-02T00:00:00Z',
                    'entries_path': 'b.json',
                    'note': 'Shorter run.',
                },
            ],
        },
        base_dir='/configs',
    )
    self.assertEqual(config.title, 'Recovery anatomy')
    self.assertEqual(config.subtitle, '')
    self.assertEqual(config.description, 'Three recovery modes.')
    self.assertEqual(
        config.groups,
        [
            compare.GroupInfo('Fuji-70B', '2× v5p-128'),
            compare.GroupInfo('Fuji-7B'),
        ],
    )
    a, b = config.workloads
    self.assertEqual(a.workload_id, 'wl-a')
    self.assertEqual(a.project_id, 'proj')
    self.assertEqual(a.start_time, '2026-01-01T00:00:00Z')
    self.assertEqual(a.group, '')
    self.assertEqual(b.label, 'B')
    self.assertEqual(b.group, 'Fuji-70B')
    self.assertEqual(b.timeline_url, 'wl-b.html')
    self.assertEqual(b.start_time, '2026-01-02T00:00:00Z')
    self.assertEqual(b.entries_path, '/configs/b.json')
    self.assertEqual(b.note, 'Shorter run.')

  def test_config_errors(self):
    with self.assertRaisesRegex(ValueError, 'workloads'):
      compare.config_from_dict({'title': 'x'})
    with self.assertRaisesRegex(ValueError, 'Unknown config keys'):
      compare.config_from_dict({'workloads': ['a'], 'titel': 'x'})
    with self.assertRaisesRegex(ValueError, 'Unknown workload keys'):
      compare.config_from_dict({'workloads': [{'workload_id': 'a', 'x': 1}]})
    with self.assertRaisesRegex(ValueError, 'workload_id'):
      compare.config_from_dict({'workloads': [{'label': 'a'}]})
    with self.assertRaisesRegex(ValueError, 'Unknown group keys'):
      compare.config_from_dict(
          {'workloads': ['a'], 'groups': [{'name': 'g', 'desc': 'x'}]}
      )
    with self.assertRaisesRegex(ValueError, 'without a name'):
      compare.config_from_dict(
          {'workloads': ['a'], 'groups': [{'description': 'x'}]}
      )
    with self.assertRaisesRegex(ValueError, 'groups'):
      compare.config_from_dict({'workloads': ['a'], 'groups': 'g'})

  def test_format_scale_chips(self):
    self.assertEqual(compare.format_scale_chips('4×1k'), '4×1k (4,096)')
    self.assertEqual(compare.format_scale_chips('2x128'), '2x128 (256)')
    self.assertEqual(compare.format_scale_chips('v5p-128'), 'v5p-128')
    self.assertEqual(compare.format_scale_chips(''), '')
    self.assertEqual(compare.shape_chips('2\u00d71K'), 2048)
    self.assertIsNone(compare.shape_chips('2\u00d7 v5p-128'))
    self.assertEqual(compare.format_chips(1024), '1k')
    self.assertEqual(compare.format_chips(4096), '4k')
    self.assertEqual(compare.format_chips(64), '64')
    self.assertEqual(compare.format_chips(1536), '1536')

  def test_config_model_group_and_baseline(self):
    config = compare.config_from_dict({
        'model': 'Llama-405B',
        'workloads': [
            'a',
            {'workload_id': 'b', 'baseline': True},
            {'workload_id': 'c', 'model': 'Qwen-235B'},
            {'workload_id': 'd', 'group': 'Llama-405B \u00b7 2\u00d71k'},
        ],
    })
    a, b, c, d = config.workloads
    # The top-level model is a default. Without a group, the workload is
    # grouped once its scale is known; see SummaryTest.
    self.assertEqual((a.model, a.group), ('Llama-405B', ''))
    self.assertFalse(a.baseline)
    self.assertTrue(b.baseline)
    self.assertEqual(c.model, 'Qwen-235B')
    self.assertEqual(d.model, 'Llama-405B')
    self.assertEqual(d.group, 'Llama-405B \u00b7 2\u00d71k')
    # A top-level group is a default too.
    config = compare.config_from_dict({'group': 'G', 'workloads': ['x']})
    self.assertEqual(config.workloads[0].group, 'G')

  def test_baseline_errors(self):
    with self.assertRaisesRegex(ValueError, 'must be true or false'):
      compare.config_from_dict(
          {'workloads': [{'workload_id': 'a', 'baseline': 'yes'}]}
      )
    # At most one baseline per group; other groups have their own. Group
    # names match without case and spaces, and with x for ×.
    with self.assertRaisesRegex(ValueError, "in group 'M · 2×1k': a, b"):
      compare.config_from_dict({
          'workloads': [
              {'workload_id': 'a', 'group': 'M · 2×1k', 'baseline': True},
              {'workload_id': 'b', 'group': 'm·2x1k', 'baseline': True},
              {'workload_id': 'c', 'group': 'N', 'baseline': True},
          ],
      })
    # Workloads without a group are grouped by model and scale later, so
    # group_summaries checks them.
    config = compare.config_from_dict({
        'model': 'M',
        'workloads': [
            {'workload_id': 'a-2x1k-1', 'baseline': True},
            {'workload_id': 'b-4x1k-1', 'baseline': True},
        ],
    })
    self.assertLen(config.workloads, 2)

  def test_load_config_resolves_entries_path(self):
    config_dir = self.create_tempdir()
    path = config_dir.create_file(
        'cmp.json',
        content=json.dumps(
            {'workloads': [{'workload_id': 'wl', 'entries_path': 'wl.json'}]}
        ),
    ).full_path
    config = compare.load_config(path)
    self.assertEqual(
        config.workloads[0].entries_path,
        os.path.join(config_dir.full_path, 'wl.json'),
    )


def _timed(
    entries: list[dict[str, object]], start: str = '2026-01-01T00:00:00+00:00'
) -> list[tuple[str, dict[str, object]]]:
  """Pairs (timestamp, payload) a minute apart, like fetch_log_entries."""
  t0 = datetime.datetime.fromisoformat(start)
  return [
      ((t0 + datetime.timedelta(minutes=i)).isoformat(), entry)
      for i, entry in enumerate(entries)
  ]


class LoadEntriesTest(absltest.TestCase):

  def test_entries_path(self):
    path = self.create_tempfile(
        'entries.json', content=json.dumps(_SINGLE_RUN)
    ).full_path
    spec = compare.WorkloadSpec(workload_id='wl', entries_path=path)
    with mock.patch.object(compare.log_fetcher, 'fetch_log_entries') as fetch:
      self.assertEqual(compare.load_entries(spec, 'proj'), _SINGLE_RUN)
    fetch.assert_not_called()

  def test_fetch_uses_cache_for_same_query(self):
    cache_dir = self.create_tempdir().full_path
    spec = compare.WorkloadSpec(
        workload_id='wl', start_time='2026-01-01T00:00:00Z'
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN),
    ) as fetch:
      first = compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      second = compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      fetch.assert_called_once_with('proj', 'wl', '2026-01-01T00:00:00Z', None)
      # An earlier start time is not in the cache, so it is fetched again.
      spec.start_time = '2025-12-31T00:00:00Z'
      compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      self.assertEqual(fetch.call_count, 2)
    self.assertEqual(first, _SINGLE_RUN)
    self.assertEqual(second, _SINGLE_RUN)
    # A cache file can be used as entries_path.
    cached = compare.WorkloadSpec(
        workload_id='wl', entries_path=os.path.join(cache_dir, 'wl.json')
    )
    self.assertEqual(compare.load_entries(cached, 'proj'), _SINGLE_RUN)

  def test_unreadable_cache_is_fetched_again(self):
    cache_dir = self.create_tempdir()
    cache_dir.create_file('wl.json', content='{"query": ')  # Truncated.
    spec = compare.WorkloadSpec(
        workload_id='wl', start_time='2026-01-01T00:00:00Z'
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN),
    ) as fetch:
      entries = compare.load_entries(
          spec, 'proj', cache_dir=cache_dir.full_path
      )
    self.assertEqual(entries, _SINGLE_RUN)
    fetch.assert_called_once()
    # The cache file is rewritten and no temporary file is left behind.
    self.assertEqual(os.listdir(cache_dir.full_path), ['wl.json'])
    with open(
        os.path.join(cache_dir.full_path, 'wl.json'), encoding='utf-8'
    ) as f:
      self.assertEqual(json.load(f)['entries'], _SINGLE_RUN)

  def test_cache_write_failure_keeps_entries(self):
    # The cache directory cannot be created below a regular file.
    blocker = self.create_tempfile('blocker')
    spec = compare.WorkloadSpec(
        workload_id='wl', start_time='2026-01-01T00:00:00Z'
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN),
    ):
      entries = compare.load_entries(
          spec, 'proj', cache_dir=os.path.join(blocker.full_path, 'cache')
      )
    self.assertEqual(entries, _SINGLE_RUN)

  def test_cache_covers_narrower_time_range(self):
    cache_dir = self.create_tempdir().full_path
    spec = compare.WorkloadSpec(
        workload_id='wl',
        start_time='2026-01-01T00:00:00Z',
        end_time='2026-01-01T01:00:00Z',
    )
    # Entries at 00:00, 00:01, ..., 00:04.
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN),
    ) as fetch:
      compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      # From 00:01 (in another time zone) to 00:03, both included.
      narrower = dataclasses.replace(
          spec,
          start_time='2025-12-31T17:01:00-07:00',
          end_time='2026-01-01T00:03:00Z',
      )
      self.assertEqual(
          compare.load_entries(narrower, 'proj', cache_dir=cache_dir),
          _SINGLE_RUN[1:4],
      )
      # Without a time range, all cached entries.
      unbounded = dataclasses.replace(spec, start_time=None, end_time=None)
      self.assertEqual(
          compare.load_entries(unbounded, 'proj', cache_dir=cache_dir),
          _SINGLE_RUN,
      )
      fetch.assert_called_once()
      # A later end time is not in the cache.
      later = dataclasses.replace(spec, end_time='2026-01-01T02:00:00Z')
      compare.load_entries(later, 'proj', cache_dir=cache_dir)
      self.assertEqual(fetch.call_count, 2)
    # The new time range replaces the cached one.
    with open(os.path.join(cache_dir, 'wl.json'), encoding='utf-8') as f:
      cached = json.load(f)
    self.assertEqual(cached['query']['end_time'], '2026-01-01T02:00:00Z')
    self.assertEqual(cached['timestamps'], [t for t, _ in _timed(_SINGLE_RUN)])
    self.assertEqual(cached['entries'], _SINGLE_RUN)

  def test_refresh_fetches_again(self):
    cache_dir = self.create_tempdir().full_path
    spec = compare.WorkloadSpec(
        workload_id='wl', start_time='2026-01-01T00:00:00Z'
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        side_effect=[_timed(_SINGLE_RUN[:2]), _timed(_SINGLE_RUN)],
    ) as fetch:
      compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      self.assertEqual(
          compare.load_entries(spec, 'proj', cache_dir=cache_dir, refresh=True),
          _SINGLE_RUN,
      )
      # The refreshed entries replace the cached ones.
      self.assertEqual(
          compare.load_entries(spec, 'proj', cache_dir=cache_dir), _SINGLE_RUN
      )
    self.assertEqual(fetch.call_count, 2)

  def test_empty_result_is_cached(self):
    cache_dir = self.create_tempdir().full_path
    spec = compare.WorkloadSpec(
        workload_id='wl',
        start_time='2026-01-01T00:00:00Z',
        end_time='2026-01-01T01:00:00Z',
    )
    with mock.patch.object(
        compare.log_fetcher, 'fetch_log_entries', return_value=[]
    ) as fetch:
      for _ in range(2):
        self.assertEqual(
            compare.load_entries(spec, 'proj', cache_dir=cache_dir), []
        )
    fetch.assert_called_once()

  def test_cache_without_timestamps(self):
    # Cache files of earlier versions are only used for the same query.
    query = {
        'project_id': 'proj',
        'workload_id': 'wl',
        'start_time': '2026-01-01T00:00:00Z',
        'end_time': '2026-01-01T01:00:00Z',
    }
    cache_dir = self.create_tempdir()
    cache_dir.create_file(
        'wl.json', content=json.dumps({'query': query, 'entries': _SINGLE_RUN})
    )
    spec = compare.WorkloadSpec(
        workload_id='wl',
        start_time=query['start_time'],
        end_time=query['end_time'],
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN),
    ) as fetch:
      self.assertEqual(
          compare.load_entries(spec, 'proj', cache_dir=cache_dir.full_path),
          _SINGLE_RUN,
      )
      fetch.assert_not_called()
      spec.end_time = '2026-01-01T00:30:00Z'
      compare.load_entries(spec, 'proj', cache_dir=cache_dir.full_path)
      fetch.assert_called_once()

  def test_note_when_cache_was_read_before_end_of_range(self):
    cache_dir = self.create_tempdir().full_path
    # Without an end time, logs written after the read are not in the cache.
    running = compare.WorkloadSpec(
        workload_id='running', start_time='2020-01-01T00:00:00Z'
    )
    # A time range that ended before the read.
    done = compare.WorkloadSpec(
        workload_id='done',
        start_time='2020-01-01T00:00:00Z',
        end_time='2020-01-02T00:00:00Z',
    )
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_log_entries',
        return_value=_timed(_SINGLE_RUN, start='2020-01-01T00:00:00+00:00'),
    ):
      for spec in (running, done):
        compare.load_entries(spec, 'proj', cache_dir=cache_dir)
      with contextlib.redirect_stdout(io.StringIO()) as out:
        compare.load_entries(running, 'proj', cache_dir=cache_dir)
      self.assertIn(', read at ', out.getvalue())
      self.assertIn('NOTE: the cache was read before the end', out.getvalue())
      with contextlib.redirect_stdout(io.StringIO()) as out:
        compare.load_entries(done, 'proj', cache_dir=cache_dir)
      self.assertIn('Using cached entries', out.getvalue())
      self.assertNotIn('NOTE', out.getvalue())


class LoadWorkloadsTest(absltest.TestCase):

  def test_workloads_are_fetched_in_parallel(self):
    # Each fetch returns only once all three fetches are running, so loading
    # the workloads one by one fails with a broken barrier.
    barrier = threading.Barrier(3, timeout=10)

    def fetch(project_id, workload_id, start_time, end_time):
      del project_id, workload_id, start_time, end_time  # Unused.
      barrier.wait()
      return _timed(_SINGLE_RUN)

    specs = [compare.WorkloadSpec(workload_id=w) for w in ('a', 'b', 'c')]
    with mock.patch.object(
        compare.log_fetcher, 'fetch_log_entries', side_effect=fetch
    ):
      loaded = compare.load_workloads(specs, 'proj', max_parallel=3)
    self.assertEqual([w.error for w in loaded], ['', '', ''])
    # Results keep the order of the specs, so the baseline stays first.
    self.assertEqual([w.spec.workload_id for w in loaded], ['a', 'b', 'c'])

  def test_failures_only_skip_their_workload(self):
    def fetch(project_id, workload_id, start_time, end_time):
      del project_id, start_time, end_time  # Unused.
      if workload_id == 'slow':
        raise RuntimeError('400 API deadline too short (DEADLINE_EXCEEDED)')
      return [] if workload_id == 'empty' else _timed(_SINGLE_RUN)

    specs = [
        compare.WorkloadSpec(workload_id=w) for w in ('ok', 'slow', 'empty')
    ]
    with mock.patch.object(
        compare.log_fetcher, 'fetch_log_entries', side_effect=fetch
    ):
      ok, slow, empty = compare.load_workloads(specs, 'proj')
    self.assertEqual(ok.error, '')
    self.assertEqual(ok.entries, _SINGLE_RUN)
    self.assertIn('DEADLINE_EXCEEDED', slow.error)
    self.assertIn('narrower time range', slow.error)
    self.assertIn('no goodput log entries', empty.error)


class JobSetTest(absltest.TestCase):

  def test_pathways_jobset(self):
    info = compare.parse_jobset_request(_PATHWAYS_CREATE)
    self.assertEqual(
        info,
        compare.JobSetInfo(
            name='pw-restart-2x1k-1',
            model_name='llama3.1-405b',
            slices=2,
            chips_per_slice=1024,
            accelerator='tpu7x',
        ),
    )
    self.assertEqual(info.shape, '2\u00d71k')
    self.assertEqual(
        info.describe(),
        'llama3.1-405b, 2\u00d71k (2 slices \u00d7 1,024 chips, tpu7x)',
    )

  def test_mcjax_jobset(self):
    # One TPU job with model_name in its args; the TPU limit is a number.
    args = ['--megascale_num_slices=2', '--model_name="llama3.1-8b"']
    info = compare.parse_jobset_request(
        _jobset_create(
            'mcjax-1',
            _replicated_job(
                'slice-job', 2.0, 16.0, [_container('jax-tpu', args, tpu=4.0)]
            ),
        )
    )
    self.assertEqual(
        (info.name, info.model_name, info.shape),
        ('mcjax-1', 'llama3.1-8b', '2\u00d764'),
    )

  def test_model_name_is_a_whole_key(self):
    args = ['base_model_name=x model_name=y']
    create = _jobset_create(
        'j', _replicated_job('slice-job', 1, 1, [_container('c', args, tpu=4)])
    )
    self.assertEqual(compare.parse_jobset_request(create).model_name, 'y')

  def test_unknown_scale(self):
    # TPU jobs with different chips per slice: only the slice count is known.
    info = compare.parse_jobset_request(
        _jobset_create(
            'j',
            _replicated_job('a', 1, 256, [_container('c', tpu=4)]),
            _replicated_job('b', 1, 16, [_container('c', tpu=4)]),
        )
    )
    self.assertEqual(
        (info.slices, info.chips_per_slice, info.shape), (2, 0, '')
    )
    self.assertEqual(
        info.describe(), 'no model_name, 2 slices, chips per slice unknown'
    )
    # Neither TPU jobs nor a model_name.
    cpu_only = _jobset_create(
        'j', _replicated_job('cpu', 1, 1, [_container('c', ['echo'])])
    )
    self.assertIsNone(compare.parse_jobset_request(cpu_only))
    self.assertIsNone(compare.parse_jobset_request({}))

  def test_jobset_warnings(self):
    jobset = compare.JobSetInfo(
        model_name='llama3.1-405b', slices=2, chips_per_slice=1024
    )
    # The same model and scale, spelled differently: no warnings.
    spec = compare.WorkloadSpec(
        workload_id='pw-2X1K-1',
        model='Llama-3.1-405B',
        shape='2x1024',
        group='Llama-3.1-405B \u00b7 2\u00d71k',
    )
    self.assertEqual(compare.jobset_warnings(spec, jobset), [])
    spec = compare.WorkloadSpec(
        workload_id='pw-4x1k-1',
        model='Llama-3.1-8B',
        shape='4\u00d71k',
        group='Llama \u00b7 4\u00d71k',
    )
    self.assertEqual(
        compare.jobset_warnings(spec, jobset),
        [
            "shape 4×1k in the config differs from the JobSet's 2×1k;"
            ' using 2×1k.',
            "shape 4×1k in the workload id differs from the JobSet's 2×1k;"
            ' using 2×1k.',
            "shape 4×1k in group 'Llama · 4×1k' differs from the JobSet's"
            ' 2×1k.',
            "model 'Llama-3.1-8B' in the config differs from the JobSet's"
            " model_name 'llama3.1-405b'; showing 'Llama-3.1-8B'.",
        ],
    )
    # A group with a scale range is not checked.
    spec = compare.WorkloadSpec(
        workload_id='w', group='L \u00b7 2\u00d764\u20134\u00d71k'
    )
    self.assertEqual(compare.jobset_warnings(spec, jobset), [])

  def test_first_event_time(self):
    t = datetime.datetime(2026, 1, 20, 19, 57, 1, tzinfo=datetime.timezone.utc)
    entries = [
        {'step_count': 0, 'step_start_time': t.timestamp() + 60},
        {'job_start_time': str(t.timestamp())},  # Numeric strings count too.
        {'elastic_wait_event_type': 'elastic_slice_down'},
    ]
    self.assertEqual(compare._first_event_time(entries), t)
    # The test logs do not have Unix times.
    self.assertIsNone(compare._first_event_time(_SINGLE_RUN))

  def test_load_jobset_window_and_cache(self):
    cache_dir = self.create_tempdir().full_path
    spec = compare.WorkloadSpec(workload_id='restart-2x1k-1')
    utc = datetime.timezone.utc
    first_event = datetime.datetime(2026, 1, 20, 19, 57, tzinfo=utc)
    with mock.patch.object(
        compare.log_fetcher,
        'fetch_jobset_create',
        return_value=_PATHWAYS_CREATE,
    ) as fetch:
      info = compare.load_jobset(spec, 'proj', first_event, cache_dir=cache_dir)
      cached = compare.load_jobset(
          spec, 'proj', first_event, cache_dir=cache_dir
      )
    # From 2 days before to 10 minutes after the first goodput event.
    fetch.assert_called_once_with(
        'proj',
        'restart-2x1k-1',
        datetime.datetime(2026, 1, 18, 19, 57, tzinfo=utc),
        datetime.datetime(2026, 1, 20, 20, 7, tzinfo=utc),
    )
    self.assertEqual(info.shape, '2\u00d71k')
    self.assertEqual(cached, info)
    # When no create entry is found, that is cached too.
    other = compare.WorkloadSpec(workload_id='other')
    with mock.patch.object(
        compare.log_fetcher, 'fetch_jobset_create', return_value=None
    ) as fetch:
      for _ in range(2):
        self.assertIsNone(
            compare.load_jobset(other, 'proj', first_event, cache_dir=cache_dir)
        )
    fetch.assert_called_once()
    self.assertCountEqual(
        os.listdir(cache_dir),
        ['restart-2x1k-1.jobset.json', 'other.jobset.json'],
    )

  def test_load_workloads_looks_up_jobsets(self):
    t0 = datetime.datetime(
        2026, 1, 20, 19, 57, tzinfo=datetime.timezone.utc
    ).timestamp()
    entries = [
        {'job_start_time': t0},
        {'step_count': 0, 'step_start_time': t0 + 60},
    ]

    def fetch_jobset(project_id, workload_id, start_time, end_time):
      del project_id, start_time, end_time  # Unused.
      if workload_id == 'broken':
        raise RuntimeError('403 Permission denied')
      return _PATHWAYS_CREATE if workload_id == 'ok' else None

    specs = [
        compare.WorkloadSpec(workload_id='ok', model='Llama-3.1-8B'),
        compare.WorkloadSpec(workload_id='missing'),
        compare.WorkloadSpec(workload_id='broken'),
    ]
    with (
        mock.patch.object(
            compare.log_fetcher,
            'fetch_log_entries',
            return_value=_timed(entries),
        ),
        mock.patch.object(
            compare.log_fetcher, 'fetch_jobset_create', side_effect=fetch_jobset
        ),
    ):
      ok, missing, broken = compare.load_workloads(
          specs, 'proj', jobset_lookup=True
      )
      (not_looked_up,) = compare.load_workloads(specs[:1], 'proj')
    self.assertEqual(ok.jobset.shape, '2\u00d71k')
    self.assertEqual(
        ok.messages,
        [
            (
                'JobSet pw-restart-2x1k-1: llama3.1-405b, 2×1k (2 slices ×'
                ' 1,024 chips, tpu7x)'
            ),
            (
                "WARNING: model 'Llama-3.1-8B' in the config differs from the"
                " JobSet's model_name 'llama3.1-405b'; showing 'Llama-3.1-8B'."
            ),
        ],
    )
    self.assertIsNone(missing.jobset)
    self.assertIn(
        'No JobSet create entry in the 2 days before the first goodput event'
        ' (2026-01-20 19:57Z)',
        missing.messages[0],
    )
    # A failed lookup does not skip the workload.
    self.assertEqual(broken.error, '')
    self.assertIsNone(broken.jobset)
    self.assertIn('Permission denied', broken.messages[0])
    self.assertIsNone(not_looked_up.jobset)
    self.assertEqual(not_looked_up.messages, [])

  def test_load_jobset_retries_quota_errors(self):
    spec = compare.WorkloadSpec(workload_id='restart-2x1k-1')
    first_event = datetime.datetime(
        2026, 1, 20, 19, 57, tzinfo=datetime.timezone.utc
    )
    fetch = mock.Mock(
        side_effect=[RuntimeError('429 Quota exceeded'), _PATHWAYS_CREATE]
    )
    with (
        mock.patch.object(compare.log_fetcher, 'fetch_jobset_create', fetch),
        mock.patch.object(compare.time, 'sleep') as sleep,
    ):
      info = compare.load_jobset(spec, 'proj', first_event)
    self.assertEqual(info.shape, '2\u00d71k')
    self.assertEqual(fetch.call_count, 2)
    sleep.assert_called_once_with(5)

  def test_refresh_reads_logs_and_jobsets_again(self):
    cache_dir = self.create_tempdir().full_path
    t0 = datetime.datetime(
        2026, 1, 20, 19, 57, tzinfo=datetime.timezone.utc
    ).timestamp()
    entries = [
        {'job_start_time': t0},
        {'step_count': 0, 'step_start_time': t0 + 60},
    ]
    specs = [
        compare.WorkloadSpec(
            workload_id='restart-2x1k-1', start_time='2026-01-20T00:00:00Z'
        )
    ]
    with (
        mock.patch.object(
            compare.log_fetcher,
            'fetch_log_entries',
            return_value=_timed(entries),
        ) as fetch,
        mock.patch.object(
            compare.log_fetcher,
            'fetch_jobset_create',
            return_value=_PATHWAYS_CREATE,
        ) as fetch_jobset,
    ):
      for refresh in (False, False, True):
        (loaded,) = compare.load_workloads(
            specs,
            'proj',
            cache_dir=cache_dir,
            jobset_lookup=True,
            refresh=refresh,
        )
        self.assertEqual(loaded.jobset.shape, '2\u00d71k')
    # Read once, then cached, then read again.
    self.assertEqual(fetch.call_count, 2)
    self.assertEqual(fetch_jobset.call_count, 2)


class SummaryTest(absltest.TestCase):

  def test_summarize_workload(self):
    s = compare.summarize_workload(
        _recovery_entries(),
        compare.WorkloadSpec(workload_id='run-a-2x1k-1', label='A'),
    )
    self.assertEqual(s.label, 'A')
    self.assertEqual(s.shape, '2\u00d71k')
    self.assertEqual(s.num_incidents, 1)
    self.assertAlmostEqual(s.mean_mttr_s, 65.0)
    self.assertAlmostEqual(s.mean_time_to_recover_s, 45.0)
    self.assertAlmostEqual(s.mean_lost_progress_s, 20.0)
    for group in compare.PHASE_GROUPS:
      self.assertAlmostEqual(
          s.group_mean_s(group), _EXPECTED_GROUP_MEANS[group.key]
      )
    self.assertAlmostEqual(
        sum(s.group_share_pct(g) for g in compare.PHASE_GROUPS), 100.0
    )
    self.assertIsNotNone(s.goodput_pct)
    self.assertBetween(s.goodput_pct, 0.0, 100.0)
    self.assertGreater(s.mtbi_s, 0.0)

  def test_summarize_workload_does_not_mutate_entries(self):
    # Without job_end_time, calculate_goodput_metrics appends one to its input.
    entries = _recovery_entries()[:-1]
    expected = copy.deepcopy(entries)
    compare.summarize_workload(entries, compare.WorkloadSpec(workload_id='wl'))
    self.assertEqual(entries, expected)

  def test_summarize_workload_without_incidents(self):
    s = compare.summarize_workload(
        list(_SINGLE_RUN), compare.WorkloadSpec(workload_id='wl')
    )
    self.assertEqual(s.label, 'wl')
    self.assertEqual(s.shape, '')
    self.assertEqual(s.num_incidents, 0)
    self.assertAlmostEqual(s.goodput_pct, 75.0)
    self.assertEqual(s.mean_time_to_recover_s, 0.0)
    for group in compare.PHASE_GROUPS:
      self.assertEqual(s.group_share_pct(group), 0.0)

  def test_goodput_unavailable(self):
    with mock.patch.object(
        compare.metrics,
        'calculate_goodput_metrics',
        side_effect=TypeError('unexpected keyword argument'),
    ):
      s = compare.summarize_workload(
          list(_SINGLE_RUN), compare.WorkloadSpec(workload_id='wl')
      )
    self.assertIsNone(s.goodput_pct)
    self.assertIn('n/a', compare.format_text_table([s]))

  def test_summarize_workload_without_steps(self):
    # cloud_goodput would report 0% over the span of the checkpoint saves.
    with mock.patch.object(
        compare.metrics, 'calculate_goodput_metrics'
    ) as calculate:
      s = compare.summarize_workload(
          list(_CHECKPOINT_ONLY),
          compare.WorkloadSpec(workload_id='wl', note='McJAX run'),
      )
    calculate.assert_not_called()
    self.assertIsNone(s.goodput_pct)
    self.assertEqual(s.total_job_time_s, 0.0)
    self.assertEqual(s.num_incidents, 0)
    self.assertEqual(s.note, f'McJAX run. {compare._NO_STEPS_NOTE}')
    s = compare.summarize_workload(
        list(_CHECKPOINT_ONLY), compare.WorkloadSpec(workload_id='wl')
    )
    self.assertEqual(s.note, compare._NO_STEPS_NOTE)

  def test_model_scale_and_group_from_jobset(self):
    jobset = compare.JobSetInfo(
        model_name='llama3.1-405b', slices=2, chips_per_slice=1024
    )
    # The JobSet's scale is used instead of the id's, and the config's model
    # instead of model_name. The group is 'model · scale'.
    s = compare.summarize_workload(
        list(_SINGLE_RUN),
        compare.WorkloadSpec(workload_id='pw-4x1k-1', model='Llama-3.1-405B'),
        jobset=jobset,
    )
    self.assertEqual((s.model, s.shape), ('Llama-3.1-405B', '2\u00d71k'))
    self.assertEqual(s.group, 'Llama-3.1-405B \u00b7 2\u00d71k')
    # Without a model in the config, model_name is shown.
    s = compare.summarize_workload(
        list(_SINGLE_RUN),
        compare.WorkloadSpec(workload_id='mcjax-1'),
        jobset=jobset,
    )
    self.assertEqual(s.group, 'llama3.1-405b \u00b7 2\u00d71k')
    # Without a JobSet, the shape is the config's, else the id's. A group
    # from the config is kept.
    s = compare.summarize_workload(
        list(_SINGLE_RUN),
        compare.WorkloadSpec(workload_id='pw-4x1k-1', group='G'),
    )
    self.assertEqual((s.model, s.shape, s.group), ('', '4\u00d71k', 'G'))
    s = compare.summarize_workload(
        list(_SINGLE_RUN),
        compare.WorkloadSpec(workload_id='pw-4x1k-1', shape='2\u00d764'),
    )
    self.assertEqual(s.group, '2\u00d764')


class RenderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.summaries = [
        compare.summarize_workload(
            _recovery_entries(),
            compare.WorkloadSpec(
                workload_id='restart-4x1k-1',
                label='Restart',
                model='Llama-405B',
                timeline_url='runs/restart-4x1k-1.html',
            ),
        ),
        compare.summarize_workload(
            _recovery_entries(scale=2.0),
            compare.WorkloadSpec(
                workload_id='resize-4x1k-2',
                label='Resize',
                model='Llama-405B',
                note='Shorter run.',
            ),
        ),
        compare.summarize_workload(
            list(_SINGLE_RUN), compare.WorkloadSpec(workload_id='wl-single')
        ),
    ]

  def test_scaled_workload_doubles_every_group(self):
    base, scaled, _ = self.summaries
    for group in compare.PHASE_GROUPS:
      self.assertAlmostEqual(
          scaled.group_mean_s(group), 2 * base.group_mean_s(group)
      )

  def test_group_summaries(self):
    # Without a group, a workload is grouped by its model and scale.
    self.assertEqual(self.summaries[0].model, 'Llama-405B')
    self.assertEqual(self.summaries[0].group, _GROUP)
    groups = compare.group_summaries(self.summaries)
    self.assertEqual([g.name for g in groups], [_GROUP, 'Other'])
    self.assertEqual(
        [s.workload_id for s in groups[0].summaries],
        ['restart-4x1k-1', 'resize-4x1k-2'],
    )
    self.assertEqual(groups[1].summaries, [self.summaries[2]])
    # Configured groups come first, in the config order, with descriptions.
    # Names match without case and spaces, and with x for ×; the configured
    # name is shown.
    groups = compare.group_summaries(
        self.summaries,
        [
            compare.GroupInfo('Empty'),
            compare.GroupInfo('llama-405b\u00b74x1k', 'TPU7x'),
        ],
    )
    self.assertEqual(
        [g.name for g in groups], ['llama-405b\u00b74x1k', 'Other']
    )
    self.assertEqual(groups[0].description, 'TPU7x')
    # Without any group, all workloads form one unnamed group.
    (single,) = compare.group_summaries(self.summaries[2:])
    self.assertEqual(single.name, '')

  def test_group_row_text(self):
    group, _ = compare.group_summaries(
        self.summaries, [compare.GroupInfo(_GROUP, 'TPU7x')]
    )
    # Steps are 10 s apart in the baseline and 20 s apart in the scaled run.
    # The group name already has the 4×1k scale, so it is not repeated.
    self.assertEqual(
        compare._group_row_text(group),
        f'{_GROUP} (TPU7x \u00b7 Step time 10.00\u201320.00s)',
    )
    group = dataclasses.replace(group, name='Llama-405B', description='')
    self.assertEqual(
        compare._group_row_text(group),
        'Llama-405B (Scale 4×1k \u00b7 Step time 10.00\u201320.00s)',
    )

  def test_scale_range(self):
    def summary(shape):
      return compare.WorkloadSummary(workload_id='w', label='w', shape=shape)

    mixed = [summary('4×1k'), summary('2×1k'), summary('')]
    self.assertEqual(compare._scale_range(mixed), '2×1k\u20134×1k')
    self.assertEqual(compare._scale_range([summary('4×1k')] * 2), '4×1k')
    self.assertEqual(compare._scale_range([summary('')]), '')
    # A group named after its scale range does not repeat it.
    group = compare.SummaryGroup(
        name='Llama-405B \u00b7 2x1K\u20134×1k', description='', summaries=mixed
    )
    self.assertEqual(compare._group_row_text(group), group.name)

  def test_color_classes(self):
    self.assertEqual(compare._goodput_class(None), '')
    self.assertEqual(compare._goodput_class(91.0), 'g')
    self.assertEqual(compare._goodput_class(85.0), 'g')
    self.assertEqual(compare._goodput_class(61.6), 'y')
    self.assertEqual(compare._goodput_class(40.0), 'r')
    # Recovery times vs. the baseline: colored from a 5% change on.
    self.assertEqual(compare._change_class(95.0, 100.0), 'g')
    self.assertEqual(compare._change_class(95.5, 100.0), '')
    self.assertEqual(compare._change_class(104.9, 100.0), '')
    self.assertEqual(compare._change_class(105.0, 100.0), 'r')
    self.assertEqual(compare._change_class(0.0, 20.0), 'g')
    self.assertEqual(compare._change_class(20.0, 0.0), '')
    restart, resize, single = self.summaries
    self.assertEqual(compare._mttr_class(restart, restart), '')
    self.assertEqual(compare._mttr_class(resize, restart), 'r')
    self.assertEqual(compare._mttr_class(restart, resize), 'g')
    self.assertEqual(compare._mttr_class(single, restart), '')
    self.assertEqual(compare._mttr_class(restart, None), '')

  def test_delta_html(self):
    faster = compare._delta_html(65.0, 130.0)
    self.assertIn('-50% vs.\u00a0baseline', faster)
    self.assertIn(compare._GREEN, faster)
    slower = compare._delta_html(130.0, 65.0)
    self.assertIn('+100% vs.\u00a0baseline', slower)
    self.assertIn(compare._RED, slower)
    # Truncated toward 0, so the shown change matches the color.
    self.assertIn('-4% vs.\u00a0baseline', compare._delta_html(95.5, 100.0))
    self.assertIn(compare._GRAY, compare._delta_html(95.5, 100.0))
    self.assertEqual(compare._delta_html(20.0, 0.0), '')
    # Goodput changes are in percentage points; higher is better.
    goodput = compare._delta_html(93.12, 89.66, points=True)
    self.assertIn('+3.46\u00a0pp vs.\u00a0baseline', goodput)
    self.assertIn(compare._GREEN, goodput)
    self.assertIn(
        '+0.00\u00a0pp', compare._delta_html(50.0, 50.001, points=True)
    )

  def test_default_baseline(self):
    restart, resize, single = self.summaries
    no_incidents = dataclasses.replace(
        single, model='Llama-405B', group=restart.group
    )
    groups = compare.group_summaries([no_incidents, restart, resize])
    # The first workload with incidents, not the first workload.
    self.assertIs(groups[0].baseline, restart)
    # Without incidents in the group, the first workload is the baseline.
    (group,) = compare.group_summaries([single])
    self.assertIs(group.baseline, single)

  def test_baseline_override(self):
    restart, resize, _ = self.summaries
    resize.baseline_override = True
    groups = compare.group_summaries(self.summaries)
    self.assertIs(groups[0].baseline, resize)
    self.assertIs(groups[1].baseline, self.summaries[2])
    self.assertIn(
        '<b class="g">2.0\u00d7 faster MTTR</b>',
        compare._mttr_ratio_html(restart, resize),
    )
    self.assertEqual(compare._mttr_ratio_html(resize, resize), '')
    page = compare.to_comparison_html(self.summaries, include_plotlyjs=False)
    self.assertIn('Resize*<div class="sub">baseline</div>', page)
    self.assertIn(
        '<td class="mono g">1.1m (65s)<div class="sub delta"'
        f' style="color:{compare._GREEN}">-50% vs.\u00a0baseline</div></td>',
        page,
    )
    rows = compare.csv_rows(self.summaries)
    self.assertEqual(
        [r['is_group_baseline'] for r in rows], [False, True, True]
    )

  def test_more_than_one_baseline_per_group(self):
    for s in self.summaries[:2]:
      s.baseline_override = True
    with self.assertRaisesRegex(
        ValueError, re.escape(f'"baseline": true in group {_GROUP!r}')
    ):
      compare.group_summaries(self.summaries)
    with self.assertRaisesRegex(ValueError, 'restart-4x1k-1, resize-4x1k-2'):
      compare.check_baselines(self.summaries)

  def test_baseline_note(self):
    restart, resize, single = self.summaries
    # The default baselines have every value that the others have.
    for group in compare.group_summaries(self.summaries):
      self.assertEqual(compare._baseline_note(group), '')

    def group_with_baseline(baseline):
      (group,) = compare.group_summaries([
          restart,
          dataclasses.replace(
              baseline, group=restart.group, baseline_override=True
          ),
      ])
      return group

    self.assertEqual(
        compare._baseline_note(group_with_baseline(single)),
        'Recovery times are not compared: the baseline, wl-single, has no'
        ' recovery incidents.',
    )
    self.assertEqual(
        compare._baseline_note(
            group_with_baseline(dataclasses.replace(resize, goodput_pct=None))
        ),
        'Goodput is not compared: the baseline, Resize, has no Goodput.',
    )

  def test_baseline_without_data_stays_baseline(self):
    restart, resize, _ = self.summaries
    mcjax = compare.summarize_workload(
        list(_CHECKPOINT_ONLY),
        compare.WorkloadSpec(
            workload_id='mcjax-4x1k-1',
            label='McJAX',
            model='Llama-405B',
            baseline=True,
        ),
    )
    summaries = [restart, resize, mcjax]
    (group,) = compare.group_summaries(summaries)
    self.assertIs(group.baseline, mcjax)
    page = compare.to_comparison_html(summaries, include_plotlyjs=False)
    self.assertIn(
        '<tr class="group-note"><td colspan="11">No vs. baseline changes: the'
        ' baseline, McJAX, has no Goodput and no recovery incidents.</td></tr>',
        page,
    )
    self.assertIn('McJAX\u2020<div class="sub">baseline</div>', page)
    self.assertIn('<td class="mono">n/a</td>', page)
    self.assertIn(f'\u2020 {compare._NO_STEPS_NOTE}', page)
    # Nothing is compared with the baseline: no changes and no speed-ups.
    self.assertNotIn('vs.\u00a0baseline', page)
    self.assertNotIn('\u00d7 faster MTTR', page)
    self.assertNotIn('\u00d7 slower MTTR', page)

  def test_comparison_figure(self):
    fig = compare.build_comparison_figure(
        self.summaries, title='Recovery anatomy', subtitle='Mean per incident'
    )
    bars = [t for t in fig.data if t.type == 'bar']
    self.assertLen(bars, 2 * len(compare.PHASE_GROUPS))
    self.assertEqual(
        [t.name for t in bars if t.showlegend is not False],
        [g.label for g in compare.PHASE_GROUPS],
    )
    panel_a = [t for t in bars if t.xaxis == 'x']
    panel_b = [t for t in bars if t.xaxis == 'x2']
    for i, s in enumerate(self.summaries):
      # The largest MTTR is 130 s, so panel A is in minutes.
      self.assertAlmostEqual(sum(t.x[i] for t in panel_a), s.mean_mttr_s / 60.0)
      self.assertAlmostEqual(
          sum(t.x[i] for t in panel_b), 100.0 if s.num_incidents else 0.0
      )
    texts = [a.text for a in fig.layout.annotations]
    self.assertIn('<b>2.2m</b>', texts)
    self.assertTrue(any('No recovery incidents' in t for t in texts))
    self.assertTrue(any('<b>Resize*</b>' in t for t in texts))
    self.assertTrue(any('* Shorter run.' in t for t in texts))
    self.assertTrue(any('MTBI' in t for t in texts))
    self.assertIn('Recovery anatomy', fig.layout.title.text)
    # Panel B labels every segment from 3% on (Ckpt is 6/65 = 9.2%).
    restore = next(t for t in panel_b if t.name == 'Ckpt / Snapshot Restore')
    self.assertEqual(list(restore.text), ['9%', '9%', ''])
    # Panel A labels segments from 6% of the largest MTTR (130 s) on: Ckpt is
    # 6 s in Restart (4.6%) and 12 s in Resize (9.2%).
    restore_a = next(t for t in panel_a if t.name == 'Ckpt / Snapshot Restore')
    self.assertEqual(list(restore_a.text), ['', '<b>0.2m</b>', ''])
    # The hover of a segment lists its sub-phases and the workload totals.
    infra = next(t for t in panel_a if t.name == 'Slice Wait / Infra Recovery')
    hover = infra.customdata[0]
    self.assertIn('<b>Slice Wait / Infra Recovery</b>', hover)
    self.assertIn('Mean per incident: <b>12.0s</b> (12.0s)', hover)
    self.assertIn('<b>18.5%</b> of MTTR', hover)
    self.assertIn(
        '1. Fault Detection / Teardown Gap:</span> 2.0s (3.1%)', hover
    )
    self.assertIn(
        '2. Slice Wait / Infra Reschedule:</span> 10.0s (15.4%)', hover
    )
    self.assertIn('Time to Recover (1\u20139):</span> 45.0s', hover)
    self.assertIn('MTTR (1\u201310):</span> 1.1m', hover)
    self.assertIn('Incidents:</span> 1', hover)
    # Min/max are only shown with more than one incident.
    self.assertNotIn('\u00b7 min ', hover)
    lost = next(
        t for t in panel_a if t.name == 'Lost Progress (Repeated Steps)'
    )
    self.assertIn('Avg repeated steps:</span> 2.0', lost.customdata[0])
    self.assertEqual(list(lost.text), ['<b>0.3m</b>', '<b>0.7m</b>', ''])

  def test_figure_uses_seconds_for_short_recoveries(self):
    fig = compare.build_comparison_figure(self.summaries[:1])
    texts = [a.text for a in fig.layout.annotations]
    self.assertIn('<b>65.0s</b>', texts)
    self.assertTrue(any('(seconds, linear scale)' in t for t in texts))
    # The row label uses the chart unit too.
    self.assertTrue(any('MTTR 65.0s' in t for t in texts))

  def test_comparison_html(self):
    page = compare.to_comparison_html(
        self.summaries,
        title='Recovery <anatomy> & more',
        description='Three <modes>.',
        groups=[compare.GroupInfo(_GROUP, 'TPU7x')],
        include_plotlyjs=False,
    )
    self.assertStartsWith(page, '<!DOCTYPE html>')
    for section in ('table', 'chart', 'notes'):
      self.assertIn(f'id="{section}"', page)
    for removed in ('kpis', 'groups', 'phases'):
      self.assertNotIn(f'id="{removed}"', page)
    self.assertIn('3. Notes &amp; Definitions', page)
    self.assertIn('<title>Recovery &lt;anatomy&gt; &amp; more</title>', page)
    self.assertIn('Three &lt;modes&gt;.', page)
    self.assertIn(
        '<b>Goodput</b> = preserved productive time / total job time,', page
    )
    # Groups are model and scale, as in summarize_workload.
    self.assertIn(
        '<b>Groups</b>: workloads with the same model and scale form a group,',
        page,
    )
    for label in metrics.RECOVERY_PHASE_LABELS.values():
      self.assertIn(html.escape(label, quote=False), page)
    for group in compare.PHASE_GROUPS:
      self.assertIn(html.escape(group.label, quote=False), page)
    # Phase 10 is Lost Progress, as on the timeline page and in the CSV.
    self.assertIn('<th>Lost Progress (Phase 10)</th>', page)
    self.assertNotIn('Wasted Progress (', page)
    # Two levels: a Model column and a row per group.
    self.assertIn('<th>Model</th>', page)
    self.assertIn(
        f'<tr class="group-row"><td colspan="11">{_GROUP} (TPU7x'
        ' \u00b7 Step time 10.00\u201320.00s)</td></tr>',
        page,
    )
    # The step time is known without recovery incidents too: the single-run
    # workload's steps are 10s apart.
    self.assertIn('<td colspan="11">Other (Step time 10.00s)</td>', page)
    self.assertIn('<td class="model"><b>Llama-405B</b></td>', page)
    self.assertIn(
        '<td class="model"><b>\u2014</b></td><td>wl-single</td>', page
    )
    self.assertIn(
        '<a href="runs/restart-4x1k-1.html" target="_blank">restart-4x1k-1</a>',
        page,
    )
    self.assertIn(
        '4×1k (4,096)<div class="sub">Step time 10.00s</div>', page
    )
    self.assertIn('class="mini-bar"', page)
    self.assertIn('20.0s / 2.0 steps (30.8%)', page)
    self.assertIn('Lost Progress 20.0s (30.8%)', page)
    # The first workload with incidents is the group's baseline: it is not
    # colored, and Resize (twice its MTTR) is red.
    self.assertIn('Restart<div class="sub">baseline</div>', page)
    self.assertIn('<td class="mono">1.1m (65s)</td>', page)
    self.assertIn(
        '<td class="mono r">2.2m (130s)<div class="sub delta"'
        f' style="color:{compare._RED}">+100% vs.\u00a0baseline</div></td>',
        page,
    )
    self.assertIn('<b class="r">2.0\u00d7 slower MTTR</b>', page)
    # Goodput has two decimals, like the timeline page.
    self.assertRegex(page, r'<td class="mono \w">\d+\.\d\d%')
    self.assertIn('pp vs.\u00a0baseline', page)
    self.assertIn('<div class="sub">Job time ', page)
    self.assertIn('No recovery incidents', page)
    self.assertIn('Resize*', page)
    self.assertIn('* Shorter run.', page)
    # One Recovery anatomy chart per group, each with collapsed tables.
    self.assertEqual(page.count('class="plot-card"'), 2)
    self.assertIn(f'{_GROUP} \u00b7 2 workloads', page)
    self.assertIn('Other \u00b7 1 workload<', page)
    self.assertEqual(page.count('<details class="breakdown">'), 2)
    self.assertNotIn('<details class="breakdown" open', page)
    self.assertIn('Time to Recover (phases 1\u20139)', page)
    self.assertIn('Total MTTR (phases 1\u201310)', page)

  def test_comparison_html_without_groups(self):
    ungrouped = [
        dataclasses.replace(s, group='', model='') for s in self.summaries
    ]
    page = compare.to_comparison_html(ungrouped, include_plotlyjs=False)
    self.assertNotIn('<th>Model</th>', page)
    self.assertNotIn('class="group-row"', page)
    self.assertEqual(page.count('class="plot-card"'), 1)
    self.assertIn('All workloads \u00b7 3 workloads', page)
    self.assertIn('<title>Goodput Cross-Run Comparison</title>', page)

  def test_long_workload_id_can_wrap(self):
    s = dataclasses.replace(self.summaries[2], workload_id='long_snake_case_id')
    page = compare.to_comparison_html([s], include_plotlyjs=False)
    self.assertIn(
        '<td class="mono wl">long_<wbr>snake_<wbr>case_<wbr>id</td>', page
    )

  def test_csv(self):
    path = os.path.join(self.create_tempdir().full_path, 'cmp.csv')
    compare.to_csv_file(self.summaries, path)
    with open(path, encoding='utf-8') as f:
      rows = list(csv.DictReader(f))
    self.assertLen(rows, 3)
    self.assertEqual(rows[0]['label'], 'Restart')
    self.assertEqual(rows[0]['model'], 'Llama-405B')
    self.assertEqual(rows[0]['group'], _GROUP)
    self.assertEqual(
        [r['is_group_baseline'] for r in rows], ['True', 'False', 'True']
    )
    self.assertEqual(rows[0]['timeline_url'], 'runs/restart-4x1k-1.html')
    self.assertEqual(rows[0]['shape'], '4\u00d71k')
    self.assertAlmostEqual(float(rows[0]['baseline_step_time_s']), 10.0)
    self.assertAlmostEqual(float(rows[0]['mean_mttr_s']), 65.0)
    self.assertAlmostEqual(
        sum(
            float(rows[0][f'group_{g.key}_mean_s'])
            for g in compare.PHASE_GROUPS
        ),
        65.0,
        places=2,
    )
    self.assertAlmostEqual(float(rows[1]['group_lost_mean_s']), 40.0)
    self.assertEqual(rows[2]['num_incidents'], '0')
    for phase in metrics.RECOVERY_PHASE_ORDER:
      self.assertIn(f'phase_{phase}_mean_s', rows[0])

  def test_text_table(self):
    lines = compare.format_text_table(self.summaries).splitlines()
    self.assertLen(lines, 4)
    self.assertIn('Restart', lines[1])
    self.assertIn('1.1m', lines[1])
    self.assertIn('wl-single', lines[3])

  def test_output_file(self):
    tmp = self.create_tempdir().full_path
    # A file name is kept, and its directory created.
    path = compare.output_file(os.path.join(tmp, 'new', 'cmp.html'), 'x.html')
    self.assertEqual(path, os.path.join(tmp, 'new', 'cmp.html'))
    self.assertTrue(os.path.isdir(os.path.join(tmp, 'new')))
    # A directory, existing or ending with a slash, gets the default name.
    self.assertEqual(
        compare.output_file(tmp, 'x.html'), os.path.join(tmp, 'x.html')
    )
    path = compare.output_file(os.path.join(tmp, 'out') + os.sep, 'x.csv')
    self.assertEqual(path, os.path.join(tmp, 'out', 'x.csv'))
    self.assertTrue(os.path.isdir(os.path.join(tmp, 'out')))
    # No path: the default name in the current directory.
    self.assertEqual(compare.output_file(None, 'x.html'), 'x.html')


if __name__ == '__main__':
  absltest.main()
