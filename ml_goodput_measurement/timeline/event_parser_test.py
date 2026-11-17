from absl.testing import absltest

from cloud_goodput.ml_goodput_measurement.timeline import builder
from cloud_goodput.ml_goodput_measurement.timeline import event_parser
from cloud_goodput.ml_goodput_measurement.timeline import events
from cloud_goodput.ml_goodput_measurement.timeline import metrics
from cloud_goodput.ml_goodput_measurement.timeline import renderer
from cloud_goodput.ml_goodput_measurement.timeline import step_times


class EventParserTest(absltest.TestCase):

  def test_safe_float(self):
    self.assertEqual(event_parser.safe_float(123.45), 123.45)
    self.assertEqual(event_parser.safe_float('123.45'), 123.45)
    self.assertIsNone(event_parser.safe_float('nan'))
    self.assertIsNone(event_parser.safe_float('NaN'))
    self.assertIsNone(event_parser.safe_float(float('nan')))
    self.assertIsNone(event_parser.safe_float('invalid'))
    self.assertIsNone(event_parser.safe_float(None))

  def test_standard_pairing(self):
    entries = [
        {'tpu_init_start_time': 100.0},
        {'tpu_init_end_time': 110.0},
        {'training_prep_start_time': 115.0},
        {'training_prep_end_time': 120.0},
        {'step_count': 1, 'step_start_time': 125.0},
        {
            'custom_badput_event_type': 'my_evt',
            'custom_badput_event_start_time': 130.0,
        },
        {
            'custom_badput_event_type': 'my_evt',
            'custom_badput_event_end_time': 135.0,
        },
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [(100.0, 110.0)])
    self.assertEqual(parsed.training_prep, [(115.0, 120.0)])
    self.assertEqual(parsed.steps, [(1, 125.0)])
    self.assertEqual(parsed.custom, {'my_evt': [(130.0, 135.0)]})

  def test_missing_end(self):
    entries = [
        {'tpu_init_start_time': 100.0},
        {'training_prep_start_time': 115.0},
        {'training_prep_end_time': 120.0},
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [])
    self.assertEqual(parsed.training_prep, [(115.0, 120.0)])

  def test_missing_start(self):
    entries = [
        # missing start
        {'tpu_init_end_time': 110.0},
        {'training_prep_start_time': 115.0},
        {'training_prep_end_time': 120.0},
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [])
    self.assertEqual(parsed.training_prep, [(115.0, 120.0)])

  def test_negative_duration(self):
    entries = [
        {'tpu_init_start_time': 120.0},
        {'tpu_init_end_time': 110.0},  # End before Start
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [])

  def test_nan_values(self):
    entries = [
        {'tpu_init_start_time': 'NaN'},
        {'tpu_init_end_time': 110.0},
        {'training_prep_start_time': 115.0},
        {'training_prep_end_time': float('nan')},
        {'step_count': 1, 'step_start_time': 'NaN'},
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [])
    self.assertEqual(parsed.training_prep, [])
    self.assertEqual(parsed.steps, [])

  def test_multiple_starts_before_end(self):
    entries = [
        {'tpu_init_start_time': 100.0},
        {'tpu_init_start_time': 105.0},  # Restart before End 1
        {'tpu_init_end_time': 110.0},  # Ends the active start (105.0)
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.tpu_init, [(105.0, 110.0)])

  def test_checkpoint_events(self):
    entries = [
        {
            'checkpoint_manager_start_time': 1775348089.4568026,
            'checkpoint_manager_duration_secs': 152.9027829170227,
            'event_type': 'restore',
        },
        {
            'checkpoint_manager_blocking_start_time': 1775339579.359531,
            'checkpoint_manager_blocking_duration_secs': 5.8518266677856445,
            'event_type': 'save',
        },
        {
            'checkpoint_manager_start_time': 'NaN',
            'checkpoint_manager_duration_secs': 152.0,
        },
        {
            'checkpoint_manager_blocking_start_time': 1775339579.0,
            'checkpoint_manager_blocking_duration_secs': 'NaN',
        },
        {
            'checkpoint_manager_start_time': 100.0,
            'checkpoint_manager_duration_secs': -10.0,
        },
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(
        parsed.checkpoint_restore,
        [(1775348089.4568026, 1775348089.4568026 + 152.9027829170227)],
    )
    self.assertEqual(
        parsed.checkpoint_save,
        [(1775339579.359531, 1775339579.359531 + 5.8518266677856445)],
    )

  def test_elasticity_events(self):
    entries = [
        {
            'custom_badput_event_start_time': 100.0,
            'custom_badput_event_type': 'elastic_scale_up',
        },
        {
            'custom_badput_event_end_time': 110.0,
            'custom_badput_event_type': 'elastic_scale_up',
        },
        {
            'custom_badput_event_start_time': 120.0,
            'custom_badput_event_type': 'elastic_slice_down',
        },
        {
            'custom_badput_event_end_time': 130.0,
            'custom_badput_event_type': 'elastic_slice_down',
        },
        {
            'custom_badput_event_start_time': 140.0,
            'custom_badput_event_type': 'elastic_reinitialization',
        },
        {
            'custom_badput_event_end_time': 150.0,
            'custom_badput_event_type': 'elastic_reinitialization',
        },
        {
            'custom_badput_event_start_time': 160.0,
            'custom_badput_event_type': 'some_other_evt',
        },
        {
            'custom_badput_event_end_time': 170.0,
            'custom_badput_event_type': 'some_other_evt',
        },
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.elastic_scale_up, [(100.0, 110.0)])
    self.assertEqual(parsed.elastic_slice_down, [(120.0, 130.0)])
    self.assertEqual(parsed.elastic_reinitialization, [(140.0, 150.0)])
    self.assertEqual(parsed.custom, {'some_other_evt': [(160.0, 170.0)]})

  def test_elastic_goodput_keys(self):
    entries = [
        {
            'elastic_wait_start_time': 100.0,
            'elastic_wait_event_type': 'elastic_scale_up',
        },
        {
            'elastic_wait_end_time': 110.0,
            'elastic_wait_event_type': 'elastic_scale_up',
        },
        {
            'elastic_wait_start_time': 120.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_wait_end_time': 130.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_reinit_start_time': 140.0,
        },
        {
            'elastic_reinit_end_time': 150.0,
        },
        {
            'elastic_wait_start_time': 160.0,
            'elastic_wait_event_type': 'some_other_wait',
        },
        {
            'elastic_wait_end_time': 170.0,
            'elastic_wait_event_type': 'some_other_wait',
        },
        {
            'elastic_slice_counts_timestamp': 180.0,
            'active_slices': 2,
            'available_slices': 2,
            'total_slices': 4,
        },
    ]
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(parsed.elastic_scale_up, [(100.0, 110.0)])
    self.assertEqual(parsed.elastic_slice_down, [(120.0, 130.0)])
    self.assertEqual(parsed.elastic_reinitialization, [(140.0, 150.0)])
    self.assertEqual(parsed.elastic_slice_counts, [(180.0, 2, 2, 4)])
    self.assertEqual(parsed.custom, {'some_other_wait': [(160.0, 170.0)]})

  def test_snapshot_restore_and_data_iterator_init_custom_events(self):
    entries = []
    for ctype, start, end in (
        ('SNAPSHOT_RESTORE', 100.0, 105.0),
        ('DATA_ITERATOR_INIT', 110.0, 112.0),
        # Case-insensitive.
        ('snapshot_restore', 120.0, 121.0),
        ('SNAPSHOT_SAVE', 125.0, 127.0),
        ('COMPILATION_NO_XSC', 130.0, 140.0),
    ):
      entries.append({
          'custom_badput_event_start_time': start,
          'custom_badput_event_type': ctype,
      })
      entries.append({
          'custom_badput_event_end_time': end,
          'custom_badput_event_type': ctype,
      })
    parsed = event_parser.parse_entries(entries)
    self.assertEqual(
        parsed.checkpoint_restore,
        [(100.0, 105.0), (120.0, 121.0), (125.0, 127.0)],
    )
    self.assertEqual(
        parsed.snapshot_restore, [(100.0, 105.0), (120.0, 121.0)]
    )
    self.assertEqual(parsed.snapshot_save, [(125.0, 127.0)])
    self.assertEqual(parsed.data_iterator_init, [(110.0, 112.0)])
    # Other custom events stay custom.
    self.assertEqual(parsed.custom, {'COMPILATION_NO_XSC': [(130.0, 140.0)]})


class BuilderTest(absltest.TestCase):

  def test_build_timeline_labels_snapshot_restore(self):
    parsed = event_parser.ParsedLogs(
        job_starts=[0.0],
        job_ends=[],
        tpu_init=[],
        training_prep=[],
        data_loading=[],
        steps=[(1, 15.0), (2, 20.0)],
        checkpoint_save=[],
        checkpoint_restore=[(2.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
        elastic_scale_up=[],
        elastic_slice_down=[],
        elastic_reinitialization=[],
        snapshot_restore=[(5.0, 6.0)],
        snapshot_save=[(7.0, 8.0)],
        custom={},
    )

    segments = builder.build_timeline(parsed)

    restore = builder.EventType.CHECKPOINT_RESTORE
    save = builder.EventType.SNAPSHOT_SAVE
    restores = [
        (s.kind, s.start, s.end, s.label)
        for s in segments
        if s.kind in (restore, save)
    ]
    self.assertCountEqual(
        restores,
        [
            (restore, 2.0, 4.0, 'Checkpoint Restore'),
            (restore, 5.0, 6.0, 'Snapshot Restore'),
            (save, 7.0, 8.0, 'Snapshot Save'),
        ],
    )

  def test_build_timeline_elastic_scale_up(self):
    parsed = event_parser.ParsedLogs(
        job_starts=[0.0],
        job_ends=[],
        tpu_init=[(0.0, 10.0), (31.0, 35.0)],
        training_prep=[],
        data_loading=[],
        steps=[(1, 15.0), (2, 20.0), (3, 40.0), (4, 45.0)],
        checkpoint_save=[],
        checkpoint_restore=[],
        elastic_scale_up=[(22.0, 30.0)],
        elastic_slice_down=[],
        elastic_reinitialization=[(30.0, 38.0)],
        custom={},
    )

    segments = builder.build_timeline(parsed)

    overview_segs = [
        s
        for s in segments
        if s.kind
        in [
            builder.EventType.STARTUP,
            builder.EventType.STEP_BLOCK,
            builder.EventType.INTERRUPTION,
        ]
    ]

    expected_overview = [
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=0.0,
            end=15.0,
            run_idx=0,
            label='Startup',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=15.0,
            end=25.0,
            run_idx=0,
            label='Productive Steps (2 steps)',
        ),
        builder.Segment(
            kind=builder.EventType.INTERRUPTION,
            start=25.0,
            end=30.0,
            run_idx=0,
            label='Interruption',
        ),
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=30.0,
            end=40.0,
            run_idx=1,
            label='Recovery (restart 1)',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=40.0,
            end=50.0,
            run_idx=1,
            label='Productive Steps (2 steps)',
        ),
    ]

    self.assertEqual(len(overview_segs), len(expected_overview))
    for actual, expected in zip(overview_segs, expected_overview):
      self.assertEqual(actual.kind, expected.kind)
      self.assertEqual(actual.start, expected.start)
      self.assertEqual(actual.end, expected.end)
      self.assertEqual(actual.run_idx, expected.run_idx)


class MetricsTest(absltest.TestCase):

  def test_calculate_goodput_metrics_non_elastic(self):
    entries = [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        {'job_end_time': 140.0},
    ]
    goodput_val, badput_breakdown, last_step, total_job_time = (
        metrics.calculate_goodput_metrics(entries, 'test_job')
    )
    self.assertAlmostEqual(goodput_val, 75.0)
    self.assertEqual(last_step, 2)
    self.assertTrue(badput_breakdown)
    self.assertEqual(total_job_time, 40.0)

  def test_calculate_goodput_metrics_elastic(self):
    entries = [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {
            'elastic_wait_start_time': 125.0,
            'elastic_wait_event_type': 'elastic_scale_up',
        },
        {
            'elastic_wait_end_time': 135.0,
            'elastic_wait_event_type': 'elastic_scale_up',
        },
        {'elastic_reinit_start_time': 135.0},
        {'elastic_reinit_end_time': 145.0},
        {'step_count': 2, 'step_start_time': 155.0},
        {'step_count': 3, 'step_start_time': 165.0},
        {'job_end_time': 175.0},
    ]
    goodput_val, badput_breakdown, last_step, total_job_time = (
        metrics.calculate_goodput_metrics(entries, 'test_job')
    )
    self.assertGreater(goodput_val, 0.0)
    self.assertEqual(last_step, 3)
    self.assertTrue(badput_breakdown)
    self.assertEqual(total_job_time, 75.0)

  def test_calculate_goodput_metrics_without_job_start(self):
    # Without job_start_time, the job starts at the earliest timestamp. The
    # checkpoint duration time_between_consecutive_saves_sec is not one.
    entries = [
        {'time_between_consecutive_saves_sec': 5.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        {'job_end_time': 140.0},
    ]
    _, _, last_step, total_job_time = metrics.calculate_goodput_metrics(
        entries, 'test_job'
    )
    self.assertEqual(last_step, 2)
    self.assertEqual(total_job_time, 30.0)

  def test_calculate_timeline_metrics(self):
    segments = [
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK, start=10.0, end=20.0, run_idx=0
        ),
        builder.Segment(
            kind=builder.EventType.INTERRUPTION,
            start=20.0,
            end=30.0,
            run_idx=0,
        ),
        builder.Segment(
            kind=builder.EventType.STARTUP, start=30.0, end=35.0, run_idx=1
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK, start=35.0, end=50.0, run_idx=1
        ),
    ]
    vals = metrics.calculate_timeline_metrics(segments)
    self.assertAlmostEqual(vals['mttr'], 15.0)
    self.assertAlmostEqual(vals['mtbi'], 12.5)

  def _assert_lost_progress_consistent(self, segments, inc):
    """Checks Wasted Progress / Lost Progress / JIT Compile invariants.

    - Timeline Lost Progress blocks sum to the breakdown Lost Progress.
    - Wasted Progress == Lost Progress + JIT Compile inside the wasted window.
    - Every Lost Progress block lies inside the Wasted Progress window.

    Args:
      segments: Timeline segments from builder.build_timeline.
      inc: The single RecoveryIncident for the rolled-back run.
    """
    wasted = [
        s for s in segments if s.kind == builder.EventType.WASTED_PROGRESS
    ]
    lost = [s for s in segments if s.kind == builder.EventType.LOST_PROGRESS]
    jit = [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    self.assertLen(wasted, 1)
    self.assertNotEmpty(lost)
    lost_total = sum(s.duration_s for s in lost)
    jit_in_wasted = sum(
        max(0.0, min(w.end, j.end) - max(w.start, j.start))
        for w in wasted
        for j in jit
    )
    self.assertAlmostEqual(lost_total, inc.lost_progress_s)
    self.assertAlmostEqual(wasted[0].duration_s, lost_total + jit_in_wasted)
    for s in lost:
      self.assertGreaterEqual(s.start, wasted[0].start)
      self.assertLessEqual(s.end, wasted[0].end)

  def test_extract_recovery_incidents_10_phases(self):
    entries = [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        # Disruption after step 2 (step 2 ends around 140.0)
        {
            'elastic_wait_start_time': 142.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_wait_end_time': 152.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {'elastic_reinit_start_time': 152.0},
        {'tpu_init_start_time': 155.0},
        {'tpu_init_end_time': 160.0},
        {'training_prep_start_time': 160.0},
        {
            'checkpoint_manager_start_time': 164.0,
            'checkpoint_manager_duration_secs': 6.0,
            'event_type': 'restore',
        },
        {'training_prep_end_time': 170.0},
        {'elastic_reinit_end_time': 178.0},
        {'data_loading_start_time': 178.0},
        {'data_loading_end_time': 180.0},
        # Resumed at step 1 (rolled back 2 steps: steps 1 & 2)
        {'step_count': 1, 'step_start_time': 180.0},
        {
            'step_count': 2,
            'step_start_time': 195.0,
        },  # 15s step (5s compile overhead over 10s median)
        {'step_count': 3, 'step_start_time': 205.0},
        {'job_end_time': 215.0},
    ]
    parsed = event_parser.parse_entries(entries)
    segments = builder.build_timeline(parsed)
    rec_stats = metrics.extract_recovery_incidents(entries, segments)

    self.assertEqual(rec_stats.num_incidents, 1)
    inc = rec_stats.incidents[0]
    self.assertAlmostEqual(inc.downtime_s, 40.0)  # 180.0 - 140.0
    self.assertEqual(inc.rolled_back_steps, 2)
    self.assertAlmostEqual(inc.lost_progress_s, 20.0)  # 140.0 - 120.0
    self.assertAlmostEqual(
        inc.phases['fault_detection_teardown'], 2.0
    )  # 142 - 140
    self.assertAlmostEqual(
        inc.phases['slice_wait_reschedule'], 10.0
    )  # 152 - 142
    self.assertAlmostEqual(
        inc.phases['pathways_cluster_init'], 3.0
    )  # 155 - 152
    self.assertAlmostEqual(inc.phases['tpu_mesh_model_init'], 5.0)  # 160 - 155
    self.assertAlmostEqual(inc.phases['data_iterator_init'], 4.0)  # 164 - 160
    self.assertAlmostEqual(inc.phases['checkpoint_restore'], 6.0)  # 170 - 164
    self.assertAlmostEqual(inc.phases['aot_xla_compilation'], 8.0)  # 178 - 170
    self.assertAlmostEqual(
        inc.phases['first_batch_data_loading'], 2.0
    )  # 180 - 178
    self.assertAlmostEqual(
        inc.phases['jit_compile_overhead'], 5.0
    )  # (195 - 180) - 10
    self.assertAlmostEqual(
        sum(inc.phases[k] for k in metrics.RECOVERY_PHASE_ORDER[:8]),
        inc.downtime_s,
    )
    self.assertAlmostEqual(
        sum(inc.phases[k] for k in metrics.RECOVERY_PHASE_ORDER),
        inc.mttr_s,
    )

    html_div = renderer.to_html_div(segments, 'test-wl', entries=entries)
    self.assertIn('Per-Recovery Incident Breakdown', html_div)
    self.assertNotIn('(async)', html_div)
    self.assertNotIn('(blocking)', html_div)
    self.assertNotIn('left-shifted to 0 seconds', html_div)
    self.assertNotIn('First-Step JIT', html_div)
    self.assertNotIn('Job-Wide Cumulative', html_div)

    # Verify Wasted Progress container and Lost Progress sub-events
    wasted_segs = [
        s for s in segments if s.kind == builder.EventType.WASTED_PROGRESS
    ]
    lost_segs = [
        s for s in segments if s.kind == builder.EventType.LOST_PROGRESS
    ]
    jit_segs = [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    self.assertEqual(len(wasted_segs), 1)
    self.assertEqual(len(lost_segs), 1)
    self.assertEqual(len(jit_segs), 1)
    # Wasted Progress is the outer container
    self.assertAlmostEqual(wasted_segs[0].start, 180.0)
    self.assertAlmostEqual(wasted_segs[0].end, 205.0)
    # Lost Progress is the carved-out step execution
    self.assertAlmostEqual(lost_segs[0].start, 185.0)
    self.assertAlmostEqual(lost_segs[0].end, 205.0)
    self.assertAlmostEqual(lost_segs[0].duration_s, inc.lost_progress_s)
    self._assert_lost_progress_consistent(segments, inc)

  def _in_place_recovery_entries(self, snapshot_save=False):
    # An in-place elastic recovery that records the snapshot restore as a custom
    # badput event within the training preparation, as AXLearn does, and
    # optionally the snapshot save after it.
    save = [
        {
            'custom_badput_event_start_time': 165.0,
            'custom_badput_event_type': 'SNAPSHOT_SAVE',
        },
        {
            'custom_badput_event_end_time': 168.0,
            'custom_badput_event_type': 'SNAPSHOT_SAVE',
        },
    ]
    return [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        {
            'elastic_wait_start_time': 142.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_wait_end_time': 152.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {'elastic_reinit_start_time': 152.0},
        {'tpu_init_start_time': 152.0},
        {'tpu_init_end_time': 154.0},
        {'training_prep_start_time': 154.0},
        {
            'custom_badput_event_start_time': 155.0,
            'custom_badput_event_type': 'SNAPSHOT_RESTORE',
        },
        {
            'custom_badput_event_end_time': 165.0,
            'custom_badput_event_type': 'SNAPSHOT_RESTORE',
        },
        {'training_prep_end_time': 165.0},
        *(save if snapshot_save else []),
        {'elastic_reinit_end_time': 168.0},
        {'data_loading_start_time': 168.0},
        {'data_loading_end_time': 170.0},
        {'step_count': 2, 'step_start_time': 170.0},
        {'step_count': 3, 'step_start_time': 180.0},
        {'step_count': 4, 'step_start_time': 190.0},
        {'job_end_time': 200.0},
    ]

  def test_extract_recovery_incidents_in_place_custom_snapshot_restore(self):
    for snapshot_save in (False, True):
      with self.subTest(snapshot_save=snapshot_save):
        entries = self._in_place_recovery_entries(snapshot_save)
        parsed = event_parser.parse_entries(entries)
        segments = builder.build_timeline(parsed)
        rec_stats = metrics.extract_recovery_incidents(entries, segments)

        self.assertEqual(rec_stats.num_incidents, 1)
        inc = rec_stats.incidents[0]
        self.assertAlmostEqual(inc.downtime_s, 30.0)  # 170.0 - 140.0
        self.assertAlmostEqual(inc.phases['tpu_mesh_model_init'], 2.0)
        # The rest of the training preparation, not the snapshot restore.
        self.assertAlmostEqual(inc.phases['data_iterator_init'], 1.0)
        # The snapshot restore (165 - 155), and the snapshot save (168 - 165)
        # instead of compilation if it is recorded.
        self.assertAlmostEqual(
            inc.phases['checkpoint_restore'], 13.0 if snapshot_save else 10.0
        )
        self.assertAlmostEqual(
            inc.phases['aot_xla_compilation'], 0.0 if snapshot_save else 3.0
        )
        self.assertAlmostEqual(
            sum(inc.phases[k] for k in metrics.RECOVERY_PHASE_ORDER[:8]),
            inc.downtime_s,
        )

  def test_wasted_progress_carves_out_multiple_jit_intervals(self):
    entries = [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        {'step_count': 3, 'step_start_time': 140.0},
        {'step_count': 4, 'step_start_time': 150.0},
        {
            'elastic_wait_start_time': 160.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_wait_end_time': 170.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {'elastic_reinit_start_time': 170.0},
        {'elastic_reinit_end_time': 180.0},
        {'step_count': 1, 'step_start_time': 180.0},
        {'step_count': 2, 'step_start_time': 283.0},
        {'step_count': 3, 'step_start_time': 293.0},
        {'step_count': 4, 'step_start_time': 396.1},
        {'step_count': 5, 'step_start_time': 486.6},
        {'job_end_time': 500.0},
    ]
    parsed = event_parser.parse_entries(entries)
    segments = builder.build_timeline(parsed)
    rec_stats = metrics.extract_recovery_incidents(entries, segments)
    self.assertEqual(rec_stats.num_incidents, 1)
    inc = rec_stats.incidents[0]

    wasted_segs = [
        s for s in segments if s.kind == builder.EventType.WASTED_PROGRESS
    ]
    lost_segs = [
        s for s in segments if s.kind == builder.EventType.LOST_PROGRESS
    ]
    jit_segs = [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    self.assertEqual(len(wasted_segs), 1)
    self.assertEqual(len(jit_segs), 2)
    self.assertEqual(len(lost_segs), 2)

    # Outer container covers the entire repeated stepping period
    self.assertAlmostEqual(wasted_segs[0].start, 180.0)
    self.assertAlmostEqual(wasted_segs[0].end, 486.6)
    self.assertAlmostEqual(wasted_segs[0].duration_s, 306.6)

    # Inner sub-events partition the pure step execution (Lost Progress)
    total_lost = sum(s.duration_s for s in lost_segs)
    self.assertAlmostEqual(total_lost, inc.lost_progress_s, places=2)
    self.assertAlmostEqual(inc.rolled_back_steps, 4)
    self.assertAlmostEqual(lost_segs[0].start, 273.0)
    self.assertAlmostEqual(lost_segs[0].end, 293.0)
    self.assertAlmostEqual(lost_segs[1].start, 386.1)
    self.assertAlmostEqual(lost_segs[1].end, 486.6)
    self._assert_lost_progress_consistent(segments, inc)

  def test_wasted_progress_carves_out_jit_and_checkpoint_save(self):
    entries = [
        {'job_start_time': 100.0},
        {'step_count': 0, 'step_start_time': 110.0},
        {'step_count': 1, 'step_start_time': 120.0},
        {'step_count': 2, 'step_start_time': 130.0},
        {'step_count': 3, 'step_start_time': 140.0},
        {'step_count': 4, 'step_start_time': 150.0},
        {
            'elastic_wait_start_time': 160.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {
            'elastic_wait_end_time': 170.0,
            'elastic_wait_event_type': 'elastic_slice_down',
        },
        {'elastic_reinit_start_time': 170.0},
        {'elastic_reinit_end_time': 180.0},
        {'step_count': 1, 'step_start_time': 180.0},
        {'step_count': 2, 'step_start_time': 283.0},
        {'step_count': 3, 'step_start_time': 293.0},
        {'step_count': 4, 'step_start_time': 396.1},
        {
            'checkpoint_manager_blocking_start_time': 470.0,
            'checkpoint_manager_blocking_duration_secs': 16.6,
            'event_type': 'save',
        },
        {'step_count': 5, 'step_start_time': 486.6},
        {'job_end_time': 500.0},
    ]
    parsed = event_parser.parse_entries(entries)
    segments = builder.build_timeline(parsed)
    rec_stats = metrics.extract_recovery_incidents(entries, segments)
    self.assertEqual(rec_stats.num_incidents, 1)
    inc = rec_stats.incidents[0]

    wasted_segs = [
        s for s in segments if s.kind == builder.EventType.WASTED_PROGRESS
    ]
    lost_segs = [
        s for s in segments if s.kind == builder.EventType.LOST_PROGRESS
    ]
    jit_segs = [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    ckpt_save_segs = [
        s for s in segments if s.kind == builder.EventType.CHECKPOINT_SAVE
    ]
    self.assertEqual(len(wasted_segs), 1)
    self.assertEqual(len(jit_segs), 2)
    self.assertEqual(len(lost_segs), 2)
    self.assertEqual(len(ckpt_save_segs), 1)

    # Checkpoint save is an overlay sub-event, while Lost Progress accounts
    # for the full non-JIT step time.
    self.assertAlmostEqual(ckpt_save_segs[0].start, 470.0)
    self.assertAlmostEqual(ckpt_save_segs[0].end, 486.6)
    self.assertAlmostEqual(lost_segs[1].start, 386.1)
    self.assertAlmostEqual(lost_segs[1].end, 486.6)
    self.assertAlmostEqual(
        sum(s.duration_s for s in lost_segs), inc.lost_progress_s, places=2
    )
    self._assert_lost_progress_consistent(segments, inc)

  def test_topmost_hover_layer(self):
    # Verify that _add_topmost_hover_layer sets hoverinfo='skip' on original
    # traces, appends an extra transparent bar trace, and for stacked or
    # overlapping bars emits hover text strictly for the topmost layer.
    segs = [
        builder.Segment(
            kind=builder.EventType.WASTED_PROGRESS,
            start=100.0,
            end=200.0,
            run_idx=1,
            label='Wasted Progress (bottom)',
        ),
        builder.Segment(
            kind=builder.EventType.LOST_PROGRESS,
            start=110.0,
            end=190.0,
            run_idx=1,
            label='Lost Progress (middle)',
        ),
        builder.Segment(
            kind=builder.EventType.CHECKPOINT_SAVE,
            start=140.0,
            end=160.0,
            run_idx=1,
            label='Checkpoint Save (topmost)',
        ),
    ]
    fig = renderer.build_figure(segs, 'Test Run')

    # 1. All original drawn traces have hoverinfo='skip'
    original_traces = fig.data[:-1]
    self.assertNotEmpty(original_traces)
    for tr in original_traces:
      self.assertEqual(getattr(tr, 'hoverinfo', None), 'skip')

    # 2. An extra transparent hover-only bar trace is added at the end
    hover_tr = fig.data[-1]
    self.assertEqual(hover_tr.marker.color, 'rgba(0,0,0,0)')
    self.assertEqual(hover_tr.marker.line.width, 0)
    self.assertFalse(hover_tr.showlegend)
    self.assertEqual(hover_tr.hovertemplate, '%{customdata}<extra></extra>')

    # 3. For each elementary interval, the first emitted hover bar in customdata
    # corresponds to the topmost layer (thinnest bar).
    dt_140 = renderer._to_dt(140.0)
    dt_110 = renderer._to_dt(110.0)
    dt_100 = renderer._to_dt(100.0)

    topmost_by_base = {}
    for base, hover in zip(hover_tr.base, hover_tr.customdata):
      if base not in topmost_by_base:
        topmost_by_base[base] = hover

    self.assertIn(dt_140, topmost_by_base)
    self.assertIn(dt_110, topmost_by_base)
    self.assertIn(dt_100, topmost_by_base)

    self.assertTrue(
        topmost_by_base[dt_140].startswith('<b>Checkpoint Save (topmost)</b>')
    )
    self.assertTrue(
        topmost_by_base[dt_110].startswith('<b>Lost Progress (middle)</b>')
    )
    self.assertTrue(
        topmost_by_base[dt_100].startswith('<b>Wasted Progress (bottom)</b>')
    )

  def test_overview_skips_per_step_data_loading_fallback(self):
    # First batch data loading during startup (110-115) should be drawn on
    # 'Startup / Recovery', while per-step data loading during stepping
    # (150-150.002) should be skipped from the overview rather than falling
    # back onto the startup row.
    segs = [
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=100.0,
            end=120.0,
            run_idx=0,
            label='Startup',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=120.0,
            end=200.0,
            run_idx=0,
            label='Productive Steps',
        ),
        builder.Segment(
            kind=builder.EventType.DATA_LOADING,
            start=110.0,
            end=115.0,
            run_idx=0,
            label='First Batch Data Loading',
        ),
        builder.Segment(
            kind=builder.EventType.DATA_LOADING,
            start=150.0,
            end=150.002,
            run_idx=0,
            label='Per-step Data Loading',
        ),
    ]
    fig = renderer.build_overview_figure(segs, 'test-workload')
    dl_color = events.EVENT_STYLE_MAP[events.EventType.DATA_LOADING].color
    dl_traces = [
        tr
        for tr in fig.data
        if getattr(tr, 'marker', None)
        and getattr(tr.marker, 'color', None) == dl_color
    ]
    self.assertEqual(len(dl_traces), 1)
    # Only the startup data loading bar is drawn, not the per-step one.
    self.assertEqual(len(dl_traces[0].x), 1)

  def test_per_run_skips_per_step_data_loading(self):
    # First batch data loading inside the startup window (110-115) is drawn,
    # while per-step data loading inside the stepping container (150-150.002)
    # is skipped: it is asynchronous and fully overlapped with the step.
    segs = [
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=100.0,
            end=120.0,
            run_idx=0,
            label='Startup',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=120.0,
            end=200.0,
            run_idx=0,
            label='Productive Steps',
        ),
        builder.Segment(
            kind=builder.EventType.DATA_LOADING,
            start=110.0,
            end=115.0,
            run_idx=0,
            label='Data Loading',
        ),
        builder.Segment(
            kind=builder.EventType.DATA_LOADING,
            start=150.0,
            end=150.002,
            run_idx=0,
            label='Data Loading',
        ),
    ]
    fig = renderer.build_figure(segs, 'Test Run')
    dl_color = events.EVENT_STYLE_MAP[events.EventType.DATA_LOADING].color
    dl_traces = [
        tr
        for tr in fig.data
        if getattr(tr, 'marker', None)
        and getattr(tr.marker, 'color', None) == dl_color
    ]
    self.assertEqual(len(dl_traces), 1)
    self.assertEqual(len(dl_traces[0].x), 1)
    # The remaining bar is the 5 s first batch one (px.timeline x is in ms).
    self.assertAlmostEqual(float(dl_traces[0].x[0]), 5000.0)

  def test_widened_subevent_capped_at_container_end(self):
    # A 0.1 s JIT compile at the end of the Wasted Progress window is widened
    # to 0.4% of the 10000 s span (40 s) for visibility, but must be capped at
    # the end of its container (200.0) so it never spills onto Productive
    # Steps.
    segs = [
        builder.Segment(
            kind=builder.EventType.WASTED_PROGRESS,
            start=100.0,
            end=200.0,
            run_idx=1,
            label='Wasted Progress',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=200.0,
            end=10100.0,
            run_idx=1,
            label='Productive Steps',
        ),
        builder.Segment(
            kind=builder.EventType.JIT_COMPILE,
            start=199.5,
            end=199.6,
            run_idx=1,
            label='JIT Compile Overhead',
        ),
    ]
    fig = renderer.build_figure(segs, 'Test Run')
    jit_color = events.EVENT_STYLE_MAP[events.EventType.JIT_COMPILE].color
    jit_traces = [
        tr
        for tr in fig.data
        if getattr(tr, 'marker', None)
        and getattr(tr.marker, 'color', None) == jit_color
    ]
    self.assertEqual(len(jit_traces), 1)
    self.assertEqual(len(jit_traces[0].x), 1)
    # 199.5 -> 200.0 = 500 ms (px.timeline x is in ms).
    self.assertAlmostEqual(float(jit_traces[0].x[0]), 500.0)

  def test_rolled_back_steps_from_wasted_segs(self):
    segs = [
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=60.0,
            end=90.0,
            run_idx=0,
            label='Productive Steps Run 0',
        ),
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=100.0,
            end=120.0,
            run_idx=1,
            label='Recovery',
        ),
        builder.Segment(
            kind=builder.EventType.WASTED_PROGRESS,
            start=120.0,
            end=160.0,
            run_idx=1,
            label='Wasted Progress',
            meta={'count': 5},
        ),
        builder.Segment(
            kind=builder.EventType.LOST_PROGRESS,
            start=120.0,
            end=135.0,
            run_idx=1,
            label='Lost Progress Part 1',
            meta={'count': 2},
        ),
        builder.Segment(
            kind=builder.EventType.LOST_PROGRESS,
            start=145.0,
            end=160.0,
            run_idx=1,
            label='Lost Progress Part 2',
            meta={'count': 2},
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=160.0,
            end=200.0,
            run_idx=1,
            label='Productive Steps',
        ),
    ]
    entries = [
        {'job_start_time': 50.0},
        {'step_count': 0, 'step_start_time': 60.0},
        {'step_count': 1, 'step_start_time': 70.0},
        {'step_count': 2, 'step_start_time': 80.0},
        {'elastic_reinit_start_time': 100.0},
        {'elastic_reinit_end_time': 120.0},
        {'step_count': 0, 'step_start_time': 120.0},
        {'step_count': 1, 'step_start_time': 160.0},
        {'job_end_time': 200.0},
    ]
    rec_stats = metrics.extract_recovery_incidents(entries, segs)
    self.assertEqual(rec_stats.num_incidents, 1)
    # Single source of truth is the Wasted Progress count (5), not the sum
    # of the Lost Progress parts (4).
    self.assertEqual(rec_stats.incidents[0].rolled_back_steps, 5)

  def test_single_run_reports_baseline_step_time(self):
    entries = [
        {'job_start_time': 0.0},
        {'tpu_init_start_time': 0.0},
        {'tpu_init_end_time': 10.0},
        {'step_count': 1, 'step_start_time': 15.0},
        {'step_count': 2, 'step_start_time': 25.0},
        {'step_count': 3, 'step_start_time': 36.0},
        {'job_end_time': 50.0},
    ]
    segments = builder.build_timeline(event_parser.parse_entries(entries))
    rec_stats = metrics.extract_recovery_incidents(entries, segments)
    self.assertEqual(rec_stats.num_incidents, 0)
    # Lower median of the step deltas 10s and 11s.
    self.assertEqual(rec_stats.baseline_step_time_s, 10.0)

  def _assert_productive_steps_consistent(self, segments):
    """Checks Stepping / Productive Steps / JIT Compile invariants.

    For every Stepping container: its Productive Steps parts lie inside it,
    do not overlap the JIT Compile Overhead, and with the JIT Compile inside
    the container add up to the container.

    Args:
      segments: Timeline segments from builder.build_timeline.
    """
    containers = [
        s for s in segments if s.kind == builder.EventType.STEP_BLOCK
    ]
    self.assertNotEmpty(containers)
    for c in containers:
      parts = [
          s
          for s in segments
          if s.kind == builder.EventType.PRODUCTIVE_STEPS
          and s.run_idx == c.run_idx
      ]
      jit = [
          s
          for s in segments
          if s.kind == builder.EventType.JIT_COMPILE and s.run_idx == c.run_idx
      ]
      jit_in_container = sum(
          max(0.0, min(c.end, j.end) - max(c.start, j.start)) for j in jit
      )
      self.assertNotEmpty(parts)
      self.assertAlmostEqual(
          sum(p.duration_s for p in parts) + jit_in_container, c.duration_s
      )
      for p in parts:
        self.assertGreaterEqual(p.start, c.start)
        self.assertLessEqual(p.end, c.end)
        for j in jit:
          self.assertLessEqual(min(p.end, j.end) - max(p.start, j.start), 0.0)

  def test_productive_steps_carved_around_jit(self):
    # Steps every 5s except step 3, which takes 12s: its 7s excess is JIT
    # compile overhead (20-27), which splits the Productive Steps in two.
    entries = [
        {'job_start_time': 0.0},
        {'tpu_init_start_time': 0.0},
        {'tpu_init_end_time': 5.0},
        {'step_count': 1, 'step_start_time': 10.0},
        {'step_count': 2, 'step_start_time': 15.0},
        {'step_count': 3, 'step_start_time': 20.0},
        {'step_count': 4, 'step_start_time': 32.0},
        {'step_count': 5, 'step_start_time': 37.0},
        {'step_count': 6, 'step_start_time': 42.0},
        {'job_end_time': 50.0},
    ]
    segments = builder.build_timeline(event_parser.parse_entries(entries))
    self._assert_productive_steps_consistent(segments)

    (container,) = [
        s for s in segments if s.kind == builder.EventType.STEP_BLOCK
    ]
    self.assertEqual((container.start, container.end), (10.0, 47.0))
    self.assertEqual(container.label, 'Stepping (6 steps: 1\u20136)')
    self.assertEqual(container.meta, {'step_range': (1, 6), 'count': 6})
    (jit,) = [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    self.assertEqual((jit.start, jit.end), (20.0, 27.0))
    parts = [
        s for s in segments if s.kind == builder.EventType.PRODUCTIVE_STEPS
    ]
    self.assertEqual(
        [(p.start, p.end, p.label) for p in parts],
        [
            (10.0, 20.0, 'Productive Steps (2 steps: 1\u20132)'),
            # Step 3 continues after its JIT compile, so it counts here.
            (27.0, 47.0, 'Productive Steps (4 steps: 3\u20136)'),
        ],
    )
    self.assertEqual(parts[1].meta, {'step_range': (3, 6), 'count': 4})

    # Without JIT compile there is one Productive Steps part, named like the
    # container.
    plain = [e for e in entries if e.get('step_count') != 4] + [
        {'step_count': 4, 'step_start_time': 25.0}
    ]
    segments = builder.build_timeline(event_parser.parse_entries(plain))
    self._assert_productive_steps_consistent(segments)
    self.assertEmpty(
        [s for s in segments if s.kind == builder.EventType.JIT_COMPILE]
    )
    (part,) = [
        s for s in segments if s.kind == builder.EventType.PRODUCTIVE_STEPS
    ]
    self.assertEqual(part.label, 'Productive Steps (6 steps: 1\u20136)')

  def test_stepping_container_and_productive_steps_in_figures(self):
    entries = [
        {'job_start_time': 0.0},
        {'tpu_init_start_time': 0.0},
        {'tpu_init_end_time': 5.0},
        {'step_count': 1, 'step_start_time': 10.0},
        {'step_count': 2, 'step_start_time': 15.0},
        {'step_count': 3, 'step_start_time': 20.0},
        {'step_count': 4, 'step_start_time': 32.0},
        {'step_count': 5, 'step_start_time': 37.0},
        {'step_count': 6, 'step_start_time': 42.0},
        {'job_end_time': 50.0},
    ]
    segments = builder.build_timeline(event_parser.parse_entries(entries))
    container_color = events.EVENT_STYLE_MAP[events.EventType.STEP_BLOCK].color
    part_color = events.EVENT_STYLE_MAP[events.EventType.PRODUCTIVE_STEPS].color

    # Per-run chart: the container is a wide bar, the parts are thin bars
    # with the legend names of the Wasted Progress / Lost Progress pattern.
    fig = renderer.build_figure(segments, 'Run 1')
    by_color = {tr.marker.color: tr for tr in fig.data if tr.showlegend}
    self.assertEqual(
        by_color[container_color].name,
        'Stepping (Productive Steps + JIT compile overhead)',
    )
    self.assertEqual(
        by_color[part_color].name, 'Productive Steps (excl. JIT compile)'
    )
    self.assertEqual(by_color[container_color].width, 0.68)
    self.assertEqual(by_color[part_color].width, 0.42)
    self.assertLen(by_color[part_color].x, 2)

    # Overview: the parts are overlaid on the Stepping row and, like Lost
    # Progress, their hover does not repeat the container.
    fig = renderer.build_overview_figure(segments, 'wl', entries=entries)
    part_traces = [
        tr
        for tr in fig.data
        if tr.type == 'bar' and tr.marker.color == part_color
    ]
    self.assertNotEmpty(part_traces)
    self.assertEqual(set(part_traces[0].y), {'Stepping'})
    (legend,) = [tr for tr in part_traces if tr.showlegend]
    self.assertEqual(legend.name, 'Productive Steps (excl. JIT compile)')
    hover_tr = next(
        tr
        for tr in fig.data
        if tr.type == 'bar' and tr.marker.color == 'rgba(0,0,0,0)'
    )
    part_hovers = [
        h for h in hover_tr.customdata if 'Productive Steps (' in h
    ]
    self.assertNotEmpty(part_hovers)
    for h in part_hovers:
      self.assertNotIn('Occurs during', h)


class StepTimesTest(absltest.TestCase):

  def test_consecutive_step_deltas(self):
    steps = [
        (1, 0.0),
        (2, 10.0),
        (4, 25.0),  # 2 -> 4 is not consecutive.
        (5, 30.0),
        (6, 700.0),  # 670s spans an interruption.
        (7, 705.0),
        (8, 705.0),  # Zero delta.
    ]
    self.assertEqual(
        step_times.consecutive_step_deltas(steps),
        [
            step_times.StepDelta(step=1, start=0.0, delta_s=10.0),
            step_times.StepDelta(step=4, start=25.0, delta_s=5.0),
            step_times.StepDelta(step=6, start=700.0, delta_s=5.0),
        ],
    )
    self.assertEqual(step_times.consecutive_step_deltas([]), [])

  def test_lower_median(self):
    self.assertEqual(step_times.lower_median([3.0, 1.0, 2.0, 4.0]), 2.0)
    self.assertEqual(step_times.lower_median([7.0]), 7.0)
    self.assertEqual(step_times.lower_median([]), 0.0)

  def test_baseline_step_time(self):
    steps = [(1, 0.0), (2, 10.0), (3, 15.0), (4, 20.0), (9, 900.0)]
    self.assertEqual(step_times.baseline_step_time(steps), 5.0)
    self.assertEqual(step_times.baseline_step_time([(1, 0.0)]), 0.0)

  def test_elastic_baseline_counts_steps_with_all_slices_active(self):
    # 2s steps with 2 of 2 slices active, a slice down at 10s, and 4s steps
    # with 1 of 2 from 12s. Step 4 -> 5 spans the change.
    steps = [(1, 0.0), (2, 2.0), (3, 4.0), (4, 6.0)] + [
        (n, 20.0 + 4.0 * (n - 5)) for n in range(5, 11)
    ]
    counts = [(0.0, 2, 2, 2), (10.0, 0, 0, 2), (12.0, 1, 1, 2)]
    # Without the slice counts, the lower median mixes both: 4s.
    self.assertEqual(step_times.baseline_step_time(steps), 4.0)
    baseline = step_times.Baseline(steps, counts)
    self.assertEqual(baseline.step_time_s, 2.0)
    self.assertEqual(step_times.baseline_step_time(steps, counts), 2.0)
    self.assertTrue(baseline.full_capacity(0.0, 2.0))
    self.assertEqual(baseline.expected_s(0.0, 2.0), 2.0)
    # The 5 steps with 1 of 2 slices active are expected to take their lower
    # median.
    self.assertEqual(baseline.slices(24.0, 28.0), (1, 2))
    self.assertEqual(baseline.expected_s(24.0, 28.0), 4.0)
    # The report without active slices is left out of the expected time.
    self.assertFalse(baseline.full_capacity(6.0, 20.0))
    self.assertEqual(baseline.slices(6.0, 20.0), (1, 2))
    self.assertEqual(baseline.expected_s(6.0, 20.0), 4.0)

  def test_expected_step_time_is_measured_per_slice_count(self):
    # A fixed per-device batch: the 2s steps keep their time with 3 of 4
    # slices active from 20s. From 50s, 2 of 4 are active for only 3 steps,
    # too few for an expected time.
    steps = (
        [(n, 2.0 * n) for n in range(1, 7)]
        + [(n, 22.0 + 2.0 * (n - 7)) for n in range(7, 13)]
        + [(n, 52.0 + 2.0 * (n - 13)) for n in range(13, 17)]
    )
    counts = [(0.0, 4, 4, 4), (20.0, 3, 3, 4), (50.0, 2, 2, 4)]
    baseline = step_times.Baseline(steps, counts)
    self.assertEqual(baseline.step_time_s, 2.0)
    self.assertEqual(baseline.slices(24.0, 26.0), (3, 4))
    self.assertEqual(baseline.expected_s(24.0, 26.0), 2.0)
    # A step that spans a change has the fewest active slices during it.
    self.assertEqual(baseline.expected_s(12.0, 22.0), 2.0)
    self.assertEqual(baseline.slices(54.0, 56.0), (2, 4))
    self.assertIsNone(baseline.expected_s(54.0, 56.0))

  def test_elastic_baseline_without_steps_with_all_slices_active(self):
    # Every step ran with 1 of 2 slices: the baseline is their lower median,
    # which is also their expected time.
    steps = [(n, 10.0 + 4.0 * n) for n in range(1, 7)]
    baseline = step_times.Baseline(steps, [(0.0, 1, 1, 2)])
    self.assertEqual(baseline.step_time_s, 4.0)
    self.assertEqual(baseline.expected_s(14.0, 18.0), 4.0)

  def test_slice_counts_unknown_before_the_first_report(self):
    baseline = step_times.Baseline([(1, 0.0), (2, 2.0)], [(10.0, 1, 1, 2)])
    self.assertEqual(baseline.step_time_s, 2.0)
    self.assertIsNone(baseline.slices(0.0, 2.0))
    self.assertEqual(baseline.expected_s(0.0, 2.0), 2.0)
    self.assertFalse(baseline.full_capacity(0.0, 2.0))

  def test_overlapping(self):
    # Steps 1, 2 and 3 run 0-10s, 10-20s and 20-30s.
    deltas = step_times.consecutive_step_deltas(
        [(1, 0.0), (2, 10.0), (3, 20.0), (4, 30.0)]
    )
    hits = step_times.overlapping(
        [(25.0, 40.0), (12.0, 14.0), (0.0, 1.0)], deltas
    )
    self.assertEqual(hits, [(0.0, 1.0), (12.0, 14.0), (25.0, 40.0)])
    # An interval that ends when a step starts does not overlap it, and a
    # long interval overlaps every step it spans.
    self.assertEqual(
        step_times.overlapping([(-5.0, 25.0), (2.0, 3.0)], deltas),
        [(2.0, 3.0), (-5.0, 25.0), (-5.0, 25.0)],
    )
    self.assertEqual(
        step_times.overlapping([(5.0, 10.0)], deltas), [(5.0, 10.0), None, None]
    )
    self.assertEqual(step_times.overlapping([], deltas), [None, None, None])


def _slice_counts(ts, active, available, total):
  return {
      'elastic_slice_counts_timestamp': ts,
      'active_slices': active,
      'available_slices': available,
      'total_slices': total,
  }


# An elastic Pathways workload with two runs: steps every 5s, an elastic
# scale up at 32s and, after the reinitialization, a slow first step (12s).
# The 20s gap from step 4 to step 5 spans the interruption and must not be
# drawn as a step time. Slice counts are reported at the start and at every
# change.
_PATHWAYS_ENTRIES = [
    {'job_start_time': 0.0},
    {'tpu_init_start_time': 0.0},
    _slice_counts(0.0, 4, 4, 4),
    {'tpu_init_end_time': 10.0},
    {'step_count': 1, 'step_start_time': 15.0},
    {'step_count': 2, 'step_start_time': 20.0},
    {'step_count': 3, 'step_start_time': 25.0},
    {'step_count': 4, 'step_start_time': 30.0},
    {
        'elastic_wait_start_time': 32.0,
        'elastic_wait_event_type': 'elastic_scale_up',
    },
    _slice_counts(32.0, 3, 3, 4),
    {
        'elastic_wait_end_time': 40.0,
        'elastic_wait_event_type': 'elastic_scale_up',
    },
    {'elastic_reinit_start_time': 40.0},
    {'elastic_reinit_end_time': 48.0},
    _slice_counts(48.0, 4, 4, 4),
    {'step_count': 5, 'step_start_time': 50.0},
    {'step_count': 6, 'step_start_time': 62.0},
    {'step_count': 7, 'step_start_time': 67.0},
    {'step_count': 8, 'step_start_time': 72.0},
    {'job_end_time': 80.0},
]
# McJAX (and non-elastic Pathways) logs have no elastic_slice_counts.
_MCJAX_ENTRIES = [
    e for e in _PATHWAYS_ENTRIES if 'elastic_slice_counts_timestamp' not in e
]


def _slice_down_entries(one_slice_step_times):
  """Logs of 2s steps with 2 of 2 slices active, then 1 of 2 from 40s.

  An elastic slice down at 40s and, after the reinitialization, steps from 52s
  with 1 of 2 slices active.

  Args:
    one_slice_step_times: The times of the steps with 1 of 2 slices active.

  Returns:
    The goodput log payloads.
  """
  starts = [52.0]
  for t in one_slice_step_times:
    starts.append(starts[-1] + t)
  return (
      [
          {'job_start_time': 0.0},
          {'tpu_init_start_time': 0.0},
          _slice_counts(0.0, 2, 2, 2),
          {'tpu_init_end_time': 10.0},
      ]
      + [
          {'step_count': n, 'step_start_time': 10.0 + 2 * n}
          for n in range(1, 14)
      ]
      + [
          {
              'elastic_wait_start_time': 40.0,
              'elastic_wait_event_type': 'elastic_slice_down',
          },
          _slice_counts(40.0, 1, 1, 2),
          {
              'elastic_wait_end_time': 45.0,
              'elastic_wait_event_type': 'elastic_slice_down',
          },
          {'elastic_reinit_start_time': 45.0},
          {'elastic_reinit_end_time': 50.0},
      ]
      + [
          {'step_count': n, 'step_start_time': t}
          for n, t in enumerate(starts, start=14)
      ]
      + [{'job_end_time': starts[-1] + 4.0}]
  )


class OverviewPanelsTest(absltest.TestCase):
  """The step time and elastic slices panels of the overview figure."""

  def _figure(self, entries):
    segments = builder.build_timeline(event_parser.parse_entries(entries))
    return renderer.build_overview_figure(segments, 'wl', entries=entries)

  def test_pathways_elastic_workload_has_both_panels(self):
    fig = self._figure(_PATHWAYS_ENTRIES)
    self.assertEqual(list(fig.layout.meta['panels']), ['step_time', 'slices'])
    # Gantt rows, step time, slices and two gaps between them.
    self.assertEqual(fig.layout.height, 150 + 270 + 36 + 150 + 36 + 115 + 45)
    self.assertEqual(fig.layout.xaxis.anchor, 'y3')
    self.assertEqual(fig.layout.yaxis.domain[1], 1.0)
    self.assertGreater(fig.layout.yaxis.domain[0], fig.layout.yaxis2.domain[1])
    self.assertGreater(
        fig.layout.yaxis2.domain[0], fig.layout.yaxis3.domain[1]
    )
    self.assertEqual(fig.layout.yaxis3.domain[0], 0.0)

    # Step time: 3 deltas per run and a gap between the runs; the baseline is
    # 5s, so the 12s step is slow and the 20s gap across runs is not drawn.
    # The 12s step has JIT compile overhead after the reinitialization, so it
    # is drawn as a JIT compile square, not red.
    (step_tr,) = [tr for tr in fig.data if tr.yaxis == 'y2']
    self.assertEqual(step_tr.name, 'Step time')
    self.assertFalse(step_tr.showlegend)
    self.assertLen(step_tr.y, 7)
    self.assertIsNone(step_tr.y[3])
    self.assertEqual(
        [y for y in step_tr.y if y is not None],
        [5.0, 5.0, 5.0, 12.0, 5.0, 5.0],
    )
    self.assertNotIn(renderer._SLOW_STEP_COLOR, list(step_tr.marker.color))
    self.assertEqual(step_tr.marker.color[4], renderer._JIT_STEP_COLOR)
    self.assertEqual(step_tr.marker.symbol[4], 'square')
    self.assertIn(
        'Step time:</span> <b>12.00s</b> \u00b7 2.40\u00d7 baseline (5.00s)',
        step_tr.customdata[4],
    )
    self.assertIn(
        'Slow: JIT compile overhead after the (re)start', step_tr.customdata[4]
    )
    self.assertIn('Run 2 \u2014 Recovery #1', step_tr.customdata[4])
    (legend,) = [a for a in fig.layout.annotations if a.yref == 'y2 domain']
    self.assertIn('Slow (> 1.15\u00d7 baseline): ', legend.text)
    self.assertIn('\u25a0 JIT compile', legend.text)
    self.assertNotIn('checkpoint save', legend.text)
    self.assertEqual(fig.layout.yaxis2.range, (0, 12.5))
    self.assertEqual(fig.layout.yaxis2.title.text, 'Step time (s)')
    baseline_shapes = [
        s for s in fig.layout.shapes if s.type == 'line' and s.yref == 'y2'
    ]
    self.assertLen(baseline_shapes, 1)
    self.assertEqual(baseline_shapes[0].y0, 5.0)

    # Slices: three step lines plus the hover layer, held until the end of
    # the timeline.
    slice_trs = {tr.name: tr for tr in fig.data if tr.yaxis == 'y3'}
    self.assertCountEqual(
        slice_trs, ['Total slices', 'Available slices', 'Active slices', '']
    )
    self.assertEqual(list(slice_trs['Active slices'].y), [4, 3, 4, 4])
    self.assertEqual(slice_trs['Active slices'].line.shape, 'hv')
    self.assertEqual(slice_trs['Active slices'].hoverinfo, 'skip')
    hover_tr = slice_trs['']
    self.assertLen(hover_tr.customdata, 3)
    self.assertIn('<b>3</b>', hover_tr.customdata[1])
    self.assertIn('(75% active)', hover_tr.customdata[1])
    self.assertIn('latest report', hover_tr.customdata[2])
    self.assertEqual(fig.layout.yaxis3.dtick, 1)
    self.assertEqual(fig.layout.yaxis3.title.text, 'Slices')

  def test_mcjax_workload_has_no_slices_panel(self):
    fig = self._figure(_MCJAX_ENTRIES)
    self.assertEqual(list(fig.layout.meta['panels']), ['step_time'])
    self.assertEqual(fig.layout.height, 150 + 270 + 36 + 150 + 45)
    self.assertEqual(fig.layout.xaxis.anchor, 'y2')
    self.assertEqual(fig.layout.yaxis2.domain[0], 0.0)
    self.assertNotIn('yaxis3', fig.layout.to_plotly_json())
    self.assertEmpty([tr for tr in fig.data if tr.yaxis == 'y3'])
    self.assertLen([tr for tr in fig.data if tr.yaxis == 'y2'], 1)

  def test_segments_only_figure_is_unchanged(self):
    segs = [
        builder.Segment(
            kind=builder.EventType.STARTUP,
            start=100.0,
            end=120.0,
            run_idx=0,
            label='Startup',
        ),
        builder.Segment(
            kind=builder.EventType.STEP_BLOCK,
            start=120.0,
            end=200.0,
            run_idx=0,
            label='Productive Steps',
        ),
    ]
    fig = renderer.build_overview_figure(segs, 'wl')
    self.assertEqual(list(fig.layout.meta['panels']), [])
    self.assertEqual(fig.layout.height, 465)
    self.assertEqual(fig.layout.xaxis.anchor, 'y')
    self.assertEqual(fig.layout.yaxis.domain, (0.0, 1.0))
    self.assertEqual(fig.layout.yaxis.autorange, 'reversed')
    # Legend, range slider, range selector and title keep their old offsets.
    self.assertAlmostEqual(fig.layout.legend.y, 1.24, places=2)
    self.assertAlmostEqual(
        fig.layout.xaxis.rangeslider.thickness, 0.07, places=2
    )
    self.assertAlmostEqual(fig.layout.xaxis.rangeselector.y, 1.03, places=2)
    self.assertAlmostEqual(fig.layout.title.y, 0.97, places=2)
    self.assertFalse(fig.layout.shapes)

  def test_overview_badge_lists_panels(self):
    segments = builder.build_timeline(
        event_parser.parse_entries(_PATHWAYS_ENTRIES)
    )
    page = renderer.to_html_div(segments, 'wl', entries=_PATHWAYS_ENTRIES)
    self.assertIn(
        'Full Timeline &amp; Sub-Event Overlay &bull; Step Time vs. Baseline'
        ' &bull; Elastic Slices</span>',
        page,
    )
    self.assertIn('preserved productive / ', page)
    segments = builder.build_timeline(
        event_parser.parse_entries(_MCJAX_ENTRIES)
    )
    page = renderer.to_html_div(segments, 'wl', entries=_MCJAX_ENTRIES)
    self.assertIn('Step Time vs. Baseline</span>', page)
    self.assertNotIn('Elastic Slices', page)

  def test_steps_with_fewer_slices_are_compared_with_their_expected_time(self):
    # 4s steps with 1 of 2 slices active: they are expected to take their
    # lower median, twice the 2s baseline.
    entries = _slice_down_entries([4.0] * 6)
    segments = builder.build_timeline(event_parser.parse_entries(entries))
    # The 4s steps take their expected time: no JIT compile overhead.
    self.assertEmpty(
        [s for s in segments if s.kind == events.EventType.JIT_COMPILE]
    )
    # The baseline only counts the steps with both slices active.
    rec_stats = metrics.extract_recovery_incidents(entries, segments)
    self.assertEqual(rec_stats.baseline_step_time_s, 2.0)

    fig = renderer.build_overview_figure(segments, 'wl', entries=entries)
    step_tr = [tr for tr in fig.data if tr.name == 'Step time'][0]
    self.assertEqual(
        [y for y in step_tr.y if y is not None], [2.0] * 12 + [4.0] * 6
    )
    self.assertEqual(
        {c for c, y in zip(step_tr.marker.color, step_tr.y) if y is not None},
        {renderer._STEP_LINE_COLOR},
    )
    self.assertIn(
        '<b>4.00s</b> \u00b7 1.00\u00d7 expected (4.00s: lower median of the'
        ' steps with 1 of 2 slices active)',
        step_tr.customdata[13],
    )
    # The expected step time and the raised band, while 1 slice is active.
    (expected_tr,) = [tr for tr in fig.data if tr.name == 'Expected step time']
    self.assertEqual(expected_tr.yaxis, 'y2')
    self.assertEqual(list(expected_tr.y), [4.0, 4.0, None])
    self.assertEqual(
        list(expected_tr.x[:2]),
        [renderer._to_dt(52.0), renderer._to_dt(76.0)],
    )
    (band,) = [
        s
        for s in fig.layout.shapes
        if s.type == 'rect' and s.yref == 'y2' and s.xref == 'x'
    ]
    self.assertAlmostEqual(band.y0, 2.3)
    self.assertAlmostEqual(band.y1, 4.6)
    self.assertEqual(band.fillcolor, renderer._BASELINE_BAND_COLOR)
    self.assertGreater(fig.layout.yaxis2.range[1], 4.6)
    (legend,) = [a for a in fig.layout.annotations if a.yref == 'y2 domain']
    self.assertIn('\u25cf > 1.15\u00d7 expected', legend.text)
    self.assertIn('\u2504 Expected with fewer slices', legend.text)

  def test_band_is_lowered_when_steps_with_fewer_slices_are_faster(self):
    # 1.5s steps with 1 of 2 slices active (e.g. a smaller global batch), and
    # a 2s step that is slow for them although it takes the baseline.
    fig = self._figure(_slice_down_entries([1.5] * 5 + [2.0]))
    (step_tr,) = [tr for tr in fig.data if tr.name == 'Step time']
    self.assertEqual(step_tr.marker.color[-1], renderer._SLOW_STEP_COLOR)
    self.assertIn('1.33\u00d7 expected (1.50s', step_tr.customdata[-1])
    (expected_tr,) = [tr for tr in fig.data if tr.name == 'Expected step time']
    self.assertEqual(list(expected_tr.y), [1.5, 1.5, None])
    # The plot background covers the band above 1.15x the expected time, and
    # the baseline line is drawn over it.
    shapes = list(fig.layout.shapes)
    (band,) = [
        s
        for s in shapes
        if s.type == 'rect' and s.yref == 'y2' and s.xref == 'x'
    ]
    self.assertEqual(band.fillcolor, renderer._OVERVIEW_PLOT_BG_COLOR)
    self.assertAlmostEqual(band.y0, 2.3)
    self.assertAlmostEqual(band.y1, 1.725)
    (line,) = [s for s in shapes if s.type == 'line' and s.yref == 'y2']
    self.assertGreater(shapes.index(line), shapes.index(band))

  def test_steps_without_an_expected_time_are_not_marked(self):
    # Only 3 steps with 1 of 2 slices active: too few for an expected time,
    # so these 4s steps are not slow although they take twice the baseline.
    fig = self._figure(_slice_down_entries([4.0] * 3))
    (step_tr,) = [tr for tr in fig.data if tr.name == 'Step time']
    self.assertEqual(set(step_tr.marker.color), {renderer._STEP_LINE_COLOR})
    self.assertIn(
        '1 of 2 slices active: no expected time (fewer than 5 steps with'
        ' them)',
        step_tr.customdata[13],
    )
    self.assertEmpty(
        [tr for tr in fig.data if tr.name == 'Expected step time']
    )
    self.assertEmpty([
        s
        for s in fig.layout.shapes
        if s.type == 'rect' and s.yref == 'y2' and s.xref == 'x'
    ])
    (legend,) = [a for a in fig.layout.annotations if a.yref == 'y2 domain']
    self.assertIn('\u25cf > 1.15\u00d7 baseline', legend.text)
    self.assertNotIn('Expected with fewer slices', legend.text)

  def test_slow_steps_with_a_cause_are_not_red(self):
    # 5s steps; the steps from 30s and from 49s take 9s, the first one with a
    # checkpoint save.
    starts = [10.0, 15.0, 20.0, 25.0, 30.0, 39.0, 44.0, 49.0, 58.0, 63.0, 68.0]
    entries = (
        [
            {'job_start_time': 0.0},
            {'tpu_init_start_time': 0.0},
            {'tpu_init_end_time': 8.0},
        ]
        + [
            {'step_count': n, 'step_start_time': t}
            for n, t in enumerate(starts, start=1)
        ]
        + [
            {
                'checkpoint_manager_blocking_start_time': 31.0,
                'checkpoint_manager_blocking_duration_secs': 3.0,
            },
            {'job_end_time': 80.0},
        ]
    )
    fig = self._figure(entries)
    (step_tr,) = [tr for tr in fig.data if tr.yaxis == 'y2']
    self.assertEqual(step_tr.marker.color[4], renderer._CKPT_STEP_COLOR)
    self.assertEqual(step_tr.marker.symbol[4], 'diamond')
    self.assertIn(
        'Slow: checkpoint save (3.0s) during the step', step_tr.customdata[4]
    )
    self.assertEqual(step_tr.marker.color[7], renderer._SLOW_STEP_COLOR)
    self.assertEqual(step_tr.marker.symbol[7], 'circle')
    self.assertIn(
        'Slow: no checkpoint save or JIT compile in the step',
        step_tr.customdata[7],
    )
    self.assertEqual(
        [c for c in step_tr.marker.color if c != renderer._STEP_LINE_COLOR],
        [renderer._CKPT_STEP_COLOR, renderer._SLOW_STEP_COLOR],
    )
    (legend,) = [a for a in fig.layout.annotations if a.yref == 'y2 domain']
    self.assertIn('Slow (> 1.15\u00d7 baseline): ', legend.text)
    self.assertIn('\u25c6 checkpoint save', legend.text)
    self.assertIn('\u25cf other', legend.text)
    self.assertNotIn('JIT compile', legend.text)


if __name__ == '__main__':
  absltest.main()
