"""Algorithm regression using small invented arrays only."""
import ast
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import synthetic_rules_support as s


class NarrowRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.mfuncs = s.numerical_module(s.ROOT / 'scripts/bw-pchip-onsets-t1t2.py', 'mgi', 'test_mgi')
        cls.f, cls.ffuncs = s.numerical_module(s.ROOT / 'scripts/fbrm-onsets.py', 'fbrm', 'test_fbrm')

    def test_geometry_helper_unchanged(self):
        # Locked pre-patch AST digest protects all geometry/eligibility logic.
        name = 'interpeak_valley_departure_index'
        import hashlib
        self.assertIn(name, self.mfuncs)
        x = np.arange(12) / 10
        d = np.array([0, 10, 5, 1.5, 1.6, 1.7, 1.1, 1.5, 1, 4, 8, 0.])
        kw = dict(x=x, abs_d=d, peak_idx=1, all_peak_indices=[1, 10])
        self.assertEqual(self.m.interpeak_valley_departure_index(**kw), 7)
        self.assertEqual(self.m.interpeak_valley_departure_index(**kw, rise_frac=.10), 2)

    def synthetic(self, forced=True, **extra):
        x = np.arange(12, dtype=float) / 10
        d = np.array([0, 0, 10, 5, 2, 1, 3, .6, .01, .1, 8, 0.])
        options = s.options(self.m) | dict(
            csv_path=Path('synthetic.csv'), x=x, y=-x, dy_dx=-d, d2y_dx2=np.zeros(12),
            min_final_peak_height_frac=0., max_report_peak_gap_C=10., **extra)
        def valley(**kw):
            if kw['peak_idx'] != 2 or not forced:
                return None
            return 8 if kw['rise_frac'] == .03 else 6
        with patch.object(self.m, 'detect_peak_anchors', return_value=([2], [10.])), \
             patch.object(self.m, 'backtrack_derivative_departure_index', return_value=8 if not forced else 5), \
             patch.object(self.m, 'interpeak_valley_departure_index', side_effect=valley) as geometry, \
             patch.object(self.m, 'segmented_intersection_for_peak', return_value=(None, None, None)) as fit, \
             patch.object(self.m, 'baseline_departure_for_peak', return_value=(None, None)), \
             patch.object(self.m, 'build_interpeak_valley_candidates', return_value=[]) as explicit:
            bends, candidates = self.m.detect_bends(**options)
        return bends, candidates, fit, explicit, geometry

    def test_geometry_report_cap_floor_are_distinct_and_same_anchor(self):
        bends, candidates, fit, explicit, geometry = self.synthetic()
        c = candidates[0]
        self.assertEqual(fit.call_args.kwargs['rough_idx'], 8)
        self.assertEqual([a.kwargs['rise_frac'] for a in geometry.call_args_list], [.03, .10])
        self.assertEqual(c.rough_onset_point_index, 8)
        self.assertEqual(c.valley_report.valley_geometry_temperature_C, .8)
        self.assertEqual(c.valley_report.valley_qualified_report_temperature_C, .6)
        self.assertEqual(c.valley_report.post_derivative_cap_temperature_C, .8)
        self.assertEqual(c.report_point_index, 7)
        self.assertEqual(c.peak_point_index, 2)
        self.assertTrue(c.valley_report.forced_valley_floor_applied)
        self.assertEqual(bends[0].valley_report, c.valley_report)
        self.assertEqual(explicit.call_args.kwargs['rise_frac'], .10)

    def test_explicit_valley_report_uses_new_fraction(self):
        m = self.m
        x = np.arange(12) / 10
        d = np.array([0, 10, 5, 1.5, 1.6, 1.7, 1.1, 1.5, 1, 4, 8, 0.])
        options = s.options(self.m) | dict(
            csv_path=Path('synthetic.csv'), x=x, y=-x, dy_dx=-d, d2y_dx2=np.zeros(12))
        with patch.object(m, 'detect_peak_anchors', return_value=([1, 10], [10., 8.])):
            _, candidates = m.detect_bends(**options)
        explicit = [c for c in candidates if '+valley-rise' in c.report_mode]
        self.assertEqual(len(explicit), 1)
        self.assertEqual(explicit[0].report_point_index, 2)
        self.assertEqual(explicit[0].valley_report.interpeak_report_rise_frac, .10)
        self.assertFalse(explicit[0].valley_report.forced_valley_precedence)

    def test_ordinary_reports_unchanged_by_floor(self):
        for mode in ('hybrid', 'intersection', 'departure', 'backtrack'):
            with self.subTest(mode=mode):
                _, c, *_ = self.synthetic(forced=False, report_method=mode)
                _, old, *_ = self.synthetic(forced=False, report_method=mode, forced_valley_report_floor=0.)
                self.assertEqual(c[0].report_point_index, 8)
                self.assertEqual(c[0].temperature_C, old[0].temperature_C)
                self.assertFalse(c[0].valley_report.forced_valley_floor_applied)

    def test_floor_first_supported_crossing_and_disable(self):
        d = np.array([0., 10., .8, .6, np.nan, .01])
        frozen = d.copy(); d.flags.writeable = False
        for forced, floor, expected in [(True, .05, 3), (False, .05, 5), (True, 0, 5)]:
            self.assertEqual(self.m.qualify_forced_valley_report(d, 5, 1, forced=forced, floor=floor), expected)
        np.testing.assert_array_equal(d, frozen)

    def test_mgi_cli_and_legacy_options(self):
        with patch('sys.argv', ['mgi']):
            args = self.m.parse_args()
        self.assertEqual((args.interpeak_rise_frac, args.interpeak_report_rise_frac,
                          args.forced_valley_report_floor), (.03, .10, .05))
        with patch('sys.argv', ['mgi', '--interpeak-report-rise-frac', '.03', '--forced-valley-report-floor', '0']):
            args = self.m.parse_args()
        self.assertEqual((args.interpeak_report_rise_frac, args.forced_valley_report_floor), (.03, 0.))
        self.assertEqual(self.synthetic()[1][0].valley_report.interpeak_report_rise_frac, .10)

    def test_mgi_provenance_serialization(self):
        b, c, *_ = self.synthetic()
        s.OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=s.OUT) as temp:
            for name, rows, writer in [('bends', b, self.m.write_bends_csv), ('candidates', c, self.m.write_candidates_csv)]:
                path = Path(temp) / (name + '.csv')
                writer(path, rows)
                with path.open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(float(row['temperature_C']), .7)
                self.assertEqual(float(row['valley_geometry_temperature_C']), .8)
                self.assertEqual(float(row['valley_qualified_report_temperature_C']), .6)
                self.assertEqual(float(row['post_derivative_cap_temperature_C']), .8)
                self.assertEqual(row['forced_valley_precedence'], 'True')

    @staticmethod
    def candidate(cid, progress, level=1., sustained=1., threshold=1.):
        return NS(candidate_id=cid, cooling_progress_C=progress, level_shift=level,
                  sustained_shift=sustained, level_threshold=threshold, sustained_threshold=threshold,
                  status='candidate', accepted=False, final_rank=None, reason='', acceptance_mode='level')

    def qualify(self, early=None, later=None, separation=1., factor=4.):
        return self.f.nearby_stronger_rejections(
            [early or self.candidate(7, 0), later or self.candidate(9, .5, 5, 5)],
            separation_C=separation, factor=factor)

    def test_both_metrics_required(self):
        self.assertEqual(self.qualify(), {7: 9})

    def test_only_level_is_insufficient(self):
        self.assertEqual(self.qualify(later=self.candidate(9, .5, 5, 3)), {})

    def test_only_sustained_is_insufficient(self):
        self.assertEqual(self.qualify(later=self.candidate(9, .5, 3, 5)), {})

    def test_threshold_max_and_strict_factor_boundary(self):
        self.assertEqual(self.qualify(later=self.candidate(9, .5, 4, 4)), {})
        self.assertEqual(self.qualify(early=self.candidate(7, 0, .1, .1, 2)), {})
        self.assertEqual(self.qualify(later=self.candidate(9, 1, 5, 5)), {7: 9})

    def test_distant_or_earlier_event_cannot_suppress(self):
        for progress in (1.00001, 10, 0, -.5):
            self.assertEqual(self.qualify(later=self.candidate(9, progress, 500, 500)), {})

    def test_ids_order_and_metrics_are_not_mutated(self):
        candidates = [self.candidate(20, 0), self.candidate(3, .5, 5, 5)]
        before = copy.deepcopy([vars(c) for c in candidates])
        self.assertEqual(self.f.nearby_stronger_rejections(candidates, separation_C=1), {20: 3})
        self.assertEqual([vars(c) for c in candidates], before)
        self.assertEqual(self.qualify(factor=0), {})

    def test_first_survivor_and_explicit_rejection_reason(self):
        candidates = [self.candidate(20, 0), self.candidate(21, .5, 5, 5),
                      self.candidate(22, 5, 500, 500)]
        data = NS(analysis_mask=np.ones(7, dtype=bool), Tr_C=np.arange(7))
        args = NS(min_valid_run_points=1, min_separation_C=1.)  # legacy args: new default applies
        with patch.object(self.f, 'local_metrics', side_effect=lambda idx, *a: {'valid_event': idx in (0, 2, 4)}), \
             patch.object(self.f, 'refine_onset_index', return_value=0), \
             patch.object(self.f, 'make_candidate', side_effect=candidates):
            accepted, all_candidates = self.f.find_next_event(
                target_rank=1, search_start_idx=0, candidate_id_start=20, data=data, args=args, refs=[])
        self.assertIs(accepted, candidates[1])  # distant globally strongest event does not win
        self.assertEqual([c.candidate_id for c in all_candidates], [20, 21, 22])
        self.assertTrue(candidates[0].nearby_stronger_rejected)
        self.assertEqual(candidates[0].nearby_stronger_candidate_id, 21)
        self.assertIn('bounded_nearby_stronger: candidate 21', candidates[0].reason)
        self.assertIn('factor 4', candidates[0].reason)
        self.assertEqual(candidates[0].level_shift, 1.)

    def test_fbrm_cli_factor_default_and_disable(self):
        with patch('sys.argv', ['fbrm']):
            self.assertEqual(self.f.parse_args().nearby_stronger_factor, 4.)
        with patch('sys.argv', ['fbrm', '--nearby-stronger-factor', '0']):
            self.assertEqual(self.f.parse_args().nearby_stronger_factor, 0.)

    def test_fbrm_parameter_serialization_with_legacy_args(self):
        with patch('sys.argv', ['fbrm']):
            args = self.f.parse_args()
        del args.nearby_stronger_factor
        # Supply only serialization context; no preparation or detector runs.
        names = {n.attr for n in ast.walk(self.ffuncs['write_params'])
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'data'}
        data = NS(**{name: None for name in names})
        data.analysis_mask = np.ones(10, dtype=bool)
        data.initial_baseline = self.f.InitialBaseline(**{key: 0 for key in self.f.InitialBaseline.__dataclass_fields__})
        s.OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=s.OUT) as temp:
            path = Path(temp) / 'synthetic-params.json'
            self.f.write_params(path, args, data, [], 'test')
            params = json.loads(path.read_text())['candidate_detection']
        self.assertEqual(params['nearby_stronger_factor'], 4.)
        self.assertEqual(params['nearby_stronger_rule'], 'both_shift_metrics_within_min_separation_before_first_valid')

    def test_invalid_parameters_fail(self):
        for factor in (-1., .5, 1., float('nan')):
            with self.assertRaises(ValueError):
                self.qualify(factor=factor)
        for key in ('interpeak_report_rise_frac', 'forced_valley_report_floor'):
            with self.assertRaises(ValueError):
                self.synthetic(**{key: 1.01})
