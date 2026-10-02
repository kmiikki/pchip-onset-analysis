"""Focused saved-result regression tests; no detector or RAM writes.

Run: python -B -m unittest discover -s tests -p test_publication_rendering.py -v
Figure/manifest outputs remain in generated/publication-tests/<unique run>/.
Synthetic input CSVs exist only in a temporary directory outside the repository.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("publication_renderer", ROOT / "scripts/render-publication-figures.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


class PublicationRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.source_dir = Path(cls.temp.name)
        output_parent = ROOT / "generated/publication-tests"
        output_parent.mkdir(parents=True, exist_ok=True)
        cls.output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_parent))
        cls.curve = cls.source_dir / "raw.csv"
        cls.bends = cls.source_dir / "bends.csv"
        cls.measurements = cls.source_dir / "measurements.csv"
        cls.combo = cls.source_dir / "combo.csv"
        cls.curve.write_text("Tr (°C),BW\n60,10.123456789012345\n55,12\n50,30\n45,80\n40,100\n")
        cls.bends.write_text("input_csv,method,bend_index,temperature_C,value\nraw.csv,raw_pchip,1,52.3456,23.4567\n")
        cls.measurements.write_text("Tr (°C),BW\n61,11\n56,13\n51,29\n46,82\n39,99\n")
        cls.combo.write_text("series,kind,Tr_C,value\n"
                             "MGI,curve,60,10.123456789012345\n"
                             "FBRM,raw_curve_main_panel,60,999\n"
                             "FBRM,smoothed_curve_main_panel,60,100\n"
                             "MGI,curve,50,30\n"
                             "FBRM,smoothed_curve_main_panel,50,5000\n"
                             "FBRM,smoothed_curve_main_panel,50.2,5100\n"
                             "MGI,curve,40,100\n"
                             "FBRM,smoothed_curve_main_panel,40,18000\n"
                             "FBRM,normalized_abs_derivative_full_sg_cooling_curve,50,1\n")
        cls.source_hashes = cls.hashes()
        print(f"\nRetained publication test outputs: {cls.output_dir}")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def hashes(cls):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.source_dir.iterdir() if p.is_file()}

    def tearDown(self):
        self.assertEqual(self.hashes(), self.source_hashes)
        r.plt.close("all")

    def args(self, kind="mgi", name="test", extra=()):
        tokens = [kind, "--name", name, "--view", "full", "--output-dir", str(self.output_dir)]
        if kind == "mgi":
            tokens += ["--curve", str(self.curve), "--bends", str(self.bends), "--branch", "raw", "--points", "hide"]
        elif kind == "combo":
            tokens += ["--data", str(self.combo)]
        return r.parser().parse_args(tokens + list(extra))

    def test_mgi_exact_saved_arrays_and_view_invariance(self):
        args = self.args(extra=("--onsets", "markers"))
        saved = r.load_mgi(args)
        with self.curve.open() as stream:
            expected = np.array([(float(row["Tr (°C)"]), float(row["BW"])) for row in csv.DictReader(stream)])
        np.testing.assert_array_equal(saved.mgi.x, expected[:, 0])
        np.testing.assert_array_equal(saved.mgi.y, expected[:, 1])
        original_onsets = copy.deepcopy(saved.onsets)
        full, _ = r.build_figure(saved, args)
        args.view, args.x_range = "limited", [45, 55]
        detail, _ = r.build_figure(saved, args)
        for fig in (full, detail):
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_xydata(), expected)
            self.assertIsNone(fig.axes[0].get_legend())
        self.assertEqual(saved.onsets, original_onsets)
        self.assertEqual(saved.onsets[0]["temperature_C"], 52.3456)
        self.assertEqual(saved.onsets[0]["value"], 23.4567)
        self.assertFalse(saved.mgi.x.flags.writeable)
        np.testing.assert_array_equal(saved.mgi.y, expected[:, 1])
        self.assertEqual(detail.axes[0].get_xlim(), (55, 45))
        for text in detail.axes[0].texts:
            self.assertEqual(text.get_fontsize(), 8)
            self.assertEqual(text.get_bbox_patch().get_edgecolor()[3], 0)

    def test_locked_publication_labels_legends_and_layout(self):
        for kind, points in (("mgi", "hide"), ("mgi", "show"), ("combo", None)):
            with self.subTest(kind=kind, points=points):
                args = self.args(kind)
                if points == "show":
                    args.points = points
                    args.measurements = self.measurements
                    args.measurement_ycol = "BW"
                saved = r.load_mgi(args) if kind == "mgi" else r.load_combo(args)
                fig, _ = r.build_figure(saved, args)
                self.assertEqual(fig.axes[0].get_xlabel(), "Tr [°C]")
                self.assertEqual(fig.axes[0].get_ylabel(), "MGI")
                self.assertAlmostEqual(fig.get_size_inches()[0] * 25.4, 83.5)
                for axis in fig.axes:
                    self.assertFalse(any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines()))
                    self.assertEqual(axis.xaxis.label.get_fontsize(), 8)
                    self.assertTrue(all(t.get_fontsize() == 8 for t in axis.get_xticklabels() + axis.get_yticklabels()))
                if kind == "mgi":
                    self.assertIsNone(fig.axes[0].get_legend())
                    self.assertAlmostEqual(fig.axes[0].lines[0].get_linewidth(), 1.45 * .5)
                else:
                    self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ["MGI", "FBRM"])
                    for axis, color in zip(fig.axes, ("black", "0.40")):
                        self.assertEqual(len(axis.lines), 1)
                        self.assertEqual(axis.lines[0].get_linestyle(), "-")
                        self.assertEqual(axis.lines[0].get_color(), color)
                        self.assertAlmostEqual(axis.lines[0].get_linewidth(), 1.6 * .5)
                r.plt.close(fig)

    def test_combo_exact_arrays_full_counts_and_view_invariance(self):
        args = self.args("combo")
        saved = r.load_combo(args)
        with self.combo.open() as stream:
            rows = list(csv.DictReader(stream))
        for curve, series, kind in ((saved.mgi, "MGI", "curve"),
                                    (saved.fbrm, "FBRM", "smoothed_curve_main_panel")):
            expected = np.array([(float(row["Tr_C"]), float(row["value"])) for row in rows
                                 if row["series"] == series and row["kind"] == kind])
            np.testing.assert_array_equal(np.column_stack((curve.x, curve.y)), expected)
        full, _ = r.build_figure(saved, args)
        args.view, args.x_range = "limited", [48, 53]
        detail, _ = r.build_figure(saved, args)
        self.assertGreater(full.axes[1].get_ylim()[1], 18000)
        for fig in (full, detail):
            for axis, curve in zip(fig.axes, (saved.mgi, saved.fbrm)):
                self.assertEqual(len(axis.lines), 1)  # no markers or raw series
                self.assertEqual(len(axis.collections), 0)
                self.assertEqual(axis.lines[0].get_linestyle(), "-")
                np.testing.assert_array_equal(axis.lines[0].get_xdata(), curve.x)
                np.testing.assert_array_equal(axis.lines[0].get_ydata(), curve.y)
        self.assertEqual(saved.onsets, ())
        self.assertTrue(np.any(np.diff(saved.fbrm.x) > 0))  # Preserve the loop.

    def test_combo_physical_style_survives_host_settings_and_png_export(self):
        with r.plt.rc_context({'font.size': 22, 'font.weight': 'bold',
                               'lines.marker': 'o', 'savefig.bbox': 'tight',
                               'path.simplify': True}):
            for width in (83.5, 167):
                args = self.args('combo', extra=('--width-mm', str(width)))
                saved = r.load_combo(args)
                fig, _ = r.build_figure(saved, args)
                np.testing.assert_allclose(fig.get_size_inches()*25.4, [width, width*5.5/9])
                self.assertGreater(fig.subplotpars.left, 0)
                for axis in fig.axes:
                    self.assertEqual(axis.yaxis.label.get_fontsize(), 8)
                    self.assertEqual(axis.yaxis.label.get_fontweight(), 'normal')
                    self.assertAlmostEqual(axis.lines[0].get_linewidth(), .8)
                    self.assertEqual(axis.lines[0].get_marker(), 'None')
                    self.assertFalse(axis.lines[0].get_path().should_simplify)
                self.assertTrue(all(t.get_fontsize()==8 for t in fig.axes[0].get_legend().texts))
                png = r.encode_png(fig, 'gray')
                pixels = struct.unpack('>II', png[16:24])
                expected = (width/25.4*600, width/25.4*5.5/9*600)
                for actual, wanted in zip(pixels, expected):
                    self.assertLess(abs(actual-wanted), 1.1)
                self.assertEqual(png[24:26], bytes([8,0]))
                r.plt.close(fig)

    def test_png_modes_dpi_manifest_and_source_preservation(self):
        for mode in ("gray", "bw"):
            args = self.args(name=f"mgi-{mode}", extra=("--mode", mode, "--onsets", "markers"))
            image, manifest_path = r.render(args)
            raw = image.read_bytes()
            self.assertEqual(raw[24:26], bytes([1 if mode == "bw" else 8, 0]))
            pos, resolutions = 8, []
            while pos < len(raw):
                size = struct.unpack(">I", raw[pos:pos+4])[0]
                if raw[pos+4:pos+8] == b"pHYs":
                    resolutions.append(struct.unpack(">IIB", raw[pos+8:pos+8+size]))
                pos += size + 12
            self.assertEqual(len(resolutions), 1)
            xppm, yppm, unit = resolutions[0]
            self.assertEqual(unit, 1)
            self.assertEqual(xppm, yppm)
            self.assertAlmostEqual(xppm * .0254, 1200 if mode == "bw" else 600, delta=.02)
            decoded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            self.assertEqual(decoded.ndim, 2)
            if mode == "bw":
                self.assertEqual(set(np.unique(decoded)), {0, 255})
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["branch"], "raw")
            self.assertEqual(manifest["accepted_onsets"][0]["temperature_C"], 52.3456)
            self.assertEqual(len(manifest["git_commit"]), 40)
            for source in manifest["sources"]:
                self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                r.render(args)

    def test_combo_render_and_gray_measurements(self):
        r.render(self.args("combo", "combo-gray"))
        args = self.args(name="mgi-points", extra=("--points", "show", "--measurements", str(self.measurements),
                                                   "--measurement-ycol", "BW", "--acquisition-extent"))
        result = r.load_mgi(args)
        fig, _ = r.build_figure(result, args)
        self.assertEqual(fig.axes[0].get_xlim(), (61, 39))
        np.testing.assert_array_equal(result.mgi.y, r.load_mgi(self.args()).mgi.y)
        r.render(args)

    def test_combo_accepted_text_is_saved_and_never_markers(self):
        args = self.args("combo", extra=("--mgi-bends", str(self.bends), "--onsets", "text"))
        result = r.load_combo(args)
        before = copy.deepcopy(result.onsets)
        args.view, args.x_range = "limited", [48, 55]
        fig, _ = r.build_figure(result, args)
        self.assertEqual(result.onsets, before)
        self.assertEqual(result.onsets[0]["temperature_C"], 52.3456)
        self.assertIn("MGI T1", fig.axes[0].texts[0].get_text())
        self.assertEqual([len(axis.lines) for axis in fig.axes], [1, 1])

    def test_optional_panels_geometry_style_and_saved_arrays(self):
        args = self.args("combo", "two-panel", extra=(
            "--view", "panels", "--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45",
            "--mgi-bends", str(self.bends)))
        saved = r.load_combo(args)
        before = copy.deepcopy(saved.onsets)
        fig, _ = r.build_figure(saved, args)
        np.testing.assert_allclose(fig.get_size_inches()*25.4, [167, 167 / 2 * 5.5 / 9])
        self.assertEqual(len(fig.axes), 4)
        for i, limits in enumerate(((60, 25), (49, 45))):
            ax, right = fig.axes[i], fig.axes[i+2]
            self.assertEqual(ax.get_xlim(), limits)
            self.assertEqual(ax.get_xlabel(), "Tr [°C]")
            self.assertEqual(ax.get_title(loc="left"), ("a)", "b)")[i])
            self.assertEqual(ax._left_title.get_fontsize(), 8)
            self.assertEqual([t.get_text() for t in ax.get_legend().texts], ["MGI", "FBRM"])
            self.assertTrue(all(t.get_fontsize() == 8 for t in ax.get_legend().texts))
            for axis, curve, color in ((ax, saved.mgi, "black"), (right, saved.fbrm, "0.40")):
                self.assertEqual(len(axis.lines), 1)
                line = axis.lines[0]
                np.testing.assert_array_equal(line.get_xdata(), curve.x)
                np.testing.assert_array_equal(line.get_ydata(), curve.y)
                self.assertEqual((line.get_color(), line.get_linestyle(), line.get_marker()),
                                 (color, "-", "None"))
                self.assertEqual(line.get_linewidth(), .8)
                self.assertFalse(line.get_path().should_simplify)
                for t in axis.get_xticklabels()+axis.get_yticklabels()+[axis.xaxis.label, axis.yaxis.label]:
                    self.assertEqual(t.get_fontsize(), 8)
                self.assertFalse(any(l.get_visible() for l in axis.get_xgridlines()+axis.get_ygridlines()))
        self.assertEqual(saved.onsets, before)
        image, manifest_path = r.render(args)
        raw = image.read_bytes()
        self.assertEqual(struct.unpack(">II", raw[16:24]), (3944, 1205))
        self.assertEqual(raw[24:26], bytes([8, 0]))
        offset = raw.index(b"pHYs")
        self.assertAlmostEqual(struct.unpack(">I", raw[offset+4:offset+8])[0]*.0254, 600, delta=.02)
        manifest = json.loads(manifest_path.read_text())
        np.testing.assert_allclose(manifest["figure_size_mm"], [167, 167 / 2 * 5.5 / 9])
        self.assertEqual(manifest["accepted_onsets"], list(before))
        self.assertEqual(manifest["options"]["panel_x_ranges"], [60, 25, 49, 45])
        self.assertEqual(self.hashes(), self.source_hashes)

    def test_optional_panels_require_ranges_not_manuscript_width(self):
        with self.assertRaises(ValueError):
            r.validate(self.args("combo", extra=("--view", "panels")))
        args = self.args("combo", extra=("--view", "panels", "--width-mm", "180",
                                         "--panel-x-ranges", "60", "25", "49", "45"))
        r.validate(args)
        fig, _ = r.build_figure(r.load_combo(args), args)
        self.assertAlmostEqual(fig.get_size_inches()[0] * 25.4, 180)

    def test_combo_y_ranges_standalone_and_panels(self):
        cases = (
            ("full", (), ((0, 120), (0, 20000))),
            ("limited", ("--x-range", "49", "45"), ((0, 120), (0, 20000))),
            ("panels", ("--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45"),
             ((0, 120), (0, 120), (0, 20000), (0, 20000))),
            ("panels", ("--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45",
                        "--panel-mgi-y-ranges", "130", "10", "20", "80",
                        "--panel-fbrm-y-ranges", "0", "25000", "100", "10000"),
             ((10, 130), (20, 80), (0, 25000), (100, 10000))),
        )
        for index, (view, extra, expected) in enumerate(cases):
            with self.subTest(view=view, index=index):
                args = self.args("combo", f"y-ranges-{index}", extra=(
                    "--view", view, "--mgi-y-range", "0", "120", "--fbrm-y-range", "0", "20000",
                    "--mgi-bends", str(self.bends), *extra))
                r.validate(args)
                saved = r.load_combo(args)
                onsets = copy.deepcopy(saved.onsets)
                fig, _ = r.build_figure(saved, args)
                self.assertEqual(tuple(ax.get_ylim() for ax in fig.axes), expected)
                curves = (saved.mgi, saved.mgi, saved.fbrm, saved.fbrm) if view == "panels" else (saved.mgi, saved.fbrm)
                for ax, curve in zip(fig.axes, curves):
                    np.testing.assert_array_equal(ax.lines[0].get_xdata(), curve.x)
                    np.testing.assert_array_equal(ax.lines[0].get_ydata(), curve.y)
                self.assertEqual(saved.onsets, onsets)
                _, manifest_path = r.render(args)
                manifest = json.loads(manifest_path.read_text())
                for key in ("mgi_y_range", "fbrm_y_range", "panel_mgi_y_ranges", "panel_fbrm_y_ranges"):
                    self.assertEqual(manifest["options"][key], getattr(args, key))
                self.assertEqual([a["ylim"] for a in manifest["axes"]], [list(v) for v in expected])
                self.assertEqual(manifest["accepted_onsets"], list(onsets))

    def test_combo_y_ranges_omitted_and_independent_autoscaling(self):
        for view, extra in (("full", ()), ("limited", ("--x-range", "49", "45")),
                            ("panels", ("--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45"))):
            args = self.args("combo", extra=("--view", view, *extra))
            saved = r.load_combo(args)
            baseline, _ = r.build_figure(saved, args)
            # Derive Matplotlib's original autoscale bounds directly from the saved curves.
            reference, left = r.plt.subplots()
            right = r.draw_combo(left, saved)
            expected = [left.get_ylim(), right.get_ylim()]
            if view == "panels":
                expected = [expected[0], expected[0], expected[1], expected[1]]
            self.assertEqual([ax.get_ylim() for ax in baseline.axes], expected)
            for family in ("mgi", "fbrm"):
                changed_args = copy.deepcopy(args)
                setattr(changed_args, f"{family}_y_range", [0, 200])
                fig, _ = r.build_figure(saved, changed_args)
                for i, ax in enumerate(fig.axes):
                    is_mgi = i < len(fig.axes)//2
                    changed = is_mgi == (family == "mgi")
                    self.assertEqual(ax.get_ylim(), (0, 200) if changed else expected[i])
                    self.assertEqual(ax.get_autoscaley_on(), not changed)
                r.plt.close(fig)
            r.plt.close(reference)
            r.plt.close(baseline)

    def test_combo_y_range_validation(self):
        for family in ("mgi", "fbrm"):
            for panel in (False, True):
                flag = f"--{'panel-' if panel else ''}{family}-y-range{'s' if panel else ''}"
                context = ("--view", "panels", "--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45") if panel else ()
                for pair in (("1", "1"), ("nan", "2"), ("0", "inf")):
                    values = ("0", "10", *pair) if panel else pair
                    with self.subTest(flag=flag, pair=pair), self.assertRaisesRegex(ValueError, "finite, distinct"):
                        r.validate(self.args("combo", extra=(*context, flag, *values)))
                if panel:
                    with self.assertRaisesRegex(ValueError, "requires --view panels"):
                        r.validate(self.args("combo", extra=(flag, "0", "10", "0", "20")))

    def test_temperature_tick_policy_boundaries_and_cooling_direction(self):
        for span, step in ((2.5, .5), (2.5001, 1), (6, 1), (6.0001, 2),
                           (15, 2), (15.0001, 5), (40, 5), (40.0001, 10), (100, 10)):
            with self.subTest(span=span):
                fig, ax = r.plt.subplots()
                ax.set_xlim(60, 60-span)
                r.apply_publication_temperature_ticks(ax)
                self.assertEqual(ax.get_xlim(), (60, 60-span))
                np.testing.assert_allclose(np.diff(ax.get_xticks()), step)
                r.plt.close(fig)
        for limits, expected in (((49, 45), [45, 46, 47, 48, 49]),
                                 ((60, 25), [25, 30, 35, 40, 45, 50, 55, 60])):
            fig, ax = r.plt.subplots()
            ax.set_xlim(*limits)
            r.apply_publication_temperature_ticks(ax)
            ticks = ax.get_xticks()
            np.testing.assert_array_equal(ticks[(ticks >= limits[1]) & (ticks <= limits[0])], expected)
            self.assertEqual(ax.get_xlim(), limits)
            r.plt.close(fig)
        fig, ax = r.plt.subplots()
        ax.set_xlim(1000, 0)
        r.apply_publication_temperature_ticks(ax)
        self.assertLessEqual(len(ax.get_xticks()), 10)
        self.assertEqual(ax.get_xlim(), (1000, 0))

    def test_standalone_mgi_combo_ticks_do_not_change_saved_curves(self):
        for kind in ("mgi", "combo"):
            args = self.args(kind)
            saved = r.load_mgi(args) if kind == "mgi" else r.load_combo(args)
            onsets = copy.deepcopy(saved.onsets)
            for view, expected in (("full", [40, 45, 50, 55, 60]),
                                   ("limited", [45, 46, 47, 48, 49])):
                args.view = view
                args.x_range = [49, 45] if view == "limited" else None
                fig, _ = r.build_figure(saved, args)
                ax = fig.axes[0]
                hi, lo = ax.get_xlim()
                ticks = ax.get_xticks()
                np.testing.assert_array_equal(ticks[(ticks >= lo) & (ticks <= hi)], expected)
                self.assertGreater(hi, lo)
                for axis, curve in zip(fig.axes, (saved.mgi, saved.fbrm)):
                    np.testing.assert_array_equal(axis.lines[0].get_xdata(), curve.x)
                    np.testing.assert_array_equal(axis.lines[0].get_ydata(), curve.y)
                    self.assertTrue(all(t.get_fontsize() == 8 for t in axis.get_xticklabels()))
                self.assertEqual(saved.onsets, onsets)
                r.plt.close(fig)

    def test_legacy_fbrm_publication_style_uses_shared_temperature_ticks(self):
        spec = importlib.util.spec_from_file_location("tick_test_fbrm", ROOT / "scripts/fbrm-onsets.py")
        fbrm = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fbrm
        spec.loader.exec_module(fbrm)
        self.assertIs(fbrm.apply_publication_temperature_ticks, r.apply_publication_temperature_ticks)
        for limits, step in (((49, 45), 1), ((60, 25), 5)):
            fig, ax = r.plt.subplots()
            ax.set_xlim(*limits)
            fbrm.apply_publication_axis_style(ax)
            np.testing.assert_allclose(np.diff(ax.get_xticks()), step)
            self.assertEqual(ax.get_xlim(), limits)
            r.plt.close(fig)

    def test_independent_y_bounds_and_provenance(self):
        cases = (("lower", ("--{f}-y-lower", "0"), 0, None),
                 ("upper", ("--{f}-y-upper", "25000"), None, 25000),
                 ("both", ("--{f}-y-lower", "0", "--{f}-y-upper", "25000"), 0, 25000),
                 ("pair", ("--{f}-y-range", "0", "25000"), 0, 25000))
        for kind, family in (("mgi", "mgi"), ("combo", "mgi"), ("combo", "fbrm")):
            for view in ("full", "limited", "panels"):
                if kind == "mgi" and view == "panels":
                    continue
                context = ("--view", view)
                if view == "limited":
                    context += ("--x-range", "49", "45")
                elif view == "panels":
                    context += ("--panel-x-ranges", "60", "25", "49", "45", "--width-mm", "167")
                args = self.args(kind, extra=context)
                saved = r.load_mgi(args) if kind == "mgi" else r.load_combo(args)
                baseline, _ = r.build_figure(saved, args)
                limits = [ax.get_ylim() for ax in baseline.axes]
                for label, tokens, lower, upper in cases:
                    with self.subTest(kind=kind, family=family, view=view, case=label):
                        changed = self.args(kind, extra=(*context, *(t.format(f=family) for t in tokens)))
                        r.validate(changed)
                        fig, _ = r.build_figure(saved, changed)
                        for i, axis in enumerate(fig.axes):
                            target = i < (2 if view == "panels" else 1)
                            target = target if family == "mgi" else not target
                            lo, hi = limits[i]
                            expected = (lo if lower is None else lower, hi if upper is None else upper) if target else limits[i]
                            self.assertEqual(axis.get_ylim(), expected)
                            np.testing.assert_array_equal(axis.lines[0].get_xydata(), baseline.axes[i].lines[0].get_xydata())
                        r.plt.close(fig)
                r.plt.close(baseline)
        args = self.args("combo", "partial-y-manifest", extra=("--fbrm-y-lower", "0"))
        _, manifest_path = r.render(args)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["options"]["fbrm_y_lower"], 0)
        self.assertIsNone(manifest["options"]["fbrm_y_upper"])
        self.assertEqual(manifest["axes"][1]["ylim"][0], 0)
        self.assertGreater(manifest["axes"][1]["ylim"][1], 18000)
        args = self.args("combo", extra=("--view", "panels", "--width-mm", "167",
            "--panel-x-ranges", "60", "25", "49", "45", "--fbrm-y-lower", "0",
            "--panel-fbrm-y-ranges", "100", "20000", "200", "10000"))
        r.validate(args)
        fig, _ = r.build_figure(r.load_combo(args), args)
        self.assertEqual(fig.axes[2].get_ylim(), (100, 20000))
        self.assertEqual(fig.axes[3].get_ylim(), (200, 10000))

    def test_independent_y_bounds_validation_and_default_output(self):
        for kind, family in (("mgi", "mgi"), ("combo", "mgi"), ("combo", "fbrm")):
            for side in ("lower", "upper"):
                with self.assertRaisesRegex(ValueError, "combine"):
                    r.validate(self.args(kind, extra=(f"--{family}-y-range", "0", "100",
                                                      f"--{family}-y-{side}", "0")))
                for bad in ("nan", "inf"):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        r.validate(self.args(kind, extra=(f"--{family}-y-{side}", bad)))
            with self.assertRaisesRegex(ValueError, "less than"):
                r.validate(self.args(kind, extra=(f"--{family}-y-lower", "100", f"--{family}-y-upper", "0")))
            for flag, value in (("lower", "100000"), ("upper", "-100000")):
                args = self.args(kind, extra=(f"--{family}-y-{flag}", value))
                saved = r.load_mgi(args) if kind == "mgi" else r.load_combo(args)
                with self.assertRaisesRegex(ValueError, "Resolved"):
                    r.build_figure(saved, args)
                r.plt.close("all")
            args = self.args(kind)
            saved = r.load_mgi(args) if kind == "mgi" else r.load_combo(args)
            fig, _ = r.build_figure(saved, args)
            first = r.encode_png(fig, "gray")
            # Legacy programmatic callers lack the new attributes entirely.
            for name in list(vars(args)):
                if name.endswith(("_y_lower", "_y_upper")):
                    delattr(args, name)
            second, _ = r.build_figure(saved, args)
            self.assertEqual(first, r.encode_png(second, "gray"))
            r.plt.close("all")

    def test_combo_gray_bw_styles_and_encoding(self):
        for view in ("full", "panels"):
            for mode, style, color, line in (("gray", "dotted", "0.40", "-"),
                                            ("bw", "dashed", "black", "--"),
                                            ("bw", "dotted", "black", ":")):
                args = self.args("combo", f"styles-{view}-{mode}-{style}", extra=(
                    "--mode", mode, "--fbrm-bw-linestyle", style, "--view", view,
                    *(("--width-mm", "167", "--panel-x-ranges", "60", "25", "49", "45") if view == "panels" else ())))
                saved = r.load_combo(args)
                fig, _ = r.build_figure(saved, args)
                self.assertEqual(fig.dpi, 1200 if mode == "bw" else 600)
                for i, axis in enumerate(fig.axes):
                    mgi = i < len(fig.axes)//2
                    curve = saved.mgi if mgi else saved.fbrm
                    plotted = axis.lines[0]
                    self.assertEqual(len(axis.lines), 1)
                    self.assertEqual(plotted.get_color(), "black" if mgi else color)
                    self.assertEqual(plotted.get_linestyle(), "-" if mgi else line)
                    self.assertEqual(plotted.get_marker(), "None")
                    self.assertEqual(plotted.get_linewidth(), .8)
                    self.assertFalse(any(g.get_visible() for g in axis.get_xgridlines()+axis.get_ygridlines()))
                    self.assertTrue(all(t.get_fontsize() == 8 for t in axis.get_xticklabels()))
                    np.testing.assert_array_equal(plotted.get_xdata(), curve.x)
                    np.testing.assert_array_equal(plotted.get_ydata(), curve.y)
                r.plt.close(fig)
                image, manifest_path = r.render(args)
                raw = image.read_bytes()
                self.assertEqual(raw[24:26], bytes([1 if mode == "bw" else 8, 0]))
                offset = raw.index(b"pHYs")
                self.assertAlmostEqual(struct.unpack(">I", raw[offset+4:offset+8])[0]*.0254,
                                       1200 if mode == "bw" else 600, delta=.02)
                if mode == "bw":
                    self.assertEqual(set(np.unique(cv2.imread(str(image), cv2.IMREAD_UNCHANGED))), {0,255})
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["output_mode"], mode)
                for applied in manifest["applied_line_styles"]:
                    self.assertEqual(applied["linestyle"], "-" if applied["series"] == "MGI" else line)

    def test_force_replaces_only_named_pair_and_preserves_safety(self):
        args = self.args("combo", "force-pair")
        image, manifest_path = r.render(args)
        before = (image.read_bytes(), manifest_path.read_bytes())
        unrelated = self.output_dir / "unrelated.txt"
        unrelated.write_text("keep")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            r.render(args)
        self.assertEqual((image.read_bytes(), manifest_path.read_bytes()), before)
        args.force = True
        args.mode = "bw"
        with patch.object(r.os, "replace", side_effect=OSError("replacement blocked")):
            with self.assertRaisesRegex(OSError, "blocked"):
                r.render(args)
        self.assertEqual((image.read_bytes(), manifest_path.read_bytes()), before)
        self.assertEqual(list(self.output_dir.glob(".force-pair-*")), [])
        r.render(args)
        self.assertNotEqual(image.read_bytes(), before[0])
        manifest = json.loads(manifest_path.read_text())
        self.assertTrue(manifest["options"]["force"])
        self.assertEqual(manifest["output_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
        self.assertEqual(unrelated.read_text(), "keep")
        self.assertTrue(self.args("combo", extra=("-f",)).force)
        args.output_dir = ROOT / "final"
        with self.assertRaisesRegex(ValueError, "generated"):
            r.validate(args)
        args.output_dir = self.output_dir
        args.name = "symlink-output"
        link = self.output_dir / "symlink-output.png"
        link.symlink_to(self.combo)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            r.render(args)
        link.unlink()

    def test_generated_and_analysis_view_output_paths(self):
        self.assertEqual(r.validated_output_dir(self.output_dir), self.output_dir.resolve())
        official = r.ANALYSIS_ROOT / "synthetic/analysis/combo/views/test"
        self.assertEqual(r.validated_output_dir(official), official)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = root / "data"
            data.mkdir()
            view = data / "experiment/analysis/combo/views/test-view"
            view.mkdir(parents=True)
            with patch.object(r, "ANALYSIS_ROOT", data):
                self.assertEqual(r.validated_output_dir(view), view)
                for bad in (root, data / "experiment/views/test", data / "experiment/analysis/combo",
                            data / "experiment/analysis/combo/views", view / "extra",
                            data / "scripts/analysis/views/test"):
                    with self.subTest(path=bad), self.assertRaises(ValueError):
                        r.validated_output_dir(bad)
                outside = root / "outside/analysis/combo/views/test"
                outside.mkdir(parents=True)
                escaped = view.parent / "escape"
                escaped.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink escape"):
                    r.validated_output_dir(escaped)
                inside_nonview = view.parent / "wrong-target"
                inside_nonview.symlink_to(data, target_is_directory=True)
                with self.assertRaises(ValueError):
                    r.validated_output_dir(inside_nonview)

    def test_safety_rejections(self):
        for kind in ("mgi", "combo"):
            args = self.args(kind)
            args.output_dir = ROOT / "final"
            with self.assertRaisesRegex(ValueError, "generated"):
                r.render(args)
        args = self.args(extra=("--mode", "bw", "--points", "show"))
        with self.assertRaisesRegex(ValueError, "grayscale"):
            r.render(args)
        args = self.args(extra=("--branch", "sg"))
        with self.assertRaisesRegex(ValueError, "branch"):
            r.load_mgi(args)
        args = self.args(extra=("--measurements", str(self.measurements)))
        with self.assertRaisesRegex(ValueError, "only"):
            r.load_mgi(args)
        link = self.source_dir / "escape"
        link.symlink_to(self.source_dir, target_is_directory=True)
        args = self.args()
        args.output_dir = link
        with self.assertRaisesRegex(ValueError, "generated"):
            r.render(args)


if __name__ == "__main__":
    unittest.main()
