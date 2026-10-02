"""Saved-parameter FBRM publication replay; no RAM writes or detector execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_publication_rendering import r, ROOT
from synthetic_fixture import fixture
import saved_fbrm_replay as replay


class FbrmPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root)
        self.base = self.root/'20990101_ex_99/tl/roi/rgb/analysis'
        (ROOT/'generated/publication-tests').mkdir(parents=True, exist_ok=True)
        self.output = Path(tempfile.mkdtemp(prefix='fbrm-', dir=ROOT/'generated/publication-tests'))
        self.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}

    def tearDown(self):
        self.assertEqual(self.hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.hashes})
        r.plt.close('all')

    def args(self, base=None, extra=()):
        base = base or self.base
        return r.parser().parse_args(['fbrm','--data',str(base/'ts-fbrm-tr.csv'),
            '--onsets-file',str(base/'fbrm_onsets/fbrm-onsets.csv'),
            '--candidates-file',str(base/'fbrm_onsets/fbrm-onset-candidates.csv'),
            '--params-file',str(base/'fbrm_onsets/fbrm-onset-params.json'),
            '--name','synthetic-fbrm','--output-dir',str(self.output),'--view','full',*extra])

    def test_replay_style_limits_and_force(self):
        args = self.args(extra=('--onsets','text','--points','show'))
        saved = r.load_fbrm(args)
        self.assertEqual(len(saved.fbrm.x), 50)
        np.testing.assert_allclose(saved.fbrm.y, np.arange(50)*200, atol=1e-8)
        onsets = copy.deepcopy(saved.onsets)
        for limited in (False, True):
            if limited:
                args.view, args.x_range, args.fbrm_y_lower = 'limited', [54,57], 0
            fig,_ = r.build_figure(saved,args)
            ax = fig.axes[0]
            self.assertEqual(len(fig.axes),1)
            np.testing.assert_array_equal(ax.lines[0].get_xdata(),saved.fbrm.x)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(),saved.fbrm.y)
            self.assertEqual(ax.get_xlabel(),'Tr [°C]')
            self.assertEqual(ax.get_ylabel(),'FBRM Total Counts [counts/s]')
            self.assertEqual(ax.get_title(loc='left'),'')
            fig.canvas.draw()
            box = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
            self.assertGreaterEqual(box.y0, 0)
            self.assertLessEqual(box.y1, fig.bbox.height)
            self.assertAlmostEqual(fig.get_size_inches()[0]*25.4,83.5)
            self.assertEqual(ax.xaxis.label.get_fontsize(),8)
            self.assertEqual(ax.lines[0].get_linewidth(),.725)
            self.assertFalse(any(l.get_visible() for l in ax.get_xgridlines()+ax.get_ygridlines()))
            self.assertTrue(any('T1 = 56.00' in t.get_text() for t in ax.texts))
            self.assertTrue(any(list(line.get_xdata())==[56.,56.] for line in ax.lines))
            self.assertEqual(saved.onsets,onsets)
        image, manifest_path = r.render(args)
        manifest=json.loads(manifest_path.read_text())
        self.assertEqual(manifest['replay']['saved_sample_checks'],2)
        self.assertEqual(manifest['replay']['rendered_region'],'saved analysis prefix')
        self.assertEqual(manifest['dpi'],600)
        with self.assertRaisesRegex(ValueError,'overwrite'):r.render(args)
        other=self.output/'unrelated.txt';other.write_text('keep')
        args.force=True;args.fbrm_y_upper=12000
        r.render(args)
        self.assertEqual(other.read_text(),'keep')
        self.assertEqual(json.loads(manifest_path.read_text())['axes'][0]['ylim'],[0,12000])

    def test_mismatch_rejected_and_detector_not_imported(self):
        args=self.args()
        with patch('importlib.util.spec_from_file_location',side_effect=AssertionError('No dynamic module imports')):
            r.load_fbrm(args)
        smooth, window = replay.production_smoothing(replay.Sources())
        self.assertNotIn('detect_onsets',smooth.__globals__)
        self.assertNotIn('read_data',smooth.__globals__)
        self.assertEqual({k for k,v in smooth.__globals__.items() if callable(v)},
                         {'savgol_filter','smooth_array','adjusted_savgol_window'})
        with patch.object(replay.Sources,'rows',autospec=True,side_effect=self.bad_rows):
            with self.assertRaisesRegex(ValueError,'sample smooth.*mismatch'):r.load_fbrm(args)
        self.assertEqual(list(self.output.iterdir()),[])

    original_rows = replay.Sources.rows
    def bad_rows(self, sources, path, required=()):
        rows=self.original_rows.__func__(sources,path,required)
        if Path(path).name=='fbrm-onset-candidates.csv':rows[0]['total_counts_smooth']='99999'
        return rows
