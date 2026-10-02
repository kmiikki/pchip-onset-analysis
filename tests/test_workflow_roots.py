"""Workflow command/discovery tests only: never execute scientific children."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('workflow_roots', ROOT / 'scripts/run-onset-workflow.py')
w = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = w
spec.loader.exec_module(w)
EXPERIMENT = '20990101_ex_99'


def args(*extra):
    with patch.object(sys, 'argv', ['workflow', '--project-root', '/repo/project', *map(str, extra)]):
        return w.parse_args()


def discover_virtual(options):
    with patch.object(w, 'iter_experiments', return_value=[options.data_root / EXPERIMENT]) as discover:
        exp = w.discover_experiments(options)[0]
    discover.assert_called_once_with(options.data_root.resolve())
    return exp


def option(command, flag):
    return command[command.index(flag)+1]


class WorkflowRootsTests(unittest.TestCase):
    def setUp(self):
        self.args = args('--data-root', '/data/pchip')
        self.exp = discover_virtual(self.args)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def branch(self, branch, options=None):
        options = options or self.args
        return w.build_pchip_branch_command(options, self.exp, branch=branch,
            template=options.pchip_template if branch == 'raw' else options.pchip_sg_template)

    def test_raw_uses_repository_script_data_input_and_BW(self):
        command = self.branch('raw')
        self.assertEqual(command[1], '/repo/project/scripts/xy-pchip.py')
        self.assertEqual(option(command, '--csv'), str(self.exp.analysis_dir / 'rgb-tr.csv'))
        self.assertEqual(option(command, '--ycol'), 'BW')
        self.assertEqual(option(command, '--outstem'), str(self.exp.pchip_dir / 'bw-raw-vs-temp-pchip'))

    def test_sg_keeps_BW_smooth_and_repository_script(self):
        command = self.branch('sg')
        self.assertEqual(command[1], '/repo/project/scripts/xy-pchip.py')
        self.assertEqual(option(command, '--csv'), str(self.exp.analysis_dir / 'rgb-tr-sg.csv'))
        self.assertEqual(option(command, '--ycol'), 'BW_smooth')
        self.assertEqual(option(command, '--outstem'), str(self.exp.pchip_dir / 'bw-sg-vs-temp-pchip'))
        self.args.pchip_ycol = 'RAW_override'
        self.assertEqual(option(self.branch('sg'), '--ycol'), 'BW_smooth')

    def test_mgi_uses_active_script_and_both_data_inputs(self):
        paths = [self.exp.pchip_dir / f'bw-{b}-vs-temp-pchip.csv' for b in ('raw', 'sg')]
        with patch.object(w, 'available_mgi_pchip_csvs', return_value=paths):
            command = w.build_step_command(self.args, self.exp, 'mgi')
        self.assertEqual(command[1], '/repo/project/scripts/bw-pchip-onsets-t1t2.py')
        self.assertEqual(command[2:4], list(map(str, paths)))
        self.assertEqual(option(command, '--raw-csv'), str(self.exp.rgb_tr_csv))
        self.assertEqual(option(command, '--outdir'), str(self.exp.pchip_dir))
        self.assertEqual(w.get_onset_config_for_experiment(self.args, self.exp).path,
                         self.exp.analysis_dir / 'onset-config.ini')

    def test_all_default_children_and_cleanup_use_code_bin_and_data_targets(self):
        commands = [self.branch(b) for b in ('raw', 'sg')]
        commands += [w.build_step_command(self.args, self.exp, step) for step in ('mgi','fbrm','combo')]
        commands.append(w.build_gallery_command(self.args))
        self.assertEqual(commands[-1][2], '/data/pchip')
        self.assertEqual(option(commands[-3], '--input'), str(self.exp.ts_fbrm_tr_csv))
        self.assertEqual(option(commands[-3], '--output-dir'), str(self.exp.fbrm_onsets_dir))
        self.assertEqual(commands[-2][2], str(self.exp.analysis_dir))
        self.args.clean_mode = 'archive'
        with patch.object(w, 'ensure_script_exists', return_value=self.args.project_root/'scripts/clean-onset-outputs.py'), \
             patch.object(w, 'run_command', return_value=('dry-run', None, 0., 'not executed')) as run:
            w.run_clean_if_requested(self.args, project_root=self.args.project_root,
                                    experiments=[self.exp], log_file=io.StringIO(), results=[])
        cleanup = run.call_args.args[0]
        self.assertEqual(option(cleanup, '--project-root'), '/data/pchip')
        commands.append(cleanup)
        for command in commands:
            self.assertTrue(command[1].startswith('/repo/project/scripts/'))
            self.assertNotIn('/data/pchip/scripts/', w.command_to_string(command))

    def test_archive_cleanup_accepts_separate_data_tree_and_preserves_inputs(self):
        (ROOT / 'generated').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'generated') as temp:
            data = Path(temp)
            analysis = data / EXPERIMENT / 'tl/roi/rgb/analysis'
            output = analysis / 'pchip/result.csv'
            output.parent.mkdir(parents=True)
            output.write_text('synthetic cleanup fixture\n')
            source = analysis / 'rgb-tr.csv'
            source.write_text('synthetic input; never analyze\n')
            options = args('--project-root', ROOT, '--data-root', data,
                           '--experiments', EXPERIMENT, '--steps', 'mgi',
                           '--clean-mode', 'archive', '--run', '--python', sys.executable)
            experiments = w.discover_experiments(options)
            cleaner = ROOT / 'scripts/clean-onset-outputs.py'
            # The standalone safety guard must still reject a tree without scripts/.
            guarded = subprocess.run([sys.executable, '-B', str(cleaner),
                                      '--dry-run', '--project-root', str(data)],
                                     capture_output=True, text=True)
            self.assertEqual(guarded.returncode, 2)
            results = []
            self.assertTrue(w.run_clean_if_requested(
                options, project_root=ROOT, experiments=experiments,
                log_file=io.StringIO(), results=results))
            self.assertEqual(results[0].returncode, 0)
            self.assertFalse(output.exists())
            archived = list((data / '_cleanup_archive').rglob('result.csv'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(), 'synthetic cleanup fixture\n')
            self.assertEqual(source.read_text(), 'synthetic input; never analyze\n')

    def test_omitted_data_root_preserves_single_tree(self):
        options = args()
        self.assertEqual(options.data_root, options.project_root)
        exp = discover_virtual(options)
        command = w.build_step_command(options, exp, 'pchip')
        self.assertEqual(command[1], '/repo/project/scripts/xy-pchip.py')
        self.assertEqual(option(command, '--csv'), f'/repo/project/{EXPERIMENT}/tl/roi/rgb/analysis/rgb-tr.csv')
        self.assertEqual(w.build_gallery_command(options)[2], '/repo/project')

    def test_explicit_data_template_and_spaces_are_quoted(self):
        self.args.data_root = Path('/data/with spaces')
        self.args.project_root = Path('/repo/with spaces')
        self.args.pchip_outstem = '{data_root}/custom/{name}'
        command = self.branch('raw')
        self.assertEqual(command[1], '/repo/with spaces/scripts/xy-pchip.py')
        self.assertEqual(option(command, '--outstem'), '/data/with spaces/custom/' + EXPERIMENT)
        self.assertEqual(w.build_gallery_command(self.args)[2], '/data/with spaces')

    def test_separate_tree_discovery_symlink_and_dry_run_no_data_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'repository'
            data = Path(temp) / 'data'
            (project/'scripts').mkdir(parents=True)
            # Child files must never be executed by the dry run.
            for name in ('xy-pchip.py','bw-pchip-onsets-t1t2.py','fbrm-onsets.py',
                         'make-mgi-fbrm-combo.py','make-onset-gallery.py','onset_config.py'):
                (project/'scripts'/name).write_text('raise RuntimeError("must not execute")\n')
            analysis = data / EXPERIMENT / 'tl/roi/rgb/analysis'
            analysis.mkdir(parents=True)
            for name in ('rgb-tr.csv','rgb-tr-sg.csv','ts-fbrm-tr.csv'):
                (analysis/name).write_text('synthetic fixture; no numerical processing permitted\n')
            # A reference bin must never be selected; code-tree experiments must be ignored.
            (data/'bin').mkdir()
            (project/'20990101_ex_99').mkdir()
            (project/'ram').symlink_to(data, target_is_directory=True)
            before = {str(p): p.read_bytes() for p in data.rglob('*') if p.is_file()}
            options = args('--project-root', project, '--data-root', project/'ram', '--steps', 'pchip,mgi',
                           '--experiments', EXPERIMENT, '--force')
            exp = w.discover_experiments(options)[0]
            self.assertEqual(exp.analysis_dir, analysis)
            self.assertEqual(exp.onset_config_ini, analysis/'onset-config.ini')
            self.assertTrue(w.looks_like_project_root(project))
            with patch.object(w, 'parse_args', return_value=options), \
                 patch.object(w.subprocess, 'run', side_effect=AssertionError('dry-run executed a child')):
                self.assertEqual(w.main(), 0)
            self.assertEqual(before, {str(p): p.read_bytes() for p in data.rglob('*') if p.is_file()})
            self.assertFalse((analysis/'pchip').exists())
            self.assertFalse(exp.onset_config_ini.exists())
            self.assertIn(f'Data root: {data}', self.output.getvalue())
            self.assertIn(str(project/'scripts/xy-pchip.py'), self.output.getvalue())
            self.assertNotIn(str(data/'scripts/xy-pchip.py'), self.output.getvalue())
            self.assertTrue(list((project/'workflow_logs').glob('*.log')))

    def test_invalid_data_root_rejected_before_discovery_or_execution(self):
        options = args('--project-root', ROOT, '--data-root', ROOT/'does-not-exist-root-test')
        with patch.object(w, 'parse_args', return_value=options), \
             patch.object(w, 'discover_experiments', side_effect=AssertionError('unexpected discovery')), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(w.main(), 2)
        self.assertIn('data root is not a directory', error.getvalue())


if __name__ == '__main__':
    unittest.main()
