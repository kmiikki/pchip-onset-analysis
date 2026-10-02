#!/usr/bin/env python3
"""Plan saved-result supplementary renders; render only with explicit --run.

Limited views belong to their original family/ROI (and MGI branch when known).
No detector is imported or invoked. FBRM full is the renderer's saved analysis
prefix, not a reconstruction of the excluded post-cutoff cooling series.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / 'scripts/render-publication-figures.py'


def slug(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', value).strip('-')


def presentation_job(job, output, variant):
    """Change only the output destination and optional onset presentation."""
    command = list(job['command'])
    if Path(command[2]).resolve() != RENDERER or '--show-onset-labels' in command:
        raise ValueError('Unsupported renderer command in saved plan')
    command[command.index('--output-dir') + 1] = str(output)
    if variant == 'no-onsets':
        command[command.index('--onsets') + 1] = 'none'
    elif variant != 'with-onsets':
        raise ValueError('Unknown presentation variant')
    return job | {'command': command, 'onsets_variant': 'hidden' if variant == 'no-onsets' else 'shown'}


def saved_mgi_ycol(curve, branch):
    """Select the unambiguous saved signal column; never transform its values."""
    with curve.open(encoding='utf-8-sig', newline='') as handle:
        columns = next(csv.reader(handle))
    candidates = [c for c in ('BW', 'BW_smooth') if c in columns]
    if 'Tr (°C)' not in columns or len(candidates) != 1 or (branch == 'raw' and candidates != ['BW']):
        raise ValueError(f'Unsupported or ambiguous saved {branch} PCHIP columns: {curve}: {columns}')
    return candidates[0]


def normalized_limits(x, bounds):
    if x is None or len(x) != 2 or any(v is None for v in x):
        raise ValueError('No complete saved x-range; not inferring missing bounds')
    x = sorted(map(float, x))
    if not all(map(math.isfinite, x)) or x[0] == x[1]:
        raise ValueError('Invalid saved x-range')
    out = {'x_range': x}
    for family, pair in bounds.items():
        lo, hi = pair
        if any(v is not None and not math.isfinite(float(v)) for v in pair):
            raise ValueError('Nonfinite saved y-range')
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError('Invalid saved y-range')
        out[family + '_y_lower'] = lo
        out[family + '_y_upper'] = hi
    return out


def discover(data_root, output):
    jobs, issues, datasets = [], [], []
    for experiment in sorted(data_root.glob('*_ex_*')):
        match = re.fullmatch(r'\d+_ex_(\d+)', experiment.name)
        if not match:
            continue
        ex = int(match[1])
        for analysis in sorted(experiment.glob('**/rgb/analysis')):
            roi = str(analysis.parent.parent.relative_to(experiment))
            for family, branches in [('mgi', ('raw', 'sg')), ('fbrm', ('',)), ('combo', ('',))]:
                for branch in branches:
                    base = analysis / {'mgi': 'pchip', 'fbrm': 'fbrm_onsets', 'combo': 'combo'}[family]
                    if family == 'mgi':
                        stem = f'bw-{branch}-vs-temp-pchip'
                        inputs = {'curve': base / (stem + '.csv'), 'bends': base / (stem + '-bends.csv')}
                    elif family == 'fbrm':
                        inputs = {'data': analysis / 'ts-fbrm-tr.csv', 'onsets-file': base / 'fbrm-onsets.csv',
                                  'candidates-file': base / 'fbrm-onset-candidates.csv', 'params-file': base / 'fbrm-onset-params.json'}
                    else:
                        inputs = {'data': base / 'mgi-fbrm-combo-data.csv'}
                    missing = [str(p) for p in inputs.values() if not p.is_file()]
                    if missing:
                        issues.append({'experiment': ex, 'roi': roi, 'type': family, 'branch': branch, 'missing': missing})
                        continue
                    extra = []
                    if family == 'mgi':
                        # RAW measurements remain RAW even when the displayed curve is SG.
                        measurement = analysis / 'rgb-tr.csv'
                        extra = ['--branch', branch, '--ycol', saved_mgi_ycol(inputs['curve'], branch),
                                 '--points', 'show' if measurement.is_file() else 'hide', '--onsets', 'markers']
                        if measurement.is_file():
                            inputs['measurements'] = measurement
                            extra += ['--measurement-ycol', 'BW', '--acquisition-extent']
                        else:
                            issues.append({'experiment': ex, 'roi': roi, 'type': family, 'warning': 'RAW points unavailable'})
                    elif family == 'fbrm':
                        extra = ['--points', 'show', '--onsets', 'text']
                    else:
                        extra = ['--onsets', 'none', '--fbrm-bw-linestyle', 'dashed']
                    datasets.append(dict(experiment=ex, roi=roi, type=family, branch=branch, inputs=inputs,
                                         base=base, extra=extra, views={}))
    # Read earlier rendering manifests only as view evidence, never as scientific input.
    manifests = sorted((ROOT / 'generated').rglob('*.manifest.json'))
    manifests += sorted(data_root.glob('*_ex_*/**/analysis/**/*.manifest.json'))
    for d in datasets:
        family = d['type']
        families = ('mgi', 'fbrm') if family == 'combo' else (family,)
        def remember(limits, source):
            key = json.dumps(limits, sort_keys=True)
            d['views'].setdefault(key, {'limits': limits, 'evidence': []})['evidence'].append(str(source))
        for path in sorted(d['base'].rglob('view-limits.json')):
            try:
                meta = json.loads(path.read_text())
                if meta.get('view_type') != 'axis-limited':
                    continue
                limits = meta['limits']
                remember(normalized_limits([limits['x']['lower'], limits['x']['upper']],
                         {f: [limits[f + '_y']['lower'], limits[f + '_y']['upper']] for f in families}), path)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append({'source': str(path), 'warning': str(exc)})
        primary = 'curve' if family == 'mgi' else 'data'
        for path in manifests:
            try:
                meta = json.loads(path.read_text())
                opts = meta.get('options', {})
                if opts.get('kind') != family or opts.get('view') != 'limited':
                    continue
                source = Path(opts.get(primary, ''))
                if not source.is_absolute():
                    source = ROOT / source
                if source.resolve() != d['inputs'][primary].resolve():
                    continue
                bounds = {f: opts.get(f + '_y_range') or [opts.get(f + '_y_lower'), opts.get(f + '_y_upper')] for f in families}
                remember(normalized_limits(opts.get('x_range'), bounds), path)
            except (ValueError, KeyError, TypeError) as exc:
                issues.append({'source': str(path), 'warning': str(exc)})
        for view in [dict(limits={}, evidence=[])] + [d['views'][k] for k in sorted(d['views'])]:
            limited = bool(view['limits'])
            mode = 'bw' if family == 'combo' else 'gray'
            name = f"ex{d['experiment']}-{slug(d['roi'])}-{family}" + (f"-{d['branch']}" if d['branch'] else '')
            name += '-limited' if limited else '-full'
            if limited:
                # Include all exact bounds to distinguish independently saved views.
                name += '-' + '-'.join(slug(k) + '-' + slug(str(v).replace('-', 'minus')) for k, v in sorted(view['limits'].items()) if v is not None)
            name += '-' + mode
            command = [sys.executable, '-B', str(RENDERER), family, '--name', name, '--output-dir', str(output),
                       '--view', 'limited' if limited else 'full', '--mode', mode, '--width-mm', '83.5']
            for flag, path in d['inputs'].items():
                command += ['--' + flag, str(path)]
            command += d['extra']
            for flag, value in view['limits'].items():
                if value is not None:
                    command += ['--' + flag.replace('_', '-')] + [str(v) for v in (value if isinstance(value, list) else [value])]
            jobs.append({k: d[k] for k in ('experiment', 'roi', 'type', 'branch')} | dict(
                view='limited' if limited else 'full', mode=mode, inputs={k: str(p) for k, p in d['inputs'].items()},
                limits=view['limits'], view_evidence=view['evidence'], output=name + '.png', command=command,
                note='Full saved analysis prefix only' if family == 'fbrm' else ''))
    names = [j['output'] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError('Ambiguous output names; refusing to overwrite or merge ROI results')
    return jobs, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--plan', type=Path, help='Use an authoritative saved plan instead of discovering views')
    parser.add_argument('--presentation-variant', choices=('with-onsets', 'no-onsets'), default='with-onsets')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated/supplementary' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--run', action='store_true', help='Render the listed selection; default is dry-run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--experiments', nargs='+', type=int)
    parser.add_argument('--types', nargs='+', choices=('mgi', 'fbrm', 'combo'))
    parser.add_argument('--view', choices=('full', 'limited'))
    parser.add_argument('--branch', choices=('raw', 'sg'))
    parser.add_argument('--max-jobs', type=int, help='Explicitly bound a smoke run')
    args = parser.parse_args()
    if args.run and args.dry_run:
        parser.error('Choose --run or --dry-run')
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / 'generated').resolve()):
        parser.error('Output must stay under repository generated/')
    if args.plan:
        saved = json.loads(args.plan.read_text())
        jobs, issues = saved['jobs'], saved.get('issues', [])
    else:
        jobs, issues = discover(args.data_root.resolve(), output)
    jobs = [presentation_job(j, output, args.presentation_variant) for j in jobs]
    jobs = [j for j in jobs if (not args.experiments or j['experiment'] in args.experiments)
            and (not args.types or j['type'] in args.types) and (not args.view or j['view'] == args.view)
            and (not args.branch or j['type'] != 'mgi' or j['branch'] == args.branch)]
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            parser.error('--max-jobs must be positive')
        jobs = jobs[:args.max_jobs]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'plan.json').open('x') as handle:
        json.dump({'jobs': jobs, 'issues': issues,
                   'batch_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'renderer_sha256': hashlib.sha256(RENDERER.read_bytes()).hexdigest()}, handle, indent=2)
    writer = csv.writer(sys.stdout, delimiter='\t')
    writer.writerow(['experiment', 'ROI', 'type', 'branch', 'view', 'mode', 'inputs', 'limits', 'output'])
    for j in jobs:
        writer.writerow([j[k] if k not in ('inputs', 'limits') else json.dumps(j[k]) for k in
                         ('experiment', 'roi', 'type', 'branch', 'view', 'mode', 'inputs', 'limits', 'output')])
    print(f'{len(jobs)} planned figures; {len(issues)} issues. Plan: {output / "plan.json"}', file=sys.stderr)
    if args.run:
        with (output / 'render.log').open('x') as log:
            for job in jobs:
                log.write(json.dumps(job['command']) + '\n')
                log.flush()
                subprocess.run(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
