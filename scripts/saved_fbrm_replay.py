"""Verified saved-result FBRM replay shared by validation and publication.

Extracted unchanged from generate-onset-validation.py. Only the two pure SG
helpers are compiled from production source; no detector module is imported.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FBRM_SCRIPT = Path(__file__).resolve().parent / 'fbrm-onsets.py'

class Sources:
    """Capture exact source bytes before plotting and verify them afterward."""
    def __init__(self):
        self.data = {}
        self.frozen = False

    def add(self, path):
        path = Path(path).resolve()
        if path not in self.data:
            if not path.is_file():
                raise FileNotFoundError(path)
            if self.frozen:
                raise RuntimeError(f'Unregistered source after generation began: {path}')
            self.data[path] = path.read_bytes()
        return path

    def digest(self, path):
        return hashlib.sha256(self.data[self.add(path)]).hexdigest()

    def rows(self, path, required=()):
        reader = csv.DictReader(io.StringIO(self.data[self.add(path)].decode('utf-8-sig')))
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path.name}: missing columns {required}')
        return list(reader)

    def frame(self, path):
        return pd.read_csv(io.BytesIO(self.data[self.add(path)]))

    def json(self, path):
        return json.loads(self.data[self.add(path)])

    def hashes(self):
        return {str(p): hashlib.sha256(v).hexdigest() for p, v in sorted(self.data.items())}

    def verify(self):
        changed = [str(p) for p, value in self.data.items() if p.read_bytes() != value]
        if changed:
            raise RuntimeError('SOURCE MODIFICATION: ' + '; '.join(changed))


def finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f'Nonfinite numerical value: {value}')
    return number


def load_onsets(sources, path, family):
    columns = ('bend_index', 'temperature_C', 'value') if family == 'mgi' else ('onset_label', 'status', 'Tr_C', 'total_counts_smooth')
    rows = sources.rows(path, columns)
    onsets = []
    for row in rows:
        if family == 'fbrm' and row['status'] != 'accepted':
            continue
        label = f"T{int(row['bend_index'])}" if family == 'mgi' else row['onset_label']
        if any(o['label'] == label for o in onsets):
            raise ValueError('Duplicate saved onset labels')
        onsets.append({'label': label, 'x': finite_float(row['temperature_C'] if family == 'mgi' else row['Tr_C']),
                       'y': finite_float(row['value'] if family == 'mgi' else row['total_counts_smooth']), 'saved_row': row})
    return onsets


def production_smoothing(sources):
    """Compile only the two reviewed pure production functions, no module import."""
    tree = ast.parse(sources.data[sources.add(FBRM_SCRIPT)].decode())
    wanted = {'adjusted_savgol_window', 'smooth_array'}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in funcs} != wanted:
        raise RuntimeError('Production smoothing helpers unavailable')
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + funcs
    ns = {'np': np, 'savgol_filter': savgol_filter}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(FBRM_SCRIPT), 'exec'), ns)
    return ns['smooth_array'], ns['adjusted_savgol_window']


def replay_fbrm(sources, paths, smooth_array, adjusted_window):
    """Replay production read_data's fixed-order/prefix smoothing, NOT detection.

    Only saved input-order metadata is supported initially. Saved row numbers,
    loaded index, start sample, prefix count, cutoff sample and used SG window
    must agree. Saved onset AND candidate raw/smooth samples are numerical
    witnesses (rtol 1e-9, atol 1e-7); no failed replay is drawn as a result curve.
    """
    frame = sources.frame(paths['fbrm_input'])
    params = sources.json(paths['fbrm_params'])
    if params['order'] != 'input' or params.get('reverse_input', False):
        raise ValueError('Unsupported saved ordering; cannot guarantee replay')
    if params['x_column'] == 'Tr_regular':
        raise ValueError('Unsupported nonphysical temperature axis')
    x = pd.to_numeric(frame[params['x_column']], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(frame[params['y_column']], errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    original_rows = np.flatnonzero(mask)
    x, y = x[mask], y[mask]
    cooling, cutoff, smoothing = params['cooling_start'], params['validity_cutoff'], params['smoothing']
    matches = np.flatnonzero(original_rows == int(cooling['input_row']))
    if len(matches) != 1 or int(matches[0]) != int(cooling['loaded_index_before_trim']):
        raise ValueError('Saved cooling row/loaded index does not match input ordering')
    start = int(matches[0])
    x, y, original_rows = x[start:], y[start:], original_rows[start:]
    def equal(a, b, role):
        if not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-7):
            raise ValueError(f'Saved/replayed {role} mismatch: {a} versus {b}')
    equal(x[0], cooling['Tr_C'], 'cooling temperature')
    equal(y[0], cooling['total_counts'], 'cooling raw count')
    n = int(cutoff['analyzed_rows'])
    expected_n = len(y) if cutoff['cutoff_index'] is None else int(cutoff['cutoff_index'])
    if n != expected_n or not 0 < n <= len(y):
        raise ValueError('Inconsistent saved prefix boundary')
    if cutoff['cutoff_index'] is not None:
        if n == len(y):
            raise ValueError('Cutoff sample missing')
        equal(x[n], cutoff['cutoff_Tr_C'], 'cutoff temperature')
        equal(y[n], cutoff['cutoff_count'], 'cutoff raw count')
    method = smoothing['method']
    if method not in ('none', 'savgol'):
        raise ValueError('Unsupported saved smoothing method')
    args = SimpleNamespace(smooth_method=method, savgol_window=smoothing['savgol_window_requested'],
                           savgol_polyorder=smoothing['savgol_polyorder'])
    if method == 'savgol' and adjusted_window(n, args.savgol_window, args.savgol_polyorder) != smoothing['savgol_window_used_prefix']:
        raise ValueError('Production SG window differs from saved used window')
    smooth = np.empty_like(y)
    smooth[:n] = smooth_array(y[:n], args)
    if n < len(y):
        smooth[n:] = smooth_array(y[n:], args)
    candidates = sources.rows(paths['fbrm_candidates'], ('candidate_id', 'Tr_C', 'input_row', 'total_counts_smooth', 'accepted'))
    onsets = load_onsets(sources, paths['fbrm_result'], 'fbrm')
    lookup = {int(row): i for i, row in enumerate(original_rows)}
    witnesses = candidates + [o['saved_row'] for o in onsets]
    if not witnesses:
        raise ValueError('No saved sample witnesses to verify historical replay')
    max_error = 0.
    for row in witnesses:
        if row.get('input_row') in ('', None):
            raise ValueError('Saved sample has no input row')
        i = lookup.get(int(row['input_row']))
        if i is None:
            raise ValueError('Saved candidate is outside replayed cooling sequence')
        equal(x[i], row['Tr_C'], 'sample temperature')
        equal(y[i], row['total_counts_raw'], 'sample raw count')
        equal(smooth[i], row['total_counts_smooth'], 'sample smooth count')
        max_error = max(max_error, abs(smooth[i] - float(row['total_counts_smooth'])))
    for array in (x, y, smooth, original_rows): array.flags.writeable = False
    return {'x': x, 'raw': y, 'y': smooth, 'prefix_n': n, 'onsets': onsets, 'candidates': candidates,
            'limit': float(cutoff['max_search_count']), 'params': params,
            'replay': {'status': 'verified_saved_parameter_replay', 'saved_sample_checks': len(witnesses),
                       'maximum_saved_smooth_error': max_error, 'order': 'input',
                       'cooling_input_row': int(cooling['input_row']), 'analyzed_rows': n,
                       'smoothing': smoothing, 'implementation': str(FBRM_SCRIPT.resolve()),
                       'boundary_rule': 'smooth prefix and outside context separately; no kernel crosses cutoff',
                       'verification_scope': 'saved boundaries/window and all saved onset/candidate samples; no historical dense smooth export exists'}}
