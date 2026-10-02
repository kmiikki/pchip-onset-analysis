"""Purely synthetic linear samples; no measurements or research results."""
import numpy as np
import pandas as pd
import json
def fixture(root, rois=('roi',), variants=False):
    """Known linear saved curve, with a separate high-count context segment."""
    for roi in rois:
        p = root/'20990101_ex_99'/'tl'/roi/'rgb/analysis'
        (p/'pchip').mkdir(parents=True)
        (p/'fbrm_onsets').mkdir()
        x = 60 - np.arange(100)/5
        y = np.arange(100)*200.
        pd.DataFrame({'Tr':x, 'counts':y}).to_csv(p/'ts-fbrm-tr.csv',index=False)
        pd.DataFrame({'Tr (°C)':x, 'BW':y/1000}).to_csv(p/'rgb-tr.csv',index=False)
        raw = p/'pchip/bw-raw-vs-temp-pchip.csv'
        pd.DataFrame({'temperature_C':x, 'value':y/1000}).to_csv(raw,index=False)
        (p/'pchip/bw-sg-vs-temp-pchip.csv').write_bytes(raw.read_bytes())
        pd.DataFrame([{'bend_index':1, 'temperature_C':56., 'value':4., 'input_csv':raw.name},
                      {'bend_index':2, 'temperature_C':54., 'value':6., 'input_csv':raw.name}]).to_csv(
            p/'pchip/bw-raw-vs-temp-pchip-bends.csv',index=False)
        pd.DataFrame([{'candidate_id':1, 'temperature_C':56., 'value':4., 'accepted':True},
                      {'candidate_id':2, 'temperature_C':55., 'value':5., 'accepted':False}]).to_csv(
            p/'pchip/bw-raw-vs-temp-pchip-candidates.csv',index=False)
        onset = {'onset_label':'T1', 'status':'accepted', 'Tr_C':56.,
                 'input_row':20, 'total_counts_raw':4000., 'total_counts_smooth':4000.}
        pd.DataFrame([onset]).to_csv(p/'fbrm_onsets/fbrm-onsets.csv',index=False)
        pd.DataFrame([onset | {'candidate_id':1, 'accepted':True}]).to_csv(
            p/'fbrm_onsets/fbrm-onset-candidates.csv',index=False)
        params = {'order':'input', 'x_column':'Tr', 'y_column':'counts',
                  'cooling_start':{'input_row':0, 'loaded_index_before_trim':0, 'Tr_C':60., 'total_counts':0.},
                  'validity_cutoff':{'analyzed_rows':50, 'cutoff_index':50, 'cutoff_Tr_C':50.,
                                     'cutoff_count':10000., 'max_search_count':10000.},
                  'smoothing':{'method':'savgol', 'savgol_window_requested':5,
                               'savgol_window_used_prefix':5, 'savgol_polyorder':2},
                  'script':'historical-b' if variants and roi != 'roi' else 'historical-a'}
        (p/'fbrm_onsets/fbrm-onset-params.json').write_text(json.dumps(params))
