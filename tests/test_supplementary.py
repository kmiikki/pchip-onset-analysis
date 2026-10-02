"""Independent analytic input fixtures exercise batch rendering, never research data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_publication_rendering import r, ROOT
from synthetic_fixture import fixture


def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/name)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


class SupplementaryTests(unittest.TestCase):
 def test_saved_variants_gallery_and_header_contract(self):
  batch=module('render-supplementary-series.py');gallery=module('render-supplementary-gallery.py')
  parent=ROOT/'generated/synthetic-supplementary';parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as tmp:
   home=Path(tmp);data=home/'data';fixture(data)
   a=data/'20990101_ex_99/tl/roi/rgb/analysis'
   x=np.linspace(60,40,100);y=20+np.arange(100)/5
   for branch,column in [('raw','BW'),('sg','BW_smooth')]:
    curve=a/'pchip'/f'bw-{branch}-vs-temp-pchip.csv'
    curve.write_text('Tr (°C),'+column+'\n'+''.join(f'{t},{v}\n' for t,v in zip(x,y)))
    curve.with_name(curve.stem+'-bends.csv').write_text('input_csv,method,bend_index,temperature_C,value\n'+f'{curve.name},{branch}_pchip,1,55,25\n')
   (a/'combo').mkdir()
   (a/'combo/mgi-fbrm-combo-data.csv').write_text('series,kind,Tr_C,value\nMGI,curve,60,20\nMGI,curve,40,40\nFBRM,smoothed_curve_main_panel,60,0\nFBRM,smoothed_curve_main_panel,40,9000\n')
   hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
   jobs,issues=batch.discover(data,home/'shown/figures');self.assertEqual(len(jobs),4);self.assertFalse(issues)
   self.assertEqual(batch.saved_mgi_ycol(a/'pchip/bw-sg-vs-temp-pchip.csv','sg'),'BW_smooth')
   manifests=[];axes={}
   for variant,dirname in [('with-onsets','shown'),('no-onsets','hidden')]:
    base=home/dirname;records=[]
    for index,job in enumerate(jobs,1):
     j=batch.presentation_job(job,base/'figures',variant)
     self.assertNotIn('--show-onset-labels',j['command'])
     args=r.parser().parse_args(j['command'][3:]);png,m=r.render(args);meta=json.loads(m.read_text())
     if dirname=='shown':axes[index]=meta['axes']
     else:self.assertEqual(meta['axes'],axes[index])
     records.append(j|dict(job_index=index,status='OK',output_path='figures/'+png.name,output_sha256=meta['output_sha256']))
    path=base/'manifest.json';path.write_text(json.dumps({'jobs':records}));manifests.append(path)
   gallery.build(manifests[1],manifests[0])
   self.assertEqual(json.loads((home/'hidden/review/validation.json').read_text())['items'],8)
   self.assertEqual(len(list((home/'hidden/review/thumbnails').glob('*.png'))),8)
   self.assertEqual(hashes,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes})
