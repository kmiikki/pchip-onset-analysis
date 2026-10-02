import ast
import importlib.util
import inspect
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated/synthetic-tests'
sys.path.insert(0,str(ROOT/'scripts'))
def numerical_module(path,family,label):
 spec=importlib.util.spec_from_file_location(label,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[label]=mod;spec.loader.exec_module(mod)
 return mod,{n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
def options(mod):
 with patch('sys.argv',['synthetic']): args=mod.parse_args()
 result={k:getattr(args,k) for k in inspect.signature(mod.detect_bends).parameters if hasattr(args,k)}
 result.update(method='raw_pchip',report_method=args.method,max_bends=2,
               use_interpeak_valley_onsets=not args.no_interpeak_valley_onsets)
 return result
