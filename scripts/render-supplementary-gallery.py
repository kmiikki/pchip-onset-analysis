#!/usr/bin/env python3
"""Build an offline review gallery from a supplementary production manifest.

Usage: python scripts/render-supplementary-gallery.py generated/supplementary/production-.../manifest.json
Writes review/ and, when comparing two manifests, comparison-manifest.json.
Production PNGs are linked directly, never copied or changed. Keep both
production directories together when sharing a comparison. Rebuilding review
is supported.
"""
import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote
import cv2

ROOT = Path(__file__).resolve().parents[1]


def build(manifest_path, additional_manifest=None):
    manifest_path = manifest_path.resolve()
    base = manifest_path.parent
    if not base.is_relative_to((ROOT / 'generated').resolve()):
        raise ValueError('Gallery output must stay under generated/')
    jobs, figure_roots = [], []
    for source in [manifest_path] + ([additional_manifest.resolve()] if additional_manifest else []):
        if not source.parent.is_relative_to((ROOT / 'generated').resolve()):
            raise ValueError('Manifest must stay under generated/')
        figure_roots.append(source.parent / 'figures')
        for original in json.loads(source.read_text())['jobs']:
            job = original | {'onsets_variant': original.get('onsets_variant', 'shown')}
            job['output_path'] = os.path.relpath(source.parent / original['output_path'], base)
            jobs.append(job)
    identities = [(j['job_index'], j['onsets_variant']) for j in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate job/variant identity')
    if additional_manifest:
        target = base / 'comparison-manifest.json'
        if target.is_symlink():
            raise ValueError('Refusing manifest symlink')
        target.write_text(json.dumps({'source_manifests': [str(manifest_path), str(additional_manifest.resolve())], 'jobs': jobs}, indent=2))
    review = base / 'review'
    thumbs = review / 'thumbnails'
    for target in (review, thumbs):
        if target.is_symlink() or not target.resolve().is_relative_to(base):
            raise ValueError('Refusing gallery symlink escape')
    thumbs.mkdir(parents=True, exist_ok=True)
    cards = []
    ordered = sorted(jobs, key=lambda j: (j['experiment'], j['roi'],
                     {'mgi': 0, 'fbrm': 1, 'combo': 2}[j['type']], j['branch'], j['view'] != 'full', j['output'], j['onsets_variant'] != 'shown'))
    previous = None
    for j in ordered:
        if j['status'] != 'OK':
            raise ValueError('Cannot publish incomplete production run')
        png = (base / j['output_path']).resolve()
        if not any(png.is_relative_to(root) for root in figure_roots) or not png.is_file():
            raise ValueError('Invalid production PNG path')
        if hashlib.sha256(png.read_bytes()).hexdigest() != j['output_sha256']:
            raise ValueError('Production PNG hash mismatch')
        picture = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if picture is None:
            raise ValueError('Undecodable PNG')
        width = min(650, picture.shape[1])
        preview = cv2.resize(picture, (width, round(picture.shape[0]*width/picture.shape[1])), interpolation=cv2.INTER_AREA)
        thumb = thumbs / (j['onsets_variant'] + '-' + png.name)
        if thumb.is_symlink():
            raise ValueError('Refusing thumbnail symlink')
        if not cv2.imwrite(str(thumb), preview):
            raise OSError('Thumbnail write failed')
        group = (j['experiment'], j['roi'], j['type'], j['branch'])
        if group != previous:
            if previous is not None:
                cards.append('</div></section>')
            title = f"EX-{j['experiment']} · {j['roi']} · {j['type'].upper()} {j['branch'].upper()}"
            cards.append('<section><h2>'+escape(title)+'</h2><div class="grid">')
            previous = group
        metadata = f"EX-{j['experiment']} · {j['roi']} · {j['type'].upper()} {j['branch'].upper()} · {j['view']} · {j['mode'].upper()} · Onsets: {j['onsets_variant']}"
        limits = j['limits']
        details = []
        if j['view'] == 'limited':
            details.append('Tr: ' + ' – '.join(map(str, limits['x_range'])) + ' °C')
        for family in ('mgi', 'fbrm'):
            lo, hi = limits.get(family+'_y_lower'), limits.get(family+'_y_upper')
            if lo is not None or hi is not None:
                details.append(f'{family.upper()} y: {lo if lo is not None else "auto"} – {hi if hi is not None else "auto"}')
        if j['type'] == 'fbrm':
            details.append('Saved analysis prefix; excludes post-cutoff cooling context.')
        attrs = ' '.join(f'data-{k}="{escape(str(j[k]), quote=True)}"' for k in ('experiment','roi','type','branch','view','onsets_variant'))
        href = '../' + quote(j['output_path'], safe='/')
        src = 'thumbnails/' + quote(thumb.name)
        cards.append(f'<article {attrs}><h3>{escape(metadata)}</h3><a href="{href}" target="_blank" rel="noopener"><img loading="lazy" src="{src}" alt="{escape(metadata)}"></a><p>{escape("; ".join(details))}</p><p class="filename">{escape(png.name)}</p>')
        if j['view_evidence']:
            cards.append('<details><summary>View evidence</summary><ul>'+''.join('<li>'+escape(p)+'</li>' for p in j['view_evidence'])+'</ul></details>')
        cards.append('</article>')
    if previous is not None:
        cards.append('</div></section>')
    filters=[]
    for key in ('experiment','roi','type','view','branch','onsets_variant'):
        values=sorted({str(j[key]) for j in jobs if j[key] != ''}, key=lambda v: int(v) if key=='experiment' else v)
        filters.append(f'<label>{"Onsets" if key == "onsets_variant" else key} <select data-filter="{key}"><option value="">All</option>'+''.join(f'<option value="{escape(v)}">{escape(v)}</option>' for v in values)+'</select></label>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Supplementary review — Paula</title>
<style>body{font:16px system-ui,sans-serif;margin:24px;color:#222;background:#fafafa}nav{position:sticky;top:0;background:white;padding:12px;display:flex;gap:12px;flex-wrap:wrap;border:1px solid #ddd;z-index:2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}article{background:white;border:1px solid #ddd;padding:14px;min-width:0}img{display:block;width:100%;max-width:650px;height:auto;margin:auto}h3{font-size:16px}.filename,details{font-size:12px;overflow-wrap:anywhere}[hidden]{display:none!important}</style>
<h1>Supplementary figures — review</h1><p>Click a preview to open the original production PNG. GRAY: 600 dpi; BW: 1200 dpi. Previews are scaled for browsing. MGI RAW and SG are separate saved branches. All figures retain the approved saved results.</p>
<nav>'''+''.join(filters)+'''<span id="count"></span></nav>'''+''.join(cards)+'''
<script>const filters=[...document.querySelectorAll('[data-filter]')];function update(){let count=0;document.querySelectorAll('article').forEach(card=>{card.hidden=!filters.every(f=>!f.value||card.dataset[f.dataset.filter]===f.value);if(!card.hidden)count++});document.querySelectorAll('section').forEach(s=>s.hidden=![...s.querySelectorAll('article')].some(c=>!c.hidden));document.getElementById('count').textContent=count+' figures';}filters.forEach(f=>f.addEventListener('change',update));update();</script></html>'''
    index = review / 'index.html'
    if index.is_symlink():
        raise ValueError('Refusing index symlink')
    index.write_text(page)
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.links=[]; self.images=[]
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs)
            if tag=='a': self.links.append(attrs['href'])
            if tag=='img': self.images.append(attrs['src'])
    parsed=Links();parsed.feed(page)
    expected={str((base/j['output_path']).resolve()) for j in jobs}
    assert {str((review/unquote(p)).resolve()) for p in parsed.links}==expected
    assert len(parsed.links)==len(parsed.images)==len(jobs)==len(expected)
    for path in parsed.links+parsed.images:
        assert (review/unquote(path)).is_file()
    report={'items':len(jobs),'links_verified':len(parsed.links),'thumbnails_verified':len(parsed.images),'index':'review/index.html'}
    (review/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--additional-manifest', type=Path, help='Compare a second production variant without copying its PNGs')
    args = parser.parse_args()
    build(args.manifest, args.additional_manifest)
