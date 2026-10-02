"""Publication previews retain manifest physical widths; legacy cards do not."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_gallery', ROOT/'scripts/make-onset-gallery.py')
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)


class PublicationGalleryTests(unittest.TestCase):
    def test_full_and_single_panel_previews_use_manifest_widths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for width in (83.5,167):
                image = root/f'combo-{width}.png'
                image.with_suffix('.manifest.json').write_text(json.dumps({
                    'scientific_preparation': 'none; saved samples and accepted results only',
                    'options': {'width_mm': width}}))
                item = g.GalleryItem('combo','base',None,root,Path('combo'),'synthetic','roi',[Path(image.name)],[])
                page = g.render_item(item,root_abs=root,output_dir_abs=root,embed_images=False)
                self.assertIn('class="figure-card publication"',page)
                self.assertIn(f'style="width:{width:g}mm"',page)
            self.assertIn('height: auto',g.render_css(240))

    def test_legacy_or_invalid_manifest_does_not_claim_publication_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'main-clean-mono.png'
            self.assertIsNone(g.publication_width_mm(image))
            for content in ('{}','not json',json.dumps({'scientific_preparation':
                'none; saved samples and accepted results only','options':{'width_mm':'NaN'}})):
                image.with_suffix('.manifest.json').write_text(content)
                self.assertIsNone(g.publication_width_mm(image))

    def test_public_root_label_changes_only_visible_heading(self):
        from argparse import Namespace
        root = Path('/neutral/synthetic')
        args = Namespace(gallery_content='all', gallery_view='base', with_data=True,
                         thumbnail_height=240, view_id=None, open_all=True, no_tree=True, title='Synthetic gallery', root_label='examples/synthetic-rgb-tr')
        page = g.render_html(args=args, root_abs=root, output_path=root/'index.html', items=[])
        self.assertIn('<code>examples/synthetic-rgb-tr</code>', page)
        self.assertNotIn('/neutral/synthetic', page)
        args.root_label = '<synthetic>'
        self.assertIn('&lt;synthetic&gt;', g.render_html(args=args, root_abs=root, output_path=root/'index.html', items=[]))
