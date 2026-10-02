# Source integration review

Public baseline: commit `228825d`. Preserve its `scripts/`, `docs/`, `examples/`
layout and synthetic assets. This update is a file-level integration, not a
mirror of a private analysis workspace.

| Local component | Classification and decision |
| --- | --- |
| `bw-pchip-onsets-t1t2.py` | Existing source update: qualified valley reporting, provenance and display grid removal. |
| `fbrm-onsets.py` | Existing source update: bounded nearby stronger-event qualification and display ticks/grid. |
| `run-onset-workflow.py` | Existing source update: separate code/data roots and correct SG column. Adapt active path to public `scripts/`. |
| `onset_view_limits.py`, `make-onset-gallery.py` | Existing source updates: shared temperature ticks and physical publication preview widths. |
| `render-publication-figures.py`, `saved_fbrm_replay.py` | New source: saved-result rendering and verified replay. Private output root replaced by repository `data/` with the same view-path safety rule. |
| `render-supplementary-series.py`, `render-supplementary-gallery.py` | New source: discovery/plans, explicit presentation variants and offline review. |
| `xy-pchip.py`, `tr-bw-sg.py`, `onset_config.py`, `make-mgi-fbrm-combo.py`, `tr2csv.py`, `tr2rgb.py` | No algorithm difference; retain public versions. |
| `run-bw-pchip-workflow.py` | Retain public version: local removal of `--python` has no scientific benefit and would break compatibility. |
| `clean-onset-outputs.py` and legacy galleries | Retain public behavior; replace machine-specific examples with placeholders. |
| `fbrm-onsets-pchip.py` | Experimental PoC, excluded from production and this update. |
| `generate-onset-validation.py`, `patch-mgi-onset-config.py` | Private study/configuration tooling, excluded; shared replay is transferred separately. |
| `make-publication-font-test.py`, its gallery | Experimental duplicated preparation, excluded in favor of saved-result renderer. |
| Workspace README, agent instructions, manuscript mappings, local script inventory | Local context, excluded. Update public docs incrementally instead. |
| Private regression fixture, saved-experiment CLI tests and validation tests | Excluded; transfer only independent synthetic rules and fixtures. |
| Publication/workflow-root tests | Synthetic portions ported to public layout, removing real-data paths and assertions. |
| Generated output, runtime logs, approved manuscript output, RAM tree, caches and workspace helpers | Not public source; never copied. |
| Public-only import/alignment utilities, existing example generator and synthetic figures | Preserve public baseline. |

## Review boundaries

The new default MGI/FBRM qualification rules are scientific behavior changes
relative to the public baseline, transferred from the validated implementation
without retuning. Existing synthetic example outputs remain historical examples;
they are not regenerated or presented as results of the new rules.

Private study scripts and expected-result tables require separate privacy review
before any future publication. No experimental correctness claims or private
case outcomes are included here. The optional general-purpose public rewrite of
that study tooling remains out of scope, not silently replaced with synthetic
claims. The repository's existing synthetic-image provenance is retained as
documented upstream; generated private figures were not used as replacements.
