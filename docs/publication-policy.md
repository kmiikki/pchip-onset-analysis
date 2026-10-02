# Public repository and release policy

This repository publishes methods, source code and genuinely synthetic examples.
Do not commit private measurements, experimental onset values, derived figures,
review galleries, source-path manifests or validation reports from research data.
Renaming or anonymizing a measured series does not make it synthetic.

`generated/`, `data/`, `ram`, `final/`, logs and caches are ignored. The compact
synthetic example already tracked under `examples/` is an intentional exception;
its documented provenance must be preserved. New tests generate independent
analytic arrays and write artifacts only to ignored or temporary directories.

Before a release, inspect staged text and binary files, tests, commands and
figures. Search for private paths, experiment identifiers, onset values and
research-result tables. MGI/FBRM/COMBO terminology and algorithm parameters are
expected method content, not evidence of a leak. Review each match in context.

Private-data correctness studies are not reproduced by the public synthetic
suite. A methods reviewer should inspect parameter provenance and scientific
assumptions separately. Cite an immutable reviewed commit (and a release/DOI
when available); a mutable branch is not a versioned scientific reference.
