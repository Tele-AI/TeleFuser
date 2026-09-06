# Publishing Cookbook Guides

Cookbook model instructions live in `examples/<model>/README.md`. Optional translations live in `README_zh.md`;
long deployment or performance topics can use other Markdown files in the same directory. The website generates
its pages from these sources. Keep shared service, configuration, and architecture explanations in `docs/` and
link to them from the examples.

## Register a Guide

Add the source to `docs/cookbook.yml`, using an existing category or adding bilingual category titles:

```yaml
- slug: model-task
  category: world-models
  title:
    en: Model Task
    zh: Model Task
  sources:
    en: examples/model/README.md
    zh: examples/model/README_zh.md
```

Omit `zh` under `sources` until the translation exists. English is required; a missing Chinese translation produces
a clearly labeled English fallback at the Chinese URL. An existing `README_zh.md` must be registered explicitly.
The slug is a permanent URL identifier, independent of the source directory, navigation title, and Markdown heading.
The example above publishes at `/TeleFuser/cookbook/model-task/` and `/TeleFuser/zh/cookbook/model-task/`.

The `guides` list links existing bilingual `docs/` pages into Cookbook categories without duplicating their bodies
or changing their addresses. Experimental examples are not published unless explicitly registered.

## Write the Source

Use the example README template and lead readers through the task, verified environment, weights and inputs,
shortest run, expected output, optional configuration, performance, and troubleshooting. Commands run from the
repository root. Service examples include a client call; interactive examples include the browser URL, connection
steps, expected result, and shutdown sequence.

Document only verified runtime behavior. A successful documentation build does not establish that a model example
runs. Report hardware, software versions, date, revision, workload, and measurement scope with performance data.
If historic results lack provenance, say which fields are unavailable. Separate first-frame latency, control response,
target chunk compute FPS, and client delivery FPS for streaming workloads.

## Links and Media

Use ordinary GitHub-readable Markdown, including tables, reference links, inline code, and fenced code blocks.
Source paths are resolved relative to each Markdown file:

| Source link | Website destination |
| --- | --- |
| Registered example Markdown | Cookbook page, preferring the current language |
| A file in `docs/en/` or `docs/zh/` | Existing documentation, preferring the current language |
| Python, unpublished Markdown, or other repository files | GitHub source on `main` |
| Local images and small media | Copied asset under its full repository path |
| External URL or same-page anchor | Preserved |

Queries and anchors are retained. Use a shared explicit HTML anchor in both translations when a link must target
the same section across languages. Build validation checks the actual rendered IDs, including HTML anchors.
Absolute repository paths and missing local files are errors. Code content is checked for changes after conversion.
The generated Markdown may normalize whitespace and expand reference links; the source remains unchanged.

HTML `img`, `video`, `audio`, and `source` tags support `src` and `poster`; HTML links also work. Use an explicit `src`
instead of `srcset`. Local assets are limited to PNG, JPEG, GIF, WebP, SVG, AVIF, MP4, WebM, MP3, and OGG, with a 5 MiB
per-file and 25 MiB total limit. Referenced assets must be committed. Symlinks, weights, and datasets are not bundled;
use an external media URL for larger demonstrations. Code examples containing HTML remain literal code.

Edit and view-source actions point to the original Markdown, including the English source for fallback pages.
Revision and creation dates come from Git history of that source, using the existing localized date formatter.
Untracked files have no date. Fetch the full history in CI; generated files never supply timestamps.

## Build and Preview

From the repository root, install documentation dependencies in your Python environment:

```bash
python -m pip install -r docs/requirements.txt
python -m unittest discover -s tests/docs -v
python scripts/docs/prepare_cookbook.py build
python scripts/docs/prepare_cookbook.py serve --dev-addr 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/TeleFuser/cookbook/`. After editing example sources, stop the preview and rerun `serve`.
The first version regenerates at startup; it does not watch `examples/` for changes.

`prepare` only generates the temporary tree and inherited configuration. `check` validates an already built site.
Both local preview and CI use `.build/docs/` plus `.build/mkdocs.yml`. Never edit or commit `.build/` or `site/`.
Each generation replaces the temporary document tree, removing old pages and unused assets. The base MkDocs
configuration retains its theme, plugins, scripts, styles, existing navigation, and deployment destination.
The revision plugin's temporary-file scan is disabled in this configuration; the source hook provides its date fields.

CI runs on changes to documentation, examples, documentation tooling, and its tests. It validates the manifest, runs
the focused tests, builds both languages, checks Cookbook links and rendered anchors, and rejects warnings beyond
the recorded `scripts/docs/warnings-baseline.txt`. Existing warnings may disappear without updating the baseline;
new Cookbook problems must be fixed. The workflow provides a PR check; repository branch protection must require
that check for it to block merging.
