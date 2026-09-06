"""Keep source attribution and Git dates attached to repository files."""

import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import yaml
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs_git_revision_date_localized_plugin.dates import get_date_formats

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=256)
def source_dates(source: str) -> list[int]:
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%ct", "--", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(value) for value in result.stdout.splitlines()] if result.returncode == 0 else []


def on_pre_build(config: MkDocsConfig) -> None:
    source_dates.cache_clear()


@event_priority(-200)
def on_page_content(html: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    mapping = yaml.safe_load((ROOT / ".build/cookbook-sources.yml").read_text(encoding="utf-8"))
    locale = getattr(page.file, "locale", "en")
    source = mapping.get(page.file.src_uri) or mapping.get(f"{locale}/{page.file.src_uri}")
    if not source and "cookbook" in Path(page.file.src_uri).parts:
        source = "docs/cookbook.yml"
    if source:
        page.edit_url = f"{config.repo_url}/edit/main/{quote(source)}"
        # Material derives the view action by replacing /edit/ in edit_url.
    else:
        source = f"docs/{page.file.src_uri}"
        if not (ROOT / source).is_file():
            source = f"docs/{locale}/{page.file.src_uri}"
    if (ROOT / source).is_file():
        dates = source_dates(source)
        if dates:
            page.meta["git_revision_date_localized"] = get_date_formats(dates[0], locale=locale)["datetime"]
            page.meta["git_creation_date_localized"] = get_date_formats(dates[-1], locale=locale)["datetime"]
    return html
