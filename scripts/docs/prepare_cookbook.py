"""Build a disposable MkDocs tree from explicitly published example sources."""

from __future__ import annotations

import argparse
import copy
import html
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import yaml
from markdown_it.token import Token
from mdformat._util import build_mdit
from mdformat.renderer import MDRenderer
from mkdocs.config import load_config
from mkdocs.utils.yaml import yaml_load

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("en", "zh")
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif", ".mp4", ".webm", ".mp3", ".ogg"}
MAX_ASSET_BYTES = 5 * 1024 * 1024
MAX_TOTAL_ASSET_BYTES = 25 * 1024 * 1024


def source_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Expected a repository-relative source path: {value!r}")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source escapes repository: {value}")
    relative = path.relative_to(root)
    if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError(f"Symlink sources are not published: {value}")
    if not path.exists():
        raise ValueError(f"Missing source: {value}")
    return path.resolve()


def localized(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(LANGUAGES):
        raise ValueError(f"{label} requires en and zh titles")
    if any(not isinstance(title, str) or not title.strip() for title in value.values()):
        raise ValueError(f"Empty or invalid title: {label}")


def read_manifest(root: Path) -> dict:
    manifest = yaml.safe_load((root / "docs/cookbook.yml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Cookbook manifest requires version: 1")
    categories = manifest.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Cookbook requires categories")
    for category, titles in categories.items():
        if not isinstance(category, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", category):
            raise ValueError(f"Invalid category: {category}")
        localized(titles, category)
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Cookbook requires an explicit pages list")
    slugs = {"index"}
    sources = set()
    for page in pages:
        slug = page.get("slug")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            raise ValueError(f"Invalid slug: {slug}")
        if slug in slugs:
            raise ValueError(f"Duplicate or reserved slug: {slug}")
        slugs.add(slug)
        if page.get("category") not in categories:
            raise ValueError(f"Unknown category for {slug}")
        localized(page.get("title"), slug)
        translations = page.get("sources")
        if not isinstance(translations, dict) or "en" not in translations or set(translations) - set(LANGUAGES):
            raise ValueError(f"Invalid source languages for {slug}")
        english = source_path(root, translations["en"])
        chinese = english.with_name("README_zh.md")
        if english.name == "README.md" and chinese.is_file() and "zh" not in translations:
            raise ValueError(f"Register existing translation: {chinese.relative_to(root)}")
        for source in translations.values():
            path = source_path(root, source)
            if path.suffix != ".md" or not path.is_file() or not path.is_relative_to(root / "examples"):
                raise ValueError(f"Page sources must be Markdown files under examples/: {source}")
            if path in sources:
                raise ValueError(f"Source registered twice: {source}")
            sources.add(path)
    for guide in manifest.get("guides", []):
        localized(guide.get("title"), "guide")
        if guide.get("category") not in categories:
            raise ValueError("Unknown guide category")
        for language in LANGUAGES:
            path = source_path(root, f"docs/{language}/{guide['path']}")
            if path.suffix != ".md" or not path.is_relative_to(root / "docs" / language):
                raise ValueError(f"Invalid guide path: {path}")
    return manifest


class MediaHTML(HTMLParser):
    """Replace parsed URL attributes, preserving all other HTML bytes."""

    def __init__(self, text: str, convert: Callable[[str, bool], str]):
        super().__init__(convert_charrefs=False)
        self.text = text
        self.convert = convert
        self.edits: list[tuple[int, int, str]] = []
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.protected = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"pre", "code", "script", "style"}:
            self.protected += 1
        if self.protected:
            return
        rewritten = []
        changed = False
        for name, value in attrs:
            new = value
            if value is not None and (
                (tag == "a" and name == "href")
                or (tag in {"img", "video", "audio", "source", "track"} and name in {"src", "poster"})
            ):
                new = self.convert(value, tag != "a")
            if name == "srcset" and value:
                raise ValueError("Cookbook media uses src/poster; replace srcset with an explicit src")
            changed |= new != value
            rewritten.append(name if new is None else f'{name}="{html.escape(new, quote=True)}"')
        if changed:
            original = self.get_starttag_text()
            line, col = self.getpos()
            start = self.offsets[line - 1] + col
            ending = " />" if original.endswith("/>") else ">"
            self.edits.append((start, start + len(original), f"<{tag} {' '.join(rewritten)}{ending}"))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"pre", "code", "script", "style"} and self.protected:
            self.protected -= 1

    def rewrite(self) -> str:
        self.feed(self.text)
        self.close()
        text = self.text
        for start, end, replacement in reversed(self.edits):
            text = text[:start] + replacement + text[end:]
        return text


def walk(tokens: list[Token]) -> Iterator[Token]:
    for token in tokens:
        yield token
        yield from walk(token.children or [])


def rewrite_markdown(
    text: str,
    convert: Callable[[str, bool], str],
    convert_html: Callable[[str, bool], str] | None = None,
) -> str:
    parser = build_mdit(MDRenderer, extensions={"tables"}, mdformat_opts={"wrap": "keep"})
    env: dict = {}
    tokens = parser.parse(text, env)
    code_types = {"fence", "code_block", "code_inline"}
    code = [(token.type == "code_inline", token.content) for token in walk(tokens) if token.type in code_types]
    protected_html = 0
    for token in walk(tokens):
        attribute = {"link_open": "href", "image": "src"}.get(token.type)
        if attribute and not protected_html:
            token.attrSet(attribute, convert(token.attrGet(attribute) or "", token.type == "image"))
            # A reference can be used as both a link and an image; resolve each use.
            token.meta.pop("label", None)
        elif token.type in {"html_inline", "html_block"}:
            fragment = MediaHTML(token.content, convert_html or convert)
            fragment.protected = protected_html
            token.content = fragment.rewrite()
            protected_html = fragment.protected
    env.pop("references", None)
    result = parser.renderer.render(tokens, parser.options, env)
    rendered_code = [(t.type == "code_inline", t.content) for t in walk(parser.parse(result)) if t.type in code_types]
    if code != rendered_code:
        raise ValueError("Markdown conversion changed code content")
    return result


class Publisher:
    def __init__(
        self,
        root: Path,
        output: Path,
        manifest: dict,
        repo_url: str,
        site_url: str = "https://tele-ai.github.io/TeleFuser/",
        use_directory_urls: bool = True,
    ):
        self.root = root
        self.output = output
        self.repo_url = repo_url.rstrip("/")
        self.site_prefix = urlsplit(site_url).path.rstrip("/") + "/"
        self.use_directory_urls = use_directory_urls
        self.mapping = {
            source_path(root, source): page for page in manifest["pages"] for source in page["sources"].values()
        }
        self.assets: set[Path] = set()
        self.asset_bytes = 0

    def convert(
        self,
        url: str,
        source: Path,
        destination: Path,
        language: str,
        media: bool = False,
        raw_html: bool = False,
    ) -> str:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            return url
        if parts.path.startswith("/"):
            raise ValueError(f"Use repository-relative links in {source}: {url}")
        relative = os.path.relpath(source.parent / unquote(parts.path), self.root)
        target = source_path(self.root, relative)
        page = self.mapping.get(target)
        if page and not media:
            published = self.output / language / "cookbook" / page["slug"] / "index.md"
        elif target.is_relative_to(self.root / "docs") and target.suffix == ".md" and not media:
            document = target.relative_to(self.root / "docs")
            if document.parts[0] in LANGUAGES:
                translated = Path(language, *document.parts[1:])
                if (self.root / "docs" / translated).is_file():
                    document = translated
            published = self.output / document
        elif target.suffix.lower() in ASSET_SUFFIXES:
            size = target.stat().st_size
            if size > MAX_ASSET_BYTES:
                raise ValueError(f"Asset exceeds 5 MiB; use an external media URL: {relative}")
            if target not in self.assets:
                self.asset_bytes += size
                if self.asset_bytes > MAX_TOTAL_ASSET_BYTES:
                    raise ValueError("Cookbook assets exceed the 25 MiB total budget")
                self.assets.add(target)
            published = self.output / "assets/cookbook" / target.relative_to(self.root)
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, published)
        else:
            if media:
                raise ValueError(f"Unsupported local media: {relative}")
            kind = "tree" if target.is_dir() else "blob"
            github = f"{self.repo_url}/{kind}/main/{quote(target.relative_to(self.root).as_posix())}"
            return urlunsplit((*urlsplit(github)[:3], parts.query, parts.fragment))
        if raw_html:
            path = published.relative_to(self.output).as_posix()
            if path.startswith("en/"):
                path = path[3:]
            if published.suffix == ".md":
                if self.use_directory_urls:
                    path = path.removesuffix("index.md") if published.name == "index.md" else path[:-3] + "/"
                else:
                    path = path[:-3] + ".html"
            path = self.site_prefix + quote(path)
        else:
            path = quote(os.path.relpath(published, destination.parent).replace(os.sep, "/"))
        return urlunsplit(("", "", path, parts.query, parts.fragment))


def prepare(root: Path = ROOT) -> Path:
    root = root.resolve()
    manifest = read_manifest(root)
    base_path = root / "mkdocs.yml"
    with base_path.open(encoding="utf-8") as stream:
        base = yaml_load(stream)
    config = load_config(str(base_path))
    if Path(config.docs_dir) != root / "docs":
        raise ValueError("Cookbook expects the repository docs/ source tree")
    i18n = next((plugin["i18n"] for plugin in base["plugins"] if isinstance(plugin, dict) and "i18n" in plugin), {})
    languages = i18n.get("languages", [])
    active = {language["locale"] for language in languages if language.get("build", True)}
    default = next((language["locale"] for language in languages if language.get("default")), None)
    if i18n.get("docs_structure") != "folder" or active != set(LANGUAGES) or default != "en":
        raise ValueError("Cookbook requires folder-based en (default) and zh MkDocs languages")
    build = root / ".build"
    build.mkdir(exist_ok=True)
    if build.is_symlink():
        raise ValueError(".build must not be a symlink")
    output = build / "docs"
    if output.is_symlink():
        raise ValueError(".build/docs must not be a symlink")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(root / "docs", output, ignore=shutil.ignore_patterns("cookbook.yml", "requirements.txt"))
    publisher = Publisher(root, output, manifest, config.repo_url, config.site_url, config.use_directory_urls)
    source_map = {}
    for language in LANGUAGES:
        index = ["# Cookbook", ""]
        for category, titles in manifest["categories"].items():
            index += [f"## {titles[language]}", ""]
            category_index = [f"# {titles[language]}", ""]
            for page in manifest["pages"]:
                if page["category"] == category:
                    index.append(f"- [{page['title'][language]}]({page['slug']}/index.md)")
                    category_index.append(f"- [{page['title'][language]}](../../{page['slug']}/index.md)")
            for guide in manifest.get("guides", []):
                if guide["category"] == category:
                    index.append(f"- [{guide['title'][language]}](../{guide['path']})")
                    category_index.append(f"- [{guide['title'][language]}](../../../{guide['path']})")
            category_page = output / language / "cookbook/categories" / category / "index.md"
            category_page.parent.mkdir(parents=True, exist_ok=True)
            category_page.write_text("\n".join(category_index) + "\n", encoding="utf-8")
            index.append("")
        home = output / language / "cookbook/index.md"
        home.parent.mkdir(parents=True, exist_ok=True)
        home.write_text("\n".join(index), encoding="utf-8")
        for page in manifest["pages"]:
            source = source_path(root, page["sources"].get(language, page["sources"]["en"]))
            destination = output / language / "cookbook" / page["slug"] / "index.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            body = rewrite_markdown(
                source.read_text(encoding="utf-8"),
                lambda url, media: publisher.convert(url, source, destination, language, media),
                lambda url, media: publisher.convert(url, source, destination, language, media, raw_html=True),
            )
            notice = ""
            if language not in page["sources"]:
                notice = "> **English fallback:** This guide has not been translated into Chinese.\n\n"
            metadata = yaml.safe_dump({"title": page["title"][language]}, allow_unicode=True)
            destination.write_text("---\n" + metadata + "---\n\n" + notice + body, encoding="utf-8")
            source_map[destination.relative_to(output).as_posix()] = source.relative_to(root).as_posix()
    (build / "cookbook-sources.yml").write_text(yaml.safe_dump(source_map), encoding="utf-8")
    navigation = [{"Overview": "cookbook/index.md"}]
    translations = {"Cookbook": "Cookbook"}
    for category, titles in manifest["categories"].items():
        entries = [
            {page["title"]["en"]: f"cookbook/{page['slug']}/index.md"}
            for page in manifest["pages"]
            if page["category"] == category
        ]
        entries += [
            {guide["title"]["en"]: guide["path"]}
            for guide in manifest.get("guides", [])
            if guide["category"] == category
        ]
        # Material's navigation.indexes consumes the first index page as the
        # section link. Give each category its own index so every model is listed.
        navigation.append({titles["en"]: [{"Overview": f"cookbook/categories/{category}/index.md"}, *entries]})
        translations[titles["en"]] = titles["zh"]
    for entry in manifest["pages"] + manifest.get("guides", []):
        translations[entry["title"]["en"]] = entry["title"]["zh"]
    plugins = copy.deepcopy(base["plugins"])
    for plugin in plugins:
        if isinstance(plugin, dict) and "i18n" in plugin:
            for language in plugin["i18n"]["languages"]:
                if language["locale"] == "zh":
                    language.setdefault("nav_translations", {}).update(translations)
        if isinstance(plugin, dict) and "git-revision-date-localized" in plugin:
            # Copied files have no Git history. The source hook supplies dates.
            plugin["git-revision-date-localized"]["enabled"] = False
    generated = {
        "INHERIT": str(base_path),
        "docs_dir": str(output),
        "site_dir": str(config.site_dir),
        "nav": [*base["nav"][:1], {"Cookbook": navigation}, *base["nav"][1:]],
        "plugins": plugins,
        "hooks": [
            *[str((root / path).resolve()) for path in base.get("hooks", [])],
            str(root / "scripts/docs/cookbook_hooks.py"),
        ],
        "watch": [str((root / path).resolve()) for path in base.get("watch", [])],
    }
    if base.get("theme", {}).get("custom_dir"):
        generated["theme"] = {"custom_dir": str(Path(config.theme.custom_dir).resolve())}
    generated_path = build / "mkdocs.yml"
    generated_path.write_text(yaml.safe_dump(generated, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return generated_path


class SiteHTML(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name in ("id", "name"):
            if values.get(name):
                self.ids.add(values[name])
        for name in ("href", "src", "poster"):
            if values.get(name):
                self.links.append(values[name])


def check_site(site: Path, site_url: str) -> None:
    base = site_url.rstrip("/") + "/"
    prefix = urlsplit(base).path
    cache: dict[Path, SiteHTML] = {}
    errors = []
    for page in sorted(site.rglob("*.html")):
        parsed = SiteHTML(page.read_text(encoding="utf-8"))
        cache[page] = parsed
        relative = page.relative_to(site).as_posix()
        current = urljoin(base, relative.removesuffix("index.html"))
        for link in parsed.links:
            target = urlsplit(urljoin(current, link))
            if target.netloc != urlsplit(base).netloc or target.scheme not in {"http", "https"}:
                continue
            if not target.path.startswith(prefix):
                errors.append(f"{relative}: link escapes deployment prefix: {link}")
                continue
            local = site / unquote(target.path[len(prefix) :])
            if local.is_dir():
                local /= "index.html"
            if not local.is_file():
                errors.append(f"{relative}: missing target: {link}")
            elif target.fragment and local.suffix == ".html":
                if local not in cache:
                    cache[local] = SiteHTML(local.read_text(encoding="utf-8"))
                if unquote(target.fragment) not in cache[local].ids:
                    errors.append(f"{relative}: missing anchor: {link}")
    if errors:
        raise ValueError("Site link validation failed:\n" + "\n".join(errors))


def check_warnings(warnings: list[str], baseline: Path) -> None:
    allowed = Counter(
        line for line in baseline.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")
    )
    added = Counter(warnings) - allowed
    if added:
        raise ValueError("New MkDocs warnings:\n" + "\n".join(added.elements()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "serve", "check"), nargs="?", default="prepare")
    parser.add_argument("--dev-addr", default="127.0.0.1:8000")
    args = parser.parse_args()
    if args.command == "check":
        config = load_config(str(ROOT / ".build/mkdocs.yml"))
        check_site(Path(config.site_dir), config.site_url)
        return
    generated = prepare()
    print(f"Prepared {generated}", flush=True)
    if args.command == "prepare":
        return
    command = [sys.executable, "-m", "mkdocs", args.command, "--config-file", str(generated)]
    if args.command == "serve":
        command += ["--dev-addr", args.dev_addr]
    if args.command == "serve":
        subprocess.run(command, cwd=ROOT, check=True)
        return
    warnings = []
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            match = re.match(r"WARNING\s+-\s+(.*)", line)
            if match:
                warnings.append(match[1])
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    if args.command == "build":
        config = load_config(str(generated))
        check_site(Path(config.site_dir), config.site_url)
        check_warnings(warnings, ROOT / "scripts/docs/warnings-baseline.txt")
        print("Rendered site pages, assets, and anchors validated.")


if __name__ == "__main__":
    main()
