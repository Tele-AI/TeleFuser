"""CPU-only documentation checks; runnable without importing TeleFuser or torch."""

import base64
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.docs.prepare_cookbook import (
    ROOT,
    Publisher,
    check_site,
    check_warnings,
    prepare,
    read_manifest,
    rewrite_markdown,
)


class CookbookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {
            "version": 1,
            "categories": {"models": {"en": "Models", "zh": "Models ZH"}},
            "pages": [
                {
                    "slug": "first",
                    "category": "models",
                    "title": {"en": "First", "zh": "First ZH"},
                    "sources": {"en": "examples/first/README.md"},
                },
                {
                    "slug": "second",
                    "category": "models",
                    "title": {"en": "Second", "zh": "Second ZH"},
                    "sources": {"en": "examples/second/README.md", "zh": "examples/second/README_zh.md"},
                },
            ],
            "guides": [{"category": "models", "title": {"en": "Guide", "zh": "Guide ZH"}, "path": "guide.md"}],
        }
        for lang in ("en", "zh"):
            self.write(f"docs/{lang}/index.md", "# Home\n")
            self.write(f"docs/{lang}/guide.md", '# Guide\n\n<a id="stable-anchor"></a>\n')
        self.write("examples/first/README.md", "# First\n\n[Second](../second/README.md#run)\n")
        self.write("examples/second/README.md", "# Second\n\n## Run\n")
        self.write("examples/second/README_zh.md", "# Second translation\n\n## Run\n")
        self.write("examples/first/run script.py", "print('ready')\n")
        self.write("docs/styles/extra.css", "body { color: black; }\n")
        self.write("docs/scripts/extra.js", "void 0;\n")
        self.write(
            "mkdocs.yml",
            yaml.safe_dump(
                {
                    "site_name": "Test Cookbook",
                    "site_url": "https://example.org/TeleFuser/",
                    "repo_url": "https://github.com/Tele-AI/TeleFuser",
                    "edit_uri": "edit/main/docs/",
                    "nav": [{"Home": "index.md"}, {"Guide": "guide.md"}],
                    "theme": {"name": "material", "features": ["content.action.edit", "content.action.view"]},
                    "plugins": [
                        "search",
                        {
                            "i18n": {
                                "docs_structure": "folder",
                                "languages": [
                                    {"locale": "en", "default": True, "name": "English"},
                                    {"locale": "zh", "name": "Chinese"},
                                ],
                            }
                        },
                    ],
                    "extra_css": ["styles/extra.css"],
                    "extra_javascript": ["scripts/extra.js"],
                }
            ),
        )
        self.write("scripts/docs/cookbook_hooks.py", (ROOT / "scripts/docs/cookbook_hooks.py").read_text())
        self.save_manifest()
        self.publisher = Publisher(
            self.root, self.root / ".build/docs", self.manifest, "https://github.com/Tele-AI/TeleFuser"
        )
        self.source = self.root / "examples/first/README.md"
        self.destination = self.root / ".build/docs/en/cookbook/first/index.md"

    def write(self, path: str, text: str) -> Path:
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return file

    def save_manifest(self) -> None:
        self.write("docs/cookbook.yml", yaml.safe_dump(self.manifest))

    def convert(self, url: str, media: bool = False, **kwargs) -> str:
        return self.publisher.convert(url, self.source, self.destination, "en", media, **kwargs)

    def image(self, path: str) -> None:
        file = self.write(path, "")
        file.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5p8AAAAASUVORK5CYII="
            )
        )

    def test_links_references_queries_spaces_and_code(self) -> None:
        text = (
            '[Run][other]\n\n[other]: ../second/README.md?mode=fast#run "Run title"\n\n'
            "[Script](<run script.py>)\n\n[Docs](../../docs/en/guide.md#stable-anchor)\n\n"
            '`[Run](missing.py)`\n\n```bash\n[Run](missing.py)\n<img src="absent.png">\n```\n'
        )
        result = rewrite_markdown(text, self.convert)
        self.assertIn("../second/index.md?mode=fast#run", result)
        self.assertIn("Run title", result)
        self.assertIn("blob/main/examples/first/run%20script.py", result)
        self.assertIn("../../guide.md#stable-anchor", result)
        self.assertIn("`[Run](missing.py)`", result)
        self.assertIn('```bash\n[Run](missing.py)\n<img src="absent.png">\n```', result)

    def test_collapsed_shortcut_and_image_references(self) -> None:
        self.image("examples/first/a b.png")
        result = rewrite_markdown(
            "[go][] [go] ![photo][]\n\n[go]: ../second/README.md\n[photo]: <a b.png>\n", self.convert
        )
        self.assertEqual(result.count("../second/index.md"), 2)
        self.assertIn("assets/cookbook/examples/first/a%20b.png", result)

    def test_indented_code_and_tables_keep_content(self) -> None:
        result = rewrite_markdown(
            "    [literal](missing.py)\n\n| Name | Link |\n| --- | --- |\n| Run | [Second](../second/README.md) |\n",
            self.convert,
        )
        self.assertIn("[literal](missing.py)", result)
        self.assertIn("../second/index.md", result)
        self.assertIn("| Name", result)

    def test_html_media_and_code(self) -> None:
        self.image("examples/first/poster.png")
        self.write("examples/first/clip.mp4", "small test media")
        result = rewrite_markdown(
            '<video poster="poster.png" controls><source src="clip.mp4?x=1&amp;y=2#t=1"></video>\n\n'
            '<img src="poster.png" alt="A &amp; B">\n\n'
            '<code><img src="missing.png"></code>\n\n'
            '<a href="../second/README.md#run">Run</a>\n',
            self.convert,
            lambda url, media: self.convert(url, media, raw_html=True),
        )
        self.assertIn("/TeleFuser/assets/cookbook/examples/first/poster.png", result)
        self.assertIn("clip.mp4?x=1&amp;y=2#t=1", result)
        self.assertIn('<code><img src="missing.png"></code>', result)
        self.assertIn('href="/TeleFuser/cookbook/second/#run"', result)

    def test_assets_are_isolated_and_bounded(self) -> None:
        self.image("examples/first/same.png")
        self.image("examples/second/same.png")
        self.assertNotEqual(self.convert("same.png", True), self.convert("../second/same.png", True))
        with patch("scripts.docs.prepare_cookbook.MAX_ASSET_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                self.convert("same.png", True)
        self.write("examples/first/model.safetensors", "weights")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.convert("model.safetensors", True)
        self.assertIn("blob/main/", self.convert("model.safetensors"))

    def test_missing_and_escaping_sources_fail(self) -> None:
        for url in ("missing.png", "../../../../outside.png"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.convert(url, True)
        self.write("examples/first/target.png", "test")
        (self.source.parent / "alias.png").symlink_to(self.source.parent / "target.png")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.convert("alias.png", True)

    def test_external_urls_and_local_anchors_unchanged(self) -> None:
        for url in ("https://example.com/x?q=a#b", "//example.com/a.png", "mailto:docs@example.com", "#run"):
            self.assertEqual(self.convert(url), url)

    def test_cross_page_links_keep_language(self) -> None:
        target = self.root / ".build/docs/zh/cookbook/first/index.md"
        result = self.publisher.convert("../../docs/en/guide.md", self.source, target, "zh")
        self.assertEqual(result, "../../guide.md")
        result = self.publisher.convert("../second/README.md#run", self.source, target, "zh")
        self.assertEqual(result, "../second/index.md#run")

    def test_manifest_rejects_duplicates_and_invalid_languages(self) -> None:
        original = copy.deepcopy(self.manifest)
        for field, value in (
            ("slug", "first"),
            ("slug", "../escape"),
            ("category", "unknown"),
            ("sources", {"fr": "examples/second/README.md"}),
            ("sources", {"en": "examples/first/README.md"}),
            ("title", {"en": "Only English"}),
        ):
            with self.subTest(field=field, value=value):
                self.manifest = copy.deepcopy(original)
                self.manifest["pages"][1][field] = value
                self.save_manifest()
                with self.assertRaises(ValueError):
                    read_manifest(self.root)

    def test_existing_translation_must_be_registered(self) -> None:
        self.write("examples/first/README_zh.md", "# Translated\n")
        with self.assertRaisesRegex(ValueError, "Register existing translation"):
            read_manifest(self.root)

    def test_repeat_generation_removes_stale_pages(self) -> None:
        config = prepare(self.root)
        output = self.root / ".build/docs"
        first = (output / "zh/cookbook/first/index.md").read_text()
        second = (output / "zh/cookbook/second/index.md").read_text()
        self.assertIn("English fallback", first)
        self.assertNotIn("English fallback", second)
        self.assertIn("Second translation", second)
        self.manifest["pages"].pop(0)
        self.save_manifest()
        prepare(self.root)
        self.assertFalse((output / "en/cookbook/first").exists())
        generated = yaml.safe_load(config.read_text())
        self.assertEqual(generated["INHERIT"], str(self.root / "mkdocs.yml"))
        self.assertEqual(generated["site_dir"], str(self.root / "site"))
        self.assertTrue((output / "en/cookbook/categories/models/index.md").is_file())
        self.assertEqual((output / "styles/extra.css").read_text(), "body { color: black; }\n")

    def test_rendered_site_search_edit_links_and_media(self) -> None:
        self.image("examples/first/photo.png")
        self.write(
            "examples/first/README.md",
            '# First\n\n[Second](../second/README.md#run)\n\n![Photo](photo.png)\n\n<img src="photo.png">\n',
        )
        config = prepare(self.root)
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "-f", str(config)], capture_output=True, text=True, cwd=self.root
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("WARNING", result.stderr)
        site = self.root / "site"
        check_site(site, "https://example.org/TeleFuser/")
        for language in ("", "zh/"):
            page = (site / f"{language}cookbook/first/index.html").read_text()
            self.assertIn("/edit/main/examples/first/README.md", page)
            self.assertIn("/raw/main/examples/first/README.md", page)
        self.assertIn("cookbook/first/", (site / "search/search_index.json").read_text())
        self.assertTrue((site / "guide/index.html").is_file())
        self.assertTrue((site / "zh/guide/index.html").is_file())

    def test_relative_theme_hooks_and_watch_paths(self) -> None:
        self.write("overrides/partials/test.html", "Test template\n")
        self.write("scripts/docs/custom_hook.py", "def on_config(config):\n    return config\n")
        base_path = self.root / "mkdocs.yml"
        base = yaml.safe_load(base_path.read_text())
        base["theme"]["custom_dir"] = "overrides"
        base["hooks"] = ["scripts/docs/custom_hook.py"]
        base["watch"] = ["examples"]
        base_path.write_text(yaml.safe_dump(base))
        generated = yaml.safe_load(prepare(self.root).read_text())
        self.assertEqual(generated["theme"]["custom_dir"], str(self.root / "overrides"))
        self.assertEqual(generated["hooks"][0], str(self.root / "scripts/docs/custom_hook.py"))
        self.assertEqual(generated["watch"], [str(self.root / "examples")])

    def test_rendered_broken_anchor_or_deployment_prefix_fails(self) -> None:
        self.write("site/cookbook/index.html", '<a href="#missing">bad</a>')
        with self.assertRaisesRegex(ValueError, "missing anchor"):
            check_site(self.root / "site", "https://example.org/TeleFuser/")
        self.write("site/cookbook/index.html", '<img src="/assets/a.png">')
        with self.assertRaisesRegex(ValueError, "deployment prefix"):
            check_site(self.root / "site", "https://example.org/TeleFuser/")

    def test_new_warnings_fail_without_requiring_old_warnings(self) -> None:
        baseline = self.write("warnings.txt", "# Existing\nOld warning\n")
        check_warnings([], baseline)
        check_warnings(["Old warning"], baseline)
        with self.assertRaisesRegex(ValueError, "New MkDocs warnings"):
            check_warnings(["New warning"], baseline)


if __name__ == "__main__":
    unittest.main()
