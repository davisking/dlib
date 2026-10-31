# Copyright (C) 2026 Davis E. King (davis@dlib.net)
# License: Boost Software License. See LICENSE.txt for the full license.
"""Run htmlify's include-link regression tests: python3 test_htmlify.py ./build/htmlify"""

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.urls = []
        self.feed(page.read_text())

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href")


class IncludeLinks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="htmlify-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.main = self.put(
            "examples/main.cpp",
            '#include "pkg/header.h"\n#include "local.h"\n'
            '#include "missing.h"\n#include "duplicate.h"\n',
        )
        self.put("pkg/header.h", "// library header\n")
        self.put("examples/local.h", "// local header takes precedence\n")
        self.put("inc/local.h", "// include-directory header\n")
        self.put("inc/duplicate.h", "// first include directory\n")
        self.put("other/duplicate.h", "// second include directory\n")
        self.put("other/missing.h", "// must not match by basename\n")

    def put(self, name, contents):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def run_htmlify(self, *args):
        return subprocess.run(
            [str(BINARY), *map(str, args)], check=True, capture_output=True, text=True
        )

    def render(self, name="output", *options):
        output = self.root / name
        self.run_htmlify(
            "-i", self.source, "-o", output,
            "-I", self.source, "-I", self.source / "inc", self.main, *options,
        )
        return output

    def test_search_order_and_final_locations(self):
        output = self.render()
        self.assertFalse((output / "examples/main.cpp.html").exists())
        self.assertEqual(
            Links(output / "main.cpp.html").urls,
            ["pkg/header.h.html", "examples/local.h.html", "inc/duplicate.h.html"],
        )
        for page in output.rglob("*.html"):
            for url in Links(page).urls:
                self.assertTrue((page.parent / unquote(urlsplit(url).path)).is_file())

    def test_first_existing_header_is_not_rendered(self):
        external = self.root / "external"
        external.mkdir()
        (external / "duplicate.h").write_text("// not selected for rendering\n")
        output = self.root / "output"
        self.run_htmlify(
            "-i", self.source, "-o", output,
            "-I", external, "-I", self.source / "inc", self.main,
        )
        self.assertEqual(Links(output / "main.cpp.html").urls, ["examples/local.h.html"])

    def test_url_escaping(self):
        filename = "odd #&' name.h"
        self.main.write_text('#include "' + filename + '"\n')
        self.put("examples/" + filename, "// punctuation in filename\n")
        output = self.render()
        self.assertEqual(
            Links(output / "main.cpp.html").urls,
            ["examples/odd%20%23%26%27%20name.h.html"],
        )

    def test_output_modes(self):
        numbered = self.render("numbered", "-n")
        self.assertIn("pkg/header.h.html", Links(numbered / "main.cpp.html").urls)
        printable = self.render("printable", "-b")
        self.assertEqual(Links(printable / "main.cpp.html").urls, [])
        combined = self.root / "combined.html"
        self.run_htmlify("--cat", combined, "-i", self.source, "-I", self.source)
        self.assertEqual(Links(combined).urls, [])

    def test_flattened_output(self):
        output = self.root / "flat"
        self.run_htmlify(
            "-i", self.source / "pkg", "--flatten", "-o", output,
            "-I", self.source, self.main,
        )
        self.assertEqual(Links(output / "main.cpp.html").urls, ["header.h.html"])

    def test_invalid_option_fails(self):
        result = subprocess.run([str(BINARY), "--not-a-real-option"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    BINARY = Path(sys.argv.pop(1)).resolve()
    unittest.main()
