"""Check the generated static site's internal links, anchors and search records."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links, self.ids = [], set()
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in ("a", "link", "script", "img"):
            target = attrs.get("href") or attrs.get("src")
            if target:
                self.links.append(target)
        if tag == "option" and "value" in attrs:
            self.links.append(attrs["value"])
        if tag == "form" and "action" in attrs:
            self.links.append(attrs["action"])


def check(root, base):
    pages = {path.resolve(): Page(path) for path in root.rglob("*.html")}
    errors = []
    total = 0

    def verify(url, source):
        nonlocal total
        parsed = urlsplit(url)
        if parsed.netloc or parsed.scheme:
            return
        total += 1
        if parsed.path.startswith(base):
            dest = root / unquote(parsed.path[len(base):])
        elif parsed.path.startswith("/"):
            errors.append(f"{source}: outside base path: {url}")
            return
        else:
            dest = source.parent / unquote(parsed.path) if parsed.path else source
        if dest.is_dir():
            dest /= "index.html"
        dest = dest.resolve()
        if not dest.is_relative_to(root.resolve()) or not dest.is_file():
            errors.append(f"{source.relative_to(root)}: missing target {url}")
        elif parsed.fragment and dest in pages and unquote(parsed.fragment) not in pages[dest].ids:
            errors.append(f"{source.relative_to(root)}: missing anchor {url}")

    for path, page in pages.items():
        for url in page.links:
            verify(url, path)
    for index in root.glob("assets/search-*.json"):
        entries = json.loads(index.read_text(encoding="utf-8"))
        if not entries:
            errors.append(f"Empty search index: {index.name}")
        for entry in entries:
            verify(entry["url"], root / "index.html")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(pages)} HTML pages and {total} internal links/anchors; all passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path(__file__).parent / "_site")
    parser.add_argument("--base-path", default="/RUDA/")
    args = parser.parse_args()
    check(args.site.resolve(), args.base_path)
