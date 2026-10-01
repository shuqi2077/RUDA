"""Build the RUDA homepage and documentation from repository Markdown."""
from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "website"
REPO = "https://github.com/shuqi2077/RUDA"
ORIGIN = "https://shuqi2077.github.io"
COPY = json.loads((WEB / "content.json").read_text(encoding="utf-8"))
GUIDES = ["README", "getting-started", "programming-guide", "tensor-framework", "tensor-recipes",
          "backend-composition", "data-and-storage", "training",
          "model-inference", "samples", "fused-adamw", "gradient-guard", "muon"]
REFERENCE = ["api-reference", "compiler-guide", "ptx", "runtime-api", "driver-api", "compatibility", "debugging"]
PROJECT = ["project", "CONTRIBUTING"]


def esc(value):
    return html.escape(str(value), quote=True)


def slug(value, separator):
    value = re.sub(r"[^\w\-\s]", "", value.lower(), flags=re.UNICODE)
    return re.sub(r"\s", separator, value.strip())


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    parser = Text()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def document_sources():
    sources = {}
    for lang in COPY:
        folder = ROOT / "docs" / lang
        for name in GUIDES + REFERENCE + PROJECT:
            path = folder / (name + ".md")
            if path.exists():
                sources[path.relative_to(ROOT).as_posix()] = (lang, "index" if name == "README" else name)
        for path in sorted((folder / "libraries").glob("*.md")):
            sources[path.relative_to(ROOT).as_posix()] = (lang, "libraries/" + ("index" if path.stem == "README" else path.stem))
    sources["README.md"] = ("en", "project")
    sources["ruda-torch/README.md"] = ("en", "native-pytorch")
    return sources


class Site:
    def __init__(self, base):
        self.base = "/" + base.strip("/") + "/" if base.strip("/") else "/"
        self.sources = document_sources()
        self.routes = {path: self.url(f"{lang}/docs/{key}/") for path, (lang, key) in self.sources.items()}
        self.pages = {}
        self.index = {lang: [] for lang in COPY}
        self.files = {}

    def url(self, path=""):
        return self.base + path.lstrip("/")

    def doc(self, lang, key):
        return self.url(f"{lang}/docs/{key}/")

    def rewrite_link(self, value, source, image=False):
        value = html.unescape(value)
        parsed = urlsplit(value)
        if value.startswith("#") or parsed.scheme not in ("", "http", "https"):
            return value
        if parsed.netloc:
            prefix = REPO + "/blob/main/"
            if not value.startswith(prefix):
                return value
            target = unquote(urlsplit(value[len(prefix):]).path)
        else:
            if parsed.path.startswith("/"):
                return value
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parsed.path)))
        if target == "docs/README.md":
            return self.doc(self.sources[source][0], "index")
        # Crate README links share the same underlying library manual.
        match = re.fullmatch(r"(ruBLAS|ruDNN|ruFFT|ruPRIM|ruRAND|ruSPARSE|ruTENSOR|ruCCL)/(?:docs/(zh|ja|de|ru)/)?README.md", target)
        if match:
            target = f"docs/{match[2] or 'en'}/libraries/{match[1].lower()}.md"
        tail = ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        if target in self.routes:
            return self.routes[target] + tail
        if image:
            return "https://raw.githubusercontent.com/shuqi2077/RUDA/main/" + quote(target, safe="/") + tail
        return REPO + "/blob/main/" + quote(target, safe="/") + tail

    def load(self):
        for source, (lang, key) in self.sources.items():
            md = markdown.Markdown(extensions=["extra", "toc", "codehilite", "sane_lists"],
                                   extension_configs={"toc": {"slugify": slug, "permalink": False},
                                                      "codehilite": {"guess_lang": False}})
            body = md.convert((ROOT / source).read_text(encoding="utf-8"))
            body = re.sub(r'(href|src)="([^"]*)"', lambda m: f'{m[1]}="{esc(self.rewrite_link(m[2], source, m[1] == "src"))}"', body)
            heading = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
            title = plain(heading[1]) if heading else key
            self.pages[(lang, key)] = {"source": source, "title": title, "body": body, "toc": md.toc, "route": self.routes[source]}
            chunks = re.split(r'(<h[1-3]\b[^>]*>.*?</h[1-3]>)', body, flags=re.S)
            section, anchor = title, ""
            for chunk in chunks:
                if re.match(r"<h[1-3]\b", chunk):
                    section = plain(chunk)
                    found = re.search(r'id="([^"]+)"', chunk)
                    anchor = "#" + found[1] if found else ""
                elif plain(chunk):
                    self.index[lang].append({"title": title, "section": section, "url": self.routes[source] + anchor, "text": plain(chunk)})

    def language_links(self, lang, key=None, mode="home"):
        options = []
        for code, words in COPY.items():
            route = self.url(code + "/")
            if mode == "doc":
                route = self.doc(code, key if (code, key) in self.pages else "index")
            elif mode == "search":
                route = self.url(code + "/search/")
            options.append(f'<option value="{route}" {"selected" if code == lang else ""}>{words["name"]}</option>')
        return '<select class="language" aria-label="Language" data-language>' + "".join(options) + "</select>"

    def shell(self, lang, title, body, route, key=None, mode="home", sidebar=""):
        t = COPY[lang]
        config = json.dumps({"base": self.base, "lang": lang, "copy": t["copy"], "copied": t["copied"],
                             "copyError": t["copyError"], "loading": t["loading"], "empty": t["empty"],
                             "error": t["error"], "results": t["results"]}, ensure_ascii=False).replace("<", "\\u003c")
        description = re.sub(r"<[^>]+>", " ", t["lead"])
        menu = f'<button class="menu-button" aria-expanded="false" aria-controls="sidebar" data-menu>{t["menu"]} <span aria-hidden="true">☰</span></button>' if sidebar else ""
        return f'''<!doctype html>
<html lang="{t['html']}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · RUDA</title><meta name="description" content="{esc(description)}"><meta name="theme-color" content="#101312">
<link rel="canonical" href="{ORIGIN}{route}"><meta property="og:title" content="{esc(title)} · RUDA"><meta property="og:description" content="{esc(description)}"><meta property="og:type" content="website"><meta property="og:url" content="{ORIGIN}{route}">
<link rel="icon" href="{self.url('assets/favicon.svg')}" type="image/svg+xml"><link rel="stylesheet" href="{self.url('assets/site.css')}"><link rel="stylesheet" href="{self.url('assets/highlight.css')}">
<script id="site-config" type="application/json">{config}</script><script defer src="{self.url('assets/site.js')}"></script></head>
<body class="{mode}"><a class="skip" href="#main">{t['skip']}</a>
<header class="site-header"><a class="brand" href="{self.url(lang + '/')}"><span class="brand-symbol" aria-hidden="true">r</span>RUDA<span class="brand-dot">.</span></a>
<nav class="top-nav" aria-label="{t['menu']}"><a href="{self.doc(lang, 'index')}">{t['docs']}</a><a href="{REPO}">GitHub <span aria-hidden="true">↗</span></a></nav>
<form class="header-search" role="search" action="{self.url(lang + '/search/')}"><label class="sr-only" for="header-q">{t['search']}</label><input id="header-q" name="q" type="search" placeholder="{t['search']}" autocomplete="off"><button aria-label="{t['search']}" type="submit">⌕</button></form>
{self.language_links(lang, key, mode)}{menu}</header>{sidebar}{body}
<footer class="site-footer"><a class="brand" href="{self.url(lang + '/')}">RUDA<span class="brand-dot">.</span></a><p>{t['footer']}</p><nav><a href="{self.doc(lang, 'CONTRIBUTING')}">{t['contribute']}</a><a href="{REPO}/blob/main/LICENSE">License</a><a href="{REPO}">GitHub ↗</a></nav></footer></body></html>'''

    def sidebar(self, lang, current):
        t = COPY[lang]
        groups = [("guides", ["index"] + GUIDES[1:]),
                  ("libraries", sorted(k for l, k in self.pages if l == lang and k.startswith("libraries/"))),
                  ("reference", REFERENCE), ("community", PROJECT)]
        blocks = []
        for label, keys in groups:
            links = []
            for key in keys:
                page = self.pages.get((lang, key))
                if page:
                    title = page["title"]
                    if key.startswith("libraries/") and key != "libraries/index":
                        title = title.split(" — ")[0]
                    links.append(f'<a href="{page["route"]}" {"aria-current=\"page\"" if key == current else ""}>{esc(title)}</a>')
            if label == "guides":
                links.append(f'<a href="{self.doc("en", "native-pytorch")}" {"aria-current=\"page\"" if current == "native-pytorch" else ""}>PyTorch <span class="subtle">EN</span></a>')
            blocks.append(f'<div class="nav-group"><h2>{t[label]}</h2>{"".join(links)}</div>')
        return '<aside class="sidebar" id="sidebar"><nav aria-label="' + t["docs"] + '">' + "".join(blocks) + '</nav></aside>'

    def homepage(self, lang):
        t = COPY[lang]
        stack = [("04", "layerModel", "ruLLM / ruda-nn / ruda-optim"), ("03", "layerTensor", "ruda-tensor / ruda-autodiff / ruda-torch"),
                 ("02", "layerMath", "ruBLAS / ruDNN / ruFFT / ruTENSOR"), ("01", "layerCore", "ruda-kernel / ruda-compiler / runtime")]
        stack_html = "".join(f'<div class="stack-layer"><span>{n}</span><div><strong>{t[label]}</strong><code>{code}</code></div><span class="stack-arrow" aria-hidden="true">↗</span></div>' for n, label, code in stack)
        paths = [("01", "kernel", "kernelDesc", self.doc(lang, "programming-guide")), ("02", "torch", "torchDesc", self.doc("en", "native-pytorch")), ("03", "model", "modelDesc", self.doc(lang, "model-inference"))]
        cards = "".join(f'<a class="path-card" href="{route}"><span class="card-index">{n} /</span><h3>{t[title]} <span aria-hidden="true">↗</span></h3><p>{t[desc]}</p></a>' for n, title, desc, route in paths)
        libs = [("ruBLAS", "blas"), ("ruDNN", "dnn"), ("ruTENSOR", "tensor"), ("ruFFT", "fft"), ("ruSPARSE", "sparse"), ("ruCCL", "ccl")]
        library_html = "".join(f'<a href="{self.doc(lang, "libraries/" + name.lower())}"><strong>{name}</strong><span>{t[desc]}</span><b aria-hidden="true">↗</b></a>' for name, desc in libs)
        body = f'''<main id="main" class="home-main">
<section class="hero"><div class="hero-copy"><p class="eyebrow"><span class="status-dot"></span>{t['eyebrow']}</p><h1>{t['hero']}</h1><p class="hero-lead">{t['lead']}</p><div class="terminal hero-install"><pre><code>cargo add ruda --features cuda</code></pre></div><div class="actions"><a class="button primary" href="{self.doc(lang, 'getting-started')}">{t['start']} <span>↗</span></a><a class="text-link" href="{REPO}">{t['github']} →</a></div><div class="hero-tags"><span>Rust 2024</span><span>Direct PTX</span><span>Native PyTorch</span></div></div>
<div class="stack-art"><div class="art-heading"><span>RUDA / COMPUTE STACK</span><span class="cross">+</span></div><div class="stack-layers">{stack_html}</div><div class="art-bottom"><span>GPU KERNEL → TENSOR → MODEL</span><span aria-hidden="true">[ R ]</span></div></div></section>
<section class="entry-section"><div class="section-heading"><p class="eyebrow">01 / BUILD</p><h2>{t['choose']}</h2><p>{t['chooseLead']}</p></div><div class="path-grid">{cards}</div></section>
<section class="compute-section"><div class="section-heading"><p class="eyebrow">02 / COMPUTE</p><h2>{t['compute']}</h2><p>{t['stackLead']}</p></div><div class="library-grid">{library_html}</div><a class="text-link" href="{self.doc(lang, 'libraries/index')}">{t['allDocs']} →</a></section>
<section class="quick-section"><div><p class="eyebrow">03 / GET STARTED</p><h2>{t['quick']}</h2><p>{t['quickDesc']}</p><a class="text-link" href="{self.doc(lang, 'getting-started')}">{t['start']} →</a></div><div class="terminal"><div class="terminal-bar"><span class="terminal-dots">● ● ●</span><span>TERMINAL</span></div><pre><code>git clone https://github.com/shuqi2077/RUDA.git
cd RUDA</code></pre><p>{t['installNote']}</p></div></section>
<section class="closing"><h2>{t['stackTitle']}</h2><a class="button primary" href="{self.doc(lang, 'index')}">{t['allDocs']} ↗</a></section></main>'''
        return self.shell(lang, "RUDA — " + t["footer"], body, self.url(lang + "/"))

    def build(self):
        self.load()
        for lang in COPY:
            self.files[f"{lang}/index.html"] = self.homepage(lang)
            t = COPY[lang]
            body = f'''<main id="main" class="search-main"><p class="eyebrow">{t['docs']}</p><h1>{t['prompt']}</h1><p>{t['searchHelp']}</p><form role="search" id="search-form"><label for="search-q" class="sr-only">{t['search']}</label><input type="search" id="search-q" name="q" placeholder="{t['searchHint']}" autocomplete="off"><button class="button primary" type="submit">{t['search']}</button></form><p id="search-status" role="status" aria-live="polite"></p><div id="search-results"></div><noscript>JavaScript is required for full-text search. <a href="{self.doc(lang, 'index')}">{t['docs']}</a></noscript></main>'''
            self.files[f"{lang}/search/index.html"] = self.shell(lang, t["search"], body, self.url(lang + "/search/"), mode="search")
            self.files[f"assets/search-{lang}.json"] = json.dumps(self.index[lang], ensure_ascii=False, separators=(",", ":"))
        self.files["index.html"] = self.homepage("en").replace(ORIGIN + self.url("en/"), ORIGIN + self.url())
        for (lang, key), page in self.pages.items():
            t = COPY[lang]
            body = f'''<main id="main" class="doc-main"><div class="article-top"><span>{t['docs']} / {COPY[lang]['name']}</span><a href="{REPO}/blob/main/{page['source']}">{t['source']} ↗</a></div><article class="prose">{page['body']}</article></main><aside class="page-toc"><p>{t['toc']}</p>{page['toc']}</aside>'''
            self.files[f"{lang}/docs/{key}/index.html"] = self.shell(lang, page["title"], body, page["route"], key, "doc", self.sidebar(lang, key))
        t = COPY["en"]
        self.files["404.html"] = self.shell("en", t["notFound"], f'<main id="main" class="search-main"><p class="eyebrow">404</p><h1>{t["notFound"]}</h1><p>{t["notFoundText"]}</p><a class="button primary" href="{self.url()}">{t["home"]} →</a></main>', self.url("404.html"))
        urls = [self.url()] + [self.url(f"{l}/") for l in COPY] + [p["route"] for p in self.pages.values()]
        self.files["sitemap.xml"] = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(f"<url><loc>{ORIGIN}{esc(url)}</loc></url>" for url in urls) + '</urlset>'
        self.files[".nojekyll"] = ""
        self.files["assets/highlight.css"] = HtmlFormatter(style="github-dark").get_style_defs(".codehilite")

    def write(self, output):
        output = output.resolve()
        if output == ROOT or output in ROOT.parents:
            raise ValueError("Output must be a dedicated site directory")
        marker = output / ".ruda-site-files.json"
        if output.exists() and any(output.iterdir()) and not marker.is_file():
            raise ValueError("Refusing to overwrite a non-site directory")
        for asset in (WEB / "assets").iterdir():
            if asset.is_file():
                self.files["assets/" + asset.name] = asset.read_text(encoding="utf-8")
        if marker.exists():
            for old in json.loads(marker.read_text()):
                path = (output / old).resolve()
                if not path.is_relative_to(output):
                    raise ValueError("Invalid generated-file manifest")
                if old not in self.files and path.is_file():
                    path.unlink()
        for name, text in self.files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
        marker.write_text(json.dumps(sorted(self.files)), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/RUDA/")
    parser.add_argument("--output", type=Path, default=WEB / "_site")
    args = parser.parse_args()
    started = time.perf_counter()
    site = Site(args.base_path)
    site.build()
    site.write(args.output)
    print(f"Built {len(site.pages)} documentation pages, 5 homepages and 5 search indexes in {time.perf_counter()-started:.2f}s: {args.output}", flush=True)
