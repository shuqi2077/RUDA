"use strict";
const config = JSON.parse(document.querySelector("#site-config").textContent);
document.querySelector("[data-language]")?.addEventListener("change", event => {
  const target = new URL(event.target.value, location.origin);
  if (location.pathname.includes("/search/")) target.search = location.search;
  location.assign(target.href);
});
const menu = document.querySelector("[data-menu]");
menu?.addEventListener("click", () => {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open));
  document.querySelector("#sidebar").classList.toggle("is-open", open);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && menu) {
    menu.setAttribute("aria-expanded", "false");
    document.querySelector("#sidebar").classList.remove("is-open");
  }
  if (event.key === "/" && !event.ctrlKey && !event.metaKey && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) && !event.target.isContentEditable) {
    event.preventDefault();
    (document.querySelector("#search-q") || document.querySelector("#header-q")).focus();
  }
});
document.querySelectorAll(".codehilite, .terminal").forEach(block => {
  const code = block.querySelector("pre");
  if (!code) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-button";
  button.textContent = config.copy;
  button.setAttribute("aria-label", config.copy);
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = config.copied;
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = config.copyError;
    }
    setTimeout(() => { button.textContent = config.copy; }, 1800);
  });
  block.append(button);
});

const form = document.querySelector("#search-form");
if (form) {
  const field = document.querySelector("#search-q");
  const status = document.querySelector("#search-status");
  const results = document.querySelector("#search-results");
  let indexPromise;
  let revision = 0;
  let timer;
  const normalize = text => text.normalize("NFKC").toLocaleLowerCase();
  async function search() {
    const current = ++revision;
    const query = field.value.trim();
    const address = new URL(location.href);
    if (query) address.searchParams.set("q", query); else address.searchParams.delete("q");
    history.replaceState(null, "", address);
    results.replaceChildren();
    status.textContent = query ? config.loading : "";
    if (!query) return;
    try {
      indexPromise ||= fetch(`${config.base}assets/search-${config.lang}.json`).then(response => {
        if (!response.ok) throw new Error("Search index unavailable");
        return response.json();
      }).then(pages => pages.map(page => ({...page, haystack: normalize(`${page.title} ${page.section} ${page.text}`)})));
      const index = await indexPromise;
      if (current !== revision) return;
      const terms = normalize(query).split(/\s+/).filter(Boolean);
      const matches = index.filter(page => terms.every(term => page.haystack.includes(term))).map(page => ({
        ...page, score: terms.reduce((score, term) => score + (normalize(page.section).includes(term) ? 8 : 0) + (normalize(page.title).includes(term) ? 4 : 0), 0)
      })).sort((a, b) => b.score - a.score);
      status.textContent = matches.length ? `${matches.length} ${config.results}` : config.empty;
      const fragment = document.createDocumentFragment();
      for (const match of matches) {
        const link = document.createElement("a");
        link.className = "search-result";
        link.href = match.url;
        const title = document.createElement("h2");
        title.textContent = match.section;
        const page = document.createElement("div");
        page.className = "result-page";
        page.textContent = match.title;
        const snippet = document.createElement("p");
        const offset = Math.max(0, normalize(match.text).indexOf(terms[0]) - 70);
        snippet.textContent = (offset ? "…" : "") + match.text.slice(offset, offset + 230) + (match.text.length > offset + 230 ? "…" : "");
        link.append(page, title, snippet);
        fragment.append(link);
      }
      results.append(fragment);
    } catch {
      indexPromise = undefined;
      if (current === revision) status.textContent = config.error;
    }
  }
  form.addEventListener("submit", event => { event.preventDefault(); clearTimeout(timer); search(); });
  field.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(search, 160); });
  field.value = new URLSearchParams(location.search).get("q") || "";
  search();
}
