#!/usr/bin/env python3
"""
Download Executive Orders from FederalRegister.gov as plain-text .txt files

To install requirements: pip install requests beautifulsoup4 lxml
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup


BASE_PAGE = "https://www.federalregister.gov/presidential-documents/executive-orders"
OUTPUT_DIR = Path("executive_orders")
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_DOCS = 0.15
MAX_FETCH_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.6


def slugify(text: str, max_len: int = 140):
    text = text.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].rstrip("-") or "untitled"


def safe_filename(text: str, max_len: int = 180):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len].rstrip(" .") or "untitled"


def normalize_whitespace(text: str):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def maybe_fix_mojibake(text: str):
    # Common for older documents where UTF-8 bytes were decoded as latin-1.
    if "â" not in text and "Ã" not in text:
        return text

    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text

    return repaired if repaired.count("�") <= text.count("�") else text


def is_likely_site_chrome(text: str):
    lower = text.lower()
    markers = [
        "skip to content",
        "search the federal register",
        "reader aids",
        "enhanced content",
        "my fr",
        "site feedback",
        "request access",
        "published content - document details",
        "enable javascript and cookies to continue",
        "unblock.federalregister.gov",
    ]
    hits = sum(1 for m in markers if m in lower)
    return hits >= 3


def prepare_text_for_nlp(text: str):
    text = maybe_fix_mojibake(normalize_whitespace(text))

    boilerplate_substrings = (
        "skip to content",
        "document details",
        "published content - document details",
        "reader aids - executive order details",
        "enhanced content -",
        "search the federal register",
        "advanced document search",
        "public inspection search",
        "office of the federal register announcements",
        "using federalregister.gov",
        "understanding the federal register",
        "recent site updates",
        "federal register & cfr statistics",
        "videos & tutorials",
        "developer resources",
        "government policy and ofr procedures",
        "my account",
        "my clipboard",
        "my comments",
        "my subscriptions",
        "sign in / sign up",
        "shorter document url",
        "document page views are updated periodically",
        "this site displays a prototype",
        "request access",
    )
    nav_singletons = {
        "home",
        "sections",
        "browse",
        "agencies",
        "topics (cfr indexing terms)",
        "dates",
        "public inspection",
        "presidential documents",
        "search",
        "document search",
        "reader aids",
        "information",
        "about this site",
        "contact us",
        "privacy",
        "accessibility",
        "foia",
        "no fear act",
        "continuity information",
        "site feedback",
    }

    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue

        lower = line.lower()

        if any(snippet in lower for snippet in boilerplate_substrings):
            continue

        if lower in nav_singletons:
            continue

        # Federal Register documents sometimes contain standalone punctuation
        # lines (e.g. just '.') between section labels and headings.
        if re.fullmatch(r"[.\-_=~]+", line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def output_path_for_doc(output_dir: Path, doc: dict):
    eo_number = str(doc.get("executive_order_number") or "unknown")
    signing_date = (
        doc.get("signing_date") or doc.get("publication_date") or "unknown-date"
    )
    title = safe_filename(doc.get("title") or "untitled")
    filename = safe_filename(f"EO_{eo_number}__{signing_date}__{title}.txt")
    return output_dir / filename


def parse_query_info(url: str):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    def first(key: str, default: str = ""):
        return qs.get(key, [default])[0]

    return {
        "president": first("conditions[president]", "unknown-president"),
        "gte": first("conditions[signing_date][gte]", "unknown-start"),
        "lte": first("conditions[signing_date][lte]", "unknown-end"),
    }


def extract_bulk_json_links(session: requests.Session):
    resp = session.get(BASE_PAGE, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_url = urljoin(BASE_PAGE, href)

        # Bulk president JSON links on the EO page look like:
        # /documents/search.json?conditions[president]=...&conditions[presidential_document_type]=executive_order...
        if (
            "search.json" in abs_url
            and "conditions%5Bpresident%5D" in abs_url
            and "executive_order" in abs_url
        ):
            links.append(abs_url)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for link in links:
        if link not in seen:
            seen.add(link)
            deduped.append(link)

    return deduped


def fetch_bulk_results(session: requests.Session, bulk_url: str):
    # Convert the website's search.json URL to the API documents.json URL if needed.
    api_url = bulk_url.replace("/documents/search.json", "/api/v1/documents.json")

    resp = session.get(api_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    results = list(data.get("results", []))

    next_page_url = data.get("next_page_url")
    while next_page_url:
        resp = session.get(next_page_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        next_page_url = data.get("next_page_url")

    return results


def xml_to_text(xml_text: str):
    soup = BeautifulSoup(xml_text, "xml")
    text = soup.get_text("\n")
    return normalize_whitespace(text)


def html_to_text(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script/style/noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer document-body containers when present to avoid nav/chrome content.
    main = None
    for selector in (
        "#fulltext_content",
        "#fulltext",
        ".document-full-text",
        ".body-content",
        "article",
        "main",
    ):
        node = soup.select_one(selector)
        if node:
            main = node
            break

    text = (main or soup).get_text("\n")
    return normalize_whitespace(text)


def request_with_retries(session: requests.Session, url: str):
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            status = resp.status_code
            # Retry transient failures and rate limiting.
            if status == 429 or 500 <= status < 600:
                if attempt < MAX_FETCH_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None


def hydrate_doc_for_text_urls(session: requests.Session, doc: dict):
    doc_number = doc.get("document_number")
    if not doc_number:
        return doc

    detail_url = f"https://www.federalregister.gov/api/v1/documents/{doc_number}.json"
    resp = request_with_retries(session, detail_url)
    if not resp:
        return doc

    try:
        detailed = resp.json()
    except ValueError:
        return doc

    merged = dict(doc)
    for key in (
        "full_text_xml_url",
        "raw_text_url",
        "body_html_url",
        "html_url",
        "publication_date",
        "document_number",
    ):
        if detailed.get(key) and not merged.get(key):
            merged[key] = detailed.get(key)

    return merged


def fallback_full_text_urls(doc: dict):
    doc_number = doc.get("document_number")
    pub_date = doc.get("publication_date")
    if not doc_number or not pub_date:
        return []

    parts = str(pub_date).split("-")
    if len(parts) != 3:
        return []

    yyyy, mm, dd = parts
    return [
        f"https://www.federalregister.gov/documents/full_text/text/{yyyy}/{mm}/{dd}/{doc_number}.txt",
        f"https://www.federalregister.gov/documents/full_text/html/{yyyy}/{mm}/{dd}/{doc_number}.html",
    ]


def fetch_document_text(session: requests.Session, doc: dict):
    # Bulk results can omit text URLs for some older records, so hydrate first.
    doc = hydrate_doc_for_text_urls(session, doc)

    # Prefer XML when available, then raw/full-text, then page HTML.
    candidates = [
        doc.get("full_text_xml_url"),
        doc.get("raw_text_url"),
        doc.get("body_html_url"),
        doc.get("html_url"),
        *fallback_full_text_urls(doc),
    ]

    # Deduplicate while preserving order.
    seen = set()
    deduped_candidates = []
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        deduped_candidates.append(url)

    for url in deduped_candidates:
        resp = request_with_retries(session, url)
        if not resp:
            continue

        if "unblock.federalregister.gov" in resp.url:
            continue

        content_type = resp.headers.get("content-type", "").lower()

        if url.endswith(".xml") or "xml" in content_type:
            text = xml_to_text(resp.text)
        elif url.endswith(".txt") or "text/plain" in content_type:
            lowered = resp.text.lower()
            # Some older FR .txt endpoints are HTML pages with a <pre> body.
            if "<html" in lowered and "<pre" in lowered:
                text = html_to_text(resp.text)
            else:
                text = normalize_whitespace(resp.text)
        else:
            text = html_to_text(resp.text)

        if text and not is_likely_site_chrome(text):
            return text

    return ""


def make_presidency_dir_name(bulk_url: str):
    info = parse_query_info(bulk_url)
    president = slugify(unquote(info["president"]))
    gte = info["gte"]
    lte = info["lte"]
    return f"{president}__{gte}_to_{lte}"


def save_document(output_dir: Path, doc: dict, text: str):
    path = output_path_for_doc(output_dir, doc)
    path.write_text(prepare_text_for_nlp(text), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; EO-downloader/1.0; +https://www.federalregister.gov/)"
        }
    )

    bulk_links = extract_bulk_json_links(session)
    if not bulk_links:
        raise RuntimeError(
            "No bulk JSON presidency links found on the Executive Orders page."
        )

    print(f"Found {len(bulk_links)} presidency bulk JSON links.")

    for bulk_url in bulk_links:
        presidency_dir = OUTPUT_DIR / make_presidency_dir_name(bulk_url)
        presidency_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing presidency: {presidency_dir.name}")
        print(f"Bulk URL: {bulk_url}")

        docs = fetch_bulk_results(session, bulk_url)
        print(f"Found {len(docs)} executive orders.")

        for i, doc in enumerate(docs, start=1):
            eo_number = doc.get("executive_order_number", "unknown")
            title = doc.get("title", "untitled")
            print(f"  [{i}/{len(docs)}] EO {eo_number}: {title}")

            if output_path_for_doc(presidency_dir, doc).exists():
                print("    -> skipped (already exists)")
                continue

            body_text = fetch_document_text(session, doc)
            if not body_text:
                print("    -> skipped (could not extract text)")
                continue

            save_document(presidency_dir, doc, body_text.strip())
            time.sleep(SLEEP_BETWEEN_DOCS)

    print(f"\nDone. Files saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
