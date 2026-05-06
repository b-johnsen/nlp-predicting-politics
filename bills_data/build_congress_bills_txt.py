#!/usr/bin/env python3
"""Build a party-labeled TXT dataset of congressional bills from GovInfo bulkdata.

This script:
1. Enumerates BILLS and BILLSTATUS files using GovInfo bulkdata sitemap indexes.
2. Selects one text version per bill (prefer introduced version: ih/is).
3. Reads sponsor party from BILLSTATUS XML.
4. Extracts plain text from BILLS XML.
5. Writes output files grouped by label:

    <output_dir>/<label>/<congress>/<bill_type>/<package_id>.txt

It also writes:
    - <output_dir>/manifest.csv
    - <output_dir>/failures.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_OUTPUT_DIR = Path("congress_bills_txt_113_119")
DEFAULT_START_CONGRESS = 113
DEFAULT_END_CONGRESS = 119
DEFAULT_BILL_TYPES = ["hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"]
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 45
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF = 1.0

BILLS_SITEMAP_INDEX = "https://www.govinfo.gov/sitemap/bulkdata/BILLS/sitemapindex.xml"
BILLSTATUS_SITEMAP_INDEX = (
    "https://www.govinfo.gov/sitemap/bulkdata/BILLSTATUS/sitemapindex.xml"
)

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

BILLS_SITEMAP_RE = re.compile(
    r"^https://www\.govinfo\.gov/sitemap/bulkdata/BILLS/(?P<congress>\d+)(?P<bill_type>[a-z]+)/sitemap\.xml$"
)
BILLSTATUS_SITEMAP_RE = re.compile(
    r"^https://www\.govinfo\.gov/sitemap/bulkdata/BILLSTATUS/(?P<congress>\d+)(?P<bill_type>[a-z]+)/sitemap\.xml$"
)
BILLS_FILE_RE = re.compile(
    r"^https://www\.govinfo\.gov/bulkdata/BILLS/(?P<congress>\d+)/(?P<session>\d+)/(?P<bill_type>[a-z]+)/"
    r"BILLS-(?P=congress)(?P=bill_type)(?P<number>\d+)(?P<version>[a-z0-9]+)\.xml$"
)
BILLSTATUS_FILE_RE = re.compile(
    r"^https://www\.govinfo\.gov/bulkdata/BILLSTATUS/(?P<congress>\d+)/(?P<bill_type>[a-z]+)/"
    r"BILLSTATUS-(?P=congress)(?P=bill_type)(?P<number>\d+)\.xml$"
)


@dataclass(frozen=True)
class BillKey:
    congress: int
    bill_type: str
    number: int


@dataclass(frozen=True)
class BillTextVersion:
    key: BillKey
    session: int
    version: str
    url: str


@dataclass(frozen=True)
class BillStatusInfo:
    sponsor_name: str
    sponsor_party: str
    introduced_date: str
    congress: str
    bill_type: str
    number: str


@dataclass(frozen=True)
class ProcessResult:
    manifest_row: dict[str, str] | None
    failure_row: dict[str, str] | None


thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GovInfo bill text XML for congresses 113-119, "
            "convert to .txt, and separate by sponsor party."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for party-separated .txt files and manifests.",
    )
    parser.add_argument(
        "--start-congress",
        type=int,
        default=DEFAULT_START_CONGRESS,
        help="Starting congress number (inclusive).",
    )
    parser.add_argument(
        "--end-congress",
        type=int,
        default=DEFAULT_END_CONGRESS,
        help="Ending congress number (inclusive).",
    )
    parser.add_argument(
        "--bill-types",
        type=str,
        default=",".join(DEFAULT_BILL_TYPES),
        help="Comma-separated bill types, e.g. hr,s,hjres,sjres,hconres,sconres,hres,sres",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker threads for per-bill downloads.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries for transient HTTP failures.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Base backoff seconds used with exponential retry.",
    )
    parser.add_argument(
        "--max-bills",
        type=int,
        default=0,
        help="If > 0, process at most this many bills (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing manifest/failure files.",
    )
    return parser.parse_args()


def get_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CSCI3349-BillsDatasetBuilder/1.0; "
                    "+https://www.govinfo.gov/bulkdata)"
                )
            }
        )
        thread_local.session = session
    return session


def fetch_text(
    url: str,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = get_session().get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"Transient HTTP status {response.status_code} for {url}"
                )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = retry_backoff * (2**attempt) + random.uniform(0.0, 0.5)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_sitemap_locs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    locs: list[str] = []
    for loc in root.findall(f".//{SITEMAP_NS}loc"):
        if loc.text and loc.text.strip():
            locs.append(loc.text.strip())
    return locs


def parse_bill_types(csv_text: str) -> list[str]:
    values = [part.strip().lower() for part in csv_text.split(",") if part.strip()]
    if not values:
        raise SystemExit("--bill-types cannot be empty.")
    return values


def discover_sitemaps(
    index_url: str,
    regex: re.Pattern[str],
    start_congress: int,
    end_congress: int,
    bill_types: set[str],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> dict[tuple[int, str], str]:
    xml_text = fetch_text(index_url, timeout, max_retries, retry_backoff)
    locs = parse_sitemap_locs(xml_text)

    selected: dict[tuple[int, str], str] = {}
    for loc in locs:
        match = regex.match(loc)
        if not match:
            continue
        congress = int(match.group("congress"))
        bill_type = match.group("bill_type").lower()
        if congress < start_congress or congress > end_congress:
            continue
        if bill_type not in bill_types:
            continue
        selected[(congress, bill_type)] = loc
    return selected


def introduced_version_for_type(bill_type: str) -> str:
    return "is" if bill_type.startswith("s") else "ih"


def version_rank(candidate: BillTextVersion) -> tuple[int, int, str]:
    preferred = introduced_version_for_type(candidate.key.bill_type)
    preference_penalty = 0 if candidate.version.lower() == preferred else 1
    return (preference_penalty, candidate.session, candidate.version)


def build_billstatus_map(
    sitemap_urls: Iterable[str],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> dict[BillKey, str]:
    status_map: dict[BillKey, str] = {}
    for idx, sitemap_url in enumerate(sorted(sitemap_urls), 1):
        xml_text = fetch_text(sitemap_url, timeout, max_retries, retry_backoff)
        locs = parse_sitemap_locs(xml_text)
        for loc in locs:
            match = BILLSTATUS_FILE_RE.match(loc)
            if not match:
                continue
            key = BillKey(
                congress=int(match.group("congress")),
                bill_type=match.group("bill_type").lower(),
                number=int(match.group("number")),
            )
            status_map[key] = loc

        print(
            f"Parsed BILLSTATUS sitemap {idx}: {sitemap_url} "
            f"(running status files: {len(status_map)})"
        )

    return status_map


def build_selected_bill_text_map(
    sitemap_urls: Iterable[str],
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> dict[BillKey, BillTextVersion]:
    selected: dict[BillKey, BillTextVersion] = {}
    for idx, sitemap_url in enumerate(sorted(sitemap_urls), 1):
        xml_text = fetch_text(sitemap_url, timeout, max_retries, retry_backoff)
        locs = parse_sitemap_locs(xml_text)

        for loc in locs:
            match = BILLS_FILE_RE.match(loc)
            if not match:
                continue
            candidate = BillTextVersion(
                key=BillKey(
                    congress=int(match.group("congress")),
                    bill_type=match.group("bill_type").lower(),
                    number=int(match.group("number")),
                ),
                session=int(match.group("session")),
                version=match.group("version").lower(),
                url=loc,
            )
            existing = selected.get(candidate.key)
            if existing is None or version_rank(candidate) < version_rank(existing):
                selected[candidate.key] = candidate

        print(
            f"Parsed BILLS sitemap {idx}: {sitemap_url} "
            f"(running selected bills: {len(selected)})"
        )

    return selected


def text_or_empty(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def parse_billstatus_info(xml_text: str) -> BillStatusInfo:
    root = ET.fromstring(xml_text)

    sponsor_item = root.find(".//sponsors/item")
    sponsor_name = text_or_empty(
        None if sponsor_item is None else sponsor_item.find("fullName")
    )
    sponsor_party = text_or_empty(
        None if sponsor_item is None else sponsor_item.find("party")
    )

    introduced_date = text_or_empty(root.find(".//introducedDate"))
    congress = text_or_empty(root.find(".//congress"))
    bill_type = text_or_empty(root.find(".//type")).lower()
    number = text_or_empty(root.find(".//number"))

    return BillStatusInfo(
        sponsor_name=sponsor_name,
        sponsor_party=sponsor_party,
        introduced_date=introduced_date,
        congress=congress,
        bill_type=bill_type,
        number=number,
    )


def normalize_label_from_party(party: str) -> str:
    party_code = (party or "").strip().upper()
    if party_code == "D":
        return "democrat"
    if party_code == "R":
        return "republican"
    return "independent_or_unknown"


def extract_bill_text(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    legis_body = root.find(".//legis-body")
    source = legis_body if legis_body is not None else root

    lines: list[str] = []
    for chunk in source.itertext():
        if not chunk:
            continue
        for line in chunk.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)

    return "\n".join(lines).strip()


def process_bill(
    key: BillKey,
    text_version: BillTextVersion,
    status_url: str,
    output_dir: Path,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> ProcessResult:
    bill_id = f"{key.bill_type}{key.number}-{key.congress}"
    package_id = (
        f"BILLS-{key.congress}{key.bill_type}{key.number}{text_version.version}"
    )

    try:
        billstatus_xml = fetch_text(status_url, timeout, max_retries, retry_backoff)
        info = parse_billstatus_info(billstatus_xml)
    except Exception as exc:  # noqa: BLE001
        return ProcessResult(
            manifest_row=None,
            failure_row={
                "bill_id": bill_id,
                "package_id": package_id,
                "stage": "billstatus",
                "url": status_url,
                "error": str(exc),
            },
        )

    try:
        bill_xml = fetch_text(text_version.url, timeout, max_retries, retry_backoff)
        plain_text = extract_bill_text(bill_xml)
        if not plain_text:
            raise ValueError("Extracted empty plain text")
    except Exception as exc:  # noqa: BLE001
        return ProcessResult(
            manifest_row=None,
            failure_row={
                "bill_id": bill_id,
                "package_id": package_id,
                "stage": "bill_text",
                "url": text_version.url,
                "error": str(exc),
            },
        )

    label = normalize_label_from_party(info.sponsor_party)
    rel_dir = Path(label) / str(key.congress) / key.bill_type
    destination = output_dir / rel_dir / f"{package_id}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(plain_text + "\n", encoding="utf-8")

    row = {
        "bill_id": bill_id,
        "package_id": package_id,
        "congress": str(key.congress),
        "bill_type": key.bill_type,
        "bill_number": str(key.number),
        "session": str(text_version.session),
        "version": text_version.version,
        "party_code": (info.sponsor_party or "").strip().upper(),
        "label": label,
        "sponsor_name": info.sponsor_name,
        "introduced_date": info.introduced_date,
        "bills_url": text_version.url,
        "billstatus_url": status_url,
        "output_path": destination.relative_to(output_dir).as_posix(),
        "char_count": str(len(plain_text)),
        "status": "written",
    }
    return ProcessResult(manifest_row=row, failure_row=None)


def write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "bill_id",
        "package_id",
        "congress",
        "bill_type",
        "bill_number",
        "session",
        "version",
        "party_code",
        "label",
        "sponsor_name",
        "introduced_date",
        "bills_url",
        "billstatus_url",
        "output_path",
        "char_count",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failures_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["bill_id", "package_id", "stage", "url", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_paths(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    failures_path = output_dir / "failures.csv"
    if not overwrite:
        for file_path in (manifest_path, failures_path):
            if file_path.exists():
                raise SystemExit(
                    f"File already exists: {file_path}. Use --overwrite to replace it."
                )
    return manifest_path, failures_path


def main() -> None:
    args = parse_args()
    if args.start_congress > args.end_congress:
        raise SystemExit("--start-congress must be <= --end-congress")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    bill_types = parse_bill_types(args.bill_types)
    bill_types_set = set(bill_types)

    manifest_path, failures_path = ensure_output_paths(args.output_dir, args.overwrite)

    print("Discovering eligible sitemap files...")
    bills_sitemaps = discover_sitemaps(
        index_url=BILLS_SITEMAP_INDEX,
        regex=BILLS_SITEMAP_RE,
        start_congress=args.start_congress,
        end_congress=args.end_congress,
        bill_types=bill_types_set,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
    status_sitemaps = discover_sitemaps(
        index_url=BILLSTATUS_SITEMAP_INDEX,
        regex=BILLSTATUS_SITEMAP_RE,
        start_congress=args.start_congress,
        end_congress=args.end_congress,
        bill_types=bill_types_set,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    if not bills_sitemaps:
        raise SystemExit(
            "No BILLS sitemaps matched requested congress range and bill types."
        )
    if not status_sitemaps:
        raise SystemExit(
            "No BILLSTATUS sitemaps matched requested congress range and bill types."
        )

    print(
        f"Selected BILLS sitemaps: {len(bills_sitemaps)} | "
        f"Selected BILLSTATUS sitemaps: {len(status_sitemaps)}"
    )

    print("Building BILLSTATUS URL map...")
    status_map = build_billstatus_map(
        sitemap_urls=status_sitemaps.values(),
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    print("Building selected bill text URL map...")
    selected_text_map = build_selected_bill_text_map(
        sitemap_urls=bills_sitemaps.values(),
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    candidate_keys = sorted(
        (key for key in selected_text_map if key in status_map),
        key=lambda k: (k.congress, k.bill_type, k.number),
    )
    if args.max_bills > 0:
        candidate_keys = candidate_keys[: args.max_bills]

    if not candidate_keys:
        raise SystemExit("No intersecting BILLS and BILLSTATUS records were found.")

    print(f"Processing bills: {len(candidate_keys)}")

    manifest_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_bill,
                key,
                selected_text_map[key],
                status_map[key],
                args.output_dir,
                args.timeout,
                args.max_retries,
                args.retry_backoff,
            ): key
            for key in candidate_keys
        }

        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.manifest_row is not None:
                manifest_rows.append(result.manifest_row)
            if result.failure_row is not None:
                failure_rows.append(result.failure_row)
            if idx % 200 == 0 or idx == len(futures):
                print(
                    f"Completed {idx}/{len(futures)} | "
                    f"written: {len(manifest_rows)} | failures: {len(failure_rows)}"
                )

    manifest_rows.sort(
        key=lambda r: (int(r["congress"]), r["bill_type"], int(r["bill_number"]))
    )
    failure_rows.sort(key=lambda r: (r["bill_id"], r["stage"]))

    write_manifest_csv(manifest_path, manifest_rows)
    write_failures_csv(failures_path, failure_rows)

    print("Done.")
    print(f"Output directory : {args.output_dir.resolve()}")
    print(f"Manifest rows    : {len(manifest_rows)}")
    print(f"Failure rows     : {len(failure_rows)}")
    print(f"Manifest path    : {manifest_path.resolve()}")
    print(f"Failures path    : {failures_path.resolve()}")


if __name__ == "__main__":
    main()
