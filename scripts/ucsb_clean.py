# save as scripts/rebuild_ucsb_clean.py
import csv, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

IN_MANIFEST = Path("all_executive_orders_txt/manifest.csv")
OUT_ROOT = Path("all_executive_orders_txt_clean")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
TIMEOUT = 30
SLEEP = 0.2

DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)

NOISE_MARKERS = [
    "twitter",
    "facebook",
    "linkedin",
    "google+",
    "email",
    "simple search of our archives",
    "# per page",
    "apply",
    "attributes",
    "filed under",
    "categories",
]


def safe_dirname(v):
    v = re.sub(r"\s+", " ", (v or "").replace("\n", " ").strip(" ,.\t"))
    v = re.sub(r"[^\w\s&()'-]", "", v)
    v = re.sub(r"\s+", "_", v)
    return v or "Unknown_President"


def slugify(v):
    v = re.sub(r"[^\w\s-]", "", (v or "").lower())
    v = re.sub(r"[-\s]+", "-", v).strip("-_")
    return (v or "untitled")[:140]


def extract_president(soup, fallback):
    # Prefer canonical president profile link text
    a = soup.select_one('a[href^="/people/president/"]')
    if a:
        t = a.get_text(" ", strip=True)
        if t:
            return t

    text = soup.get_text("\n", strip=True)
    m = re.search(r"\n([A-Za-z .'\-()]+),\s+Executive Order", "\n" + text)
    if m:
        return m.group(1).strip()

    # Last non-empty line from fallback handles multiline dirty manifest fields
    parts = [p.strip() for p in (fallback or "").splitlines() if p.strip()]
    return parts[-1] if parts else "Unknown President"


def extract_body(soup, title):
    lines = [
        ln.strip() for ln in soup.get_text("\n", strip=True).splitlines() if ln.strip()
    ]
    if not lines:
        return ""

    end = len(lines)
    for i, ln in enumerate(lines):
        if "Online by Gerhard Peters and John T. Woolley" in ln:
            end = i
            break

    title_idxs = [i for i, ln in enumerate(lines[:end]) if ln == title]
    if not title_idxs:
        title_idxs = [
            i for i, ln in enumerate(lines[:end]) if ln.lower() == title.lower()
        ]

    candidates = []
    for idx in title_idxs or [0]:
        start = idx + 1

        # Skip nearby date if present
        for j in range(start, min(start + 8, end)):
            if DATE_RE.search(lines[j]):
                start = j + 1
                break

        cand = lines[start:end]
        if not cand:
            continue

        joined = "\n\n".join(cand).strip()
        lower = joined.lower()
        penalty = sum(1 for m in NOISE_MARKERS if m in lower)
        score = len(joined) - 400 * penalty
        candidates.append((score, joined))

    if not candidates:
        return ""

    best = max(candidates, key=lambda x: x[0])[1].strip()

    # Trim trailing citation fragment if present
    best = re.split(r"\n[A-Za-z .'\-()]+,\s+Executive Order", best)[0].strip()
    return best


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with IN_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (EO-rebuild)"})

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as outcsv:
        w = csv.DictWriter(
            outcsv, fieldnames=["title", "date", "president", "url", "local_path"]
        )
        w.writeheader()

        for i, row in enumerate(rows, 1):
            title = (row.get("title") or "").strip()
            date = (row.get("date") or "").strip()
            url = (row.get("url") or "").strip()
            fallback_pres = row.get("president") or ""

            if not url:
                continue

            try:
                r = session.get(url, timeout=TIMEOUT)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")

                president = extract_president(soup, fallback_pres)
                body = extract_body(soup, title)

                if not body:
                    continue

                pdir = OUT_ROOT / safe_dirname(president)
                pdir.mkdir(parents=True, exist_ok=True)
                fname = f"{date or 'undated'}__{slugify(title)}.txt"
                outpath = pdir / fname

                outpath.write_text(
                    f"{title}\n{date}\n{president}\n\n{body}\n", encoding="utf-8"
                )

                w.writerow(
                    {
                        "title": title,
                        "date": date,
                        "president": president,
                        "url": url,
                        "local_path": str(outpath),
                    }
                )

            except Exception:
                pass

            if i % 100 == 0:
                print(f"processed {i}/{len(rows)}")
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
