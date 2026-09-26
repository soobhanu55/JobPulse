"""Collect German AI / ML / data student job postings from the Bundesagentur fuer Arbeit job search.

Uses the public Jobsuche API (the one behind arbeitsagentur.de), not LinkedIn/Indeed/StepStone,
whose terms forbid scraping. Raw detail JSON is cached per posting in data/raw/, so re-runs only
fetch new postings. Output: data/postings.csv (one row per unique posting).

    python scrape.py
"""
from __future__ import annotations

import base64
import csv
import json
import time
from pathlib import Path

import requests

API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc"
HEADERS = {"X-API-Key": "jobboerse-jobsuche", "User-Agent": "JobPulse research script (student project)"}
QUERIES = [
    "Werkstudent Künstliche Intelligenz", "Werkstudent Machine Learning", "Werkstudent Data Science",
    "Werkstudent KI", "Werkstudent AI", "Werkstudent Data Engineering", "Werkstudent LLM",
    "Praktikum Künstliche Intelligenz", "Praktikum Machine Learning", "Praktikum Data Science",
]
PAGE_SIZE, MAX_PAGES, PAUSE_S = 50, 10, 0.4
RAW = Path("data/raw")


def search(query: str) -> list[dict]:
    out = []
    for page in range(1, MAX_PAGES + 1):
        r = requests.get(f"{API}/v6/jobs", headers=HEADERS, params={"was": query, "size": PAGE_SIZE, "page": page}, timeout=30)
        r.raise_for_status()
        batch = r.json().get("ergebnisliste", [])
        out += batch
        time.sleep(PAUSE_S)
        if len(batch) < PAGE_SIZE:
            break
    return out


def details(refnr: str) -> dict | None:
    cache = RAW / f"{refnr}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    b64 = base64.b64encode(refnr.encode()).decode()
    r = requests.get(f"{API}/v4/jobdetails/{b64}", headers=HEADERS, timeout=30)
    time.sleep(PAUSE_S)
    if r.status_code != 200:
        return None  # withdrawn between search and detail fetch
    cache.write_text(r.text, encoding="utf-8")
    return r.json()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    hits: dict[str, dict] = {}
    for q in QUERIES:
        for job in search(q):
            hits.setdefault(job["referenznummer"], job)
        print(f"{q!r}: {len(hits)} unique so far")

    rows = []
    for i, (ref, job) in enumerate(hits.items(), 1):
        d = details(ref)
        if not d:
            continue
        loc = (job.get("stellenlokationen") or [{}])[0].get("adresse", {})
        rows.append({
            "refnr": ref,
            "title": job.get("stellenangebotsTitel", ""),
            "company": job.get("firma", ""),
            "city": loc.get("ort", ""),
            "region": loc.get("region", ""),
            "type": job.get("stellenangebotsart", ""),
            "homeoffice": job.get("homeofficemoeglich", False),
            "published": job.get("datumErsteVeroeffentlichung", ""),
            "external_url": job.get("externeURL", ""),
            "description": (d.get("stellenangebotsBeschreibung") or "").replace("\r", " "),
        })
        if i % 50 == 0:
            print(f"details {i}/{len(hits)}")

    with open("data/postings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} postings to data/postings.csv")


if __name__ == "__main__":
    main()
