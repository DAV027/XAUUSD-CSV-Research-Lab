from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


BLS_CPI_ARCHIVE = "https://www.bls.gov/bls/news-release/cpi.htm"
BLS_EMPSIT_ARCHIVE = "https://www.bls.gov/bls/news-release/empsit.htm"
BEA_BASE = "https://www.bea.gov"
NY = ZoneInfo("America/New_York")

OUTPUT_COLUMNS = [
    "event_id",
    "timestamp_utc",
    "event_family",
    "event_name",
    "actual",
    "consensus",
    "previous",
    "unit",
    "source",
]


class CollectError(RuntimeError):
    pass


class _TextAndLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.links: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _html_text_and_links(raw: str) -> tuple[str, list[str]]:
    parser = _TextAndLinks()
    parser.feed(raw)
    text = " ".join(parser.text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, parser.links


def _fetch(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "XAUUSD-CSV-Research-Lab H16 research collector "
                "(archived official releases)"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _signed_change(verb: str, number: str) -> float:
    value = float(number.replace(",", ""))
    v = verb.casefold()
    if v in {"decreased", "declined", "fell", "dropped"}:
        return -value
    if v in {"unchanged", "was unchanged"}:
        return 0.0
    return value


def _parse_release_timestamp(text: str) -> datetime:
    pattern = re.compile(
        r"embargoed until(?: release at)?\s+"
        r"(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)\s*"
        r"(?:\([A-Z]{2,3}\)|[A-Z]{2,3})?,?\s*"
        r"(?:[A-Za-z]+,\s*)?"
        r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        raise CollectError("could not parse official release timestamp")

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3).casefold().replace(".", "")
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    local = datetime.strptime(
        f"{match.group(4)} {match.group(5)} {match.group(6)} {hour}:{minute}",
        "%B %d %Y %H:%M",
    ).replace(tzinfo=NY)
    return local.astimezone(timezone.utc)


def _parse_cpi(text: str) -> dict[str, float]:
    headline = re.search(
        r"Consumer Price Index for All Urban Consumers\s*\(CPI-U\)\s*"
        r"(increased|decreased|rose|declined|fell|was unchanged|unchanged)"
        r"(?:\s+by)?\s+([0-9]+(?:\.[0-9]+)?)\s+percent\s+in\s+[A-Za-z]+"
        r".{0,100}?seasonally adjusted",
        text,
        re.IGNORECASE,
    )
    core = re.search(
        r"(?:index for )?all items less food and energy\s+"
        r"(increased|decreased|rose|declined|fell|was unchanged|unchanged)"
        r"(?:\s+by)?\s+([0-9]+(?:\.[0-9]+)?)\s+percent\s+in\s+[A-Za-z]+",
        text,
        re.IGNORECASE,
    )
    if headline is None or core is None:
        raise CollectError("could not parse CPI/Core CPI monthly release values")
    return {
        "CPI": _signed_change(headline.group(1), headline.group(2)),
        "CORE_CPI": _signed_change(core.group(1), core.group(2)),
    }


def _parse_employment(text: str) -> dict[str, float]:
    payroll = re.search(
        r"Total nonfarm payroll employment\s+"
        r"(increased|decreased|rose|declined|fell|was unchanged|unchanged)"
        r"(?:\s+by)?\s+([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    unemployment = re.search(
        r"unemployment rate.{0,80}?([0-9]+(?:\.[0-9]+)?)\s+percent",
        text,
        re.IGNORECASE,
    )
    if payroll is None or unemployment is None:
        raise CollectError("could not parse NFP/unemployment release values")
    return {
        "NFP": _signed_change(payroll.group(1), payroll.group(2)) / 1000.0,
        "UNEMPLOYMENT": float(unemployment.group(1)),
    }


def _parse_pce(text: str) -> dict[str, float]:
    headline = re.search(
        r"(?:From the preceding month,\s+)?the PCE price index(?:\s+for\s+[A-Za-z]+)?\s+"
        r"(increased|decreased|rose|declined|fell|was unchanged|unchanged)"
        r"(?:\s+by)?\s+([0-9]+(?:\.[0-9]+)?)\s+percent",
        text,
        re.IGNORECASE,
    )
    core = re.search(
        r"Excluding food and energy,\s+the PCE price index\s+"
        r"(increased|decreased|rose|declined|fell|was unchanged|unchanged)"
        r"(?:\s+by)?\s+([0-9]+(?:\.[0-9]+)?)\s+percent",
        text,
        re.IGNORECASE,
    )
    if headline is None or core is None:
        raise CollectError("could not parse PCE/Core PCE monthly release values")
    return {
        "PCE": _signed_change(headline.group(1), headline.group(2)),
        "CORE_PCE": _signed_change(core.group(1), core.group(2)),
    }


def _row(event_family: str, ts: datetime, actual: float, source: str) -> dict[str, str]:
    names = {
        "CPI": "US CPI m/m, seasonally adjusted",
        "CORE_CPI": "US Core CPI m/m, seasonally adjusted",
        "NFP": "US Total Nonfarm Payroll Employment change",
        "UNEMPLOYMENT": "US Unemployment Rate",
        "PCE": "US PCE Price Index m/m",
        "CORE_PCE": "US Core PCE Price Index m/m",
    }
    units = {
        "CPI": "% m/m",
        "CORE_CPI": "% m/m",
        "NFP": "k",
        "UNEMPLOYMENT": "%",
        "PCE": "% m/m",
        "CORE_PCE": "% m/m",
    }
    date_id = ts.date().isoformat()
    return {
        "event_id": f"{date_id}_{event_family}",
        "timestamp_utc": ts.isoformat().replace("+00:00", "Z"),
        "event_family": event_family,
        "event_name": names[event_family],
        "actual": format(actual, ".12g"),
        "consensus": "",
        "previous": "",
        "unit": units[event_family],
        "source": source,
    }


def _archive_date_from_href(href: str, prefix: str) -> date | None:
    match = re.search(
        rf"{prefix}_(\d{{2}})(\d{{2}})(\d{{4}})\.(?:pdf|htm)",
        href,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return date(int(match.group(3)), int(match.group(1)), int(match.group(2)))


def _discover_bls_urls(
    index_url: str,
    prefix: str,
    start: date,
    end: date,
) -> list[str]:
    raw = _fetch(index_url)
    _, links = _html_text_and_links(raw)
    urls: set[str] = set()
    for href in links:
        archive_date = _archive_date_from_href(href, prefix)
        if archive_date is None or not (start <= archive_date < end):
            continue
        absolute = urljoin(index_url, href)
        absolute = re.sub(r"\.pdf(?:\?.*)?$", ".htm", absolute, flags=re.IGNORECASE)
        urls.add(absolute)
    return sorted(urls)


def _month_iter(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def _bea_candidate_urls() -> list[str]:
    month_names = [
        "",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    urls: list[str] = []
    # Dec-2022 is the first normal monthly PCE release occurring in Jan-2023.
    # Jul-2025 is the last normal monthly period expected to release before Sep-2025.
    for year, month in _month_iter(2022, 12, 2025, 7):
        release_year = year + 1 if month == 12 else year
        slug = f"personal-income-and-outlays-{month_names[month]}-{year}"
        urls.append(f"{BEA_BASE}/news/{release_year}/{slug}")
    return urls


def collect_development_actuals(
    config: dict,
    sleep_seconds: float = 0.1,
) -> list[dict[str, str]]:
    start = date.fromisoformat(config["development_period"]["from"])
    end = date.fromisoformat(config["final_oos"]["from"])
    start_ts = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_ts = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)

    rows: list[dict[str, str]] = []

    for index_url, prefix, parser in [
        (BLS_CPI_ARCHIVE, "cpi", _parse_cpi),
        (BLS_EMPSIT_ARCHIVE, "empsit", _parse_employment),
    ]:
        urls = _discover_bls_urls(index_url, prefix, start, end)
        for url in urls:
            text, _ = _html_text_and_links(_fetch(url))
            ts = _parse_release_timestamp(text)
            if not (start_ts <= ts < end_ts):
                continue
            values = parser(text)
            for family, actual in values.items():
                rows.append(_row(family, ts, actual, url))
            time.sleep(sleep_seconds)

    for url in _bea_candidate_urls():
        try:
            text, _ = _html_text_and_links(_fetch(url))
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        ts = _parse_release_timestamp(text)
        if not (start_ts <= ts < end_ts):
            continue
        values = _parse_pce(text)
        for family, actual in values.items():
            rows.append(_row(family, ts, actual, url))
        time.sleep(sleep_seconds)

    rows.sort(key=lambda row: (row["timestamp_utc"], row["event_family"]))
    ids = [row["event_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise CollectError("duplicate event_id generated by official collector")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect H16 development-period actuals from archived BLS/BEA "
            "release pages. Consensus is intentionally left blank."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/h16_macro_surprise_v1_frozen.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/macro/h16_official_actuals.csv"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = collect_development_actuals(
        config,
        sleep_seconds=args.sleep_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["event_family"]] = counts.get(row["event_family"], 0) + 1
    print(
        json.dumps(
            {
                "rows": len(rows),
                "counts": counts,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
