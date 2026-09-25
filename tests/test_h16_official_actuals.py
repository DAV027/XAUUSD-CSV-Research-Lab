import runpy
from datetime import date


MODULE = runpy.run_path("scripts/collect_h16_official_actuals.py")
CollectError = MODULE["CollectError"]
_parse_cpi = MODULE["_parse_cpi"]
_parse_employment = MODULE["_parse_employment"]
_parse_pce = MODULE["_parse_pce"]
_parse_release_timestamp = MODULE["_parse_release_timestamp"]
_archive_date_from_href = MODULE["_archive_date_from_href"]
_row = MODULE["_row"]


def test_bls_standard_time_release_converts_to_utc():
    text = (
        "Transmission of material in this release is embargoed until "
        "8:30 a.m. (ET) Tuesday, February 13, 2024."
    )
    assert _parse_release_timestamp(text).isoformat() == "2024-02-13T13:30:00+00:00"


def test_bls_daylight_time_release_converts_to_utc():
    text = (
        "Transmission of material in this release is embargoed until "
        "8:30 a.m. (ET) Friday, June 7, 2024."
    )
    assert _parse_release_timestamp(text).isoformat() == "2024-06-07T12:30:00+00:00"


def test_cpi_parser_extracts_monthly_headline_and_core():
    text = (
        "The Consumer Price Index for All Urban Consumers (CPI-U) increased "
        "0.3 percent in January on a seasonally adjusted basis. "
        "The index for all items less food and energy increased "
        "0.4 percent in January."
    )
    assert _parse_cpi(text) == {"CPI": 0.3, "CORE_CPI": 0.4}


def test_employment_parser_outputs_nfp_in_thousands():
    text = (
        "Total nonfarm payroll employment increased by 216,000 in December, "
        "and the unemployment rate was unchanged at 3.7 percent."
    )
    assert _parse_employment(text) == {"NFP": 216.0, "UNEMPLOYMENT": 3.7}


def test_pce_parser_extracts_monthly_headline_and_core():
    text = (
        "The PCE price index increased 0.3 percent. "
        "Excluding food and energy, the PCE price index increased 0.4 percent."
    )
    assert _parse_pce(text) == {"PCE": 0.3, "CORE_PCE": 0.4}


def test_archive_date_accepts_pdf_and_html():
    assert _archive_date_from_href(
        "/news.release/archives/cpi_02132024.pdf",
        "cpi",
    ) == date(2024, 2, 13)
    assert _archive_date_from_href(
        "/news.release/archives/cpi_02132024.htm",
        "cpi",
    ) == date(2024, 2, 13)


def test_staging_row_leaves_consensus_blank():
    ts = _parse_release_timestamp(
        "EMBARGOED UNTIL RELEASE AT 8:30 a.m. EST, "
        "Thursday, February 29, 2024"
    )
    row = _row("NFP", ts, 216.0, "official-fixture")
    assert row["actual"] == "216"
    assert row["unit"] == "k"
    assert row["consensus"] == ""
