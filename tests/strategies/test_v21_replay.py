"""Synthetic execution contracts, not a replacement for the MT5 development ledger."""
import csv
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape
import zipfile

import polars as pl
import pytest


ROOT = Path(__file__).resolve().parents[2]
START = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
FIELDS = [
    "trade_index", "signal_bucket", "direction", "request_msc", "request_entry",
    "spread_points", "atr", "sl", "tp", "volume", "fill_msc", "fill_price",
    "exit_reason", "exit_msc", "exit_time", "exit_price", "price_profit",
    "commission", "swap", "net", "balance",
]


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("replay_sma_rsi_v21_standalone")


def _ticks(root, rows, day="2026-06-01"):
    path = root / f"broker_date={day}" / "ticks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=[("time_msc", pl.Int64), ("bid", pl.Float64),
                               ("ask", pl.Float64)], orient="row").write_parquet(path)
    return path


def _report(path, rows):
    """Minimal native-style worksheet, with real parser exercised by the CLI."""
    table = [["Deals"], ["Time", "Symbol", "Type", "Direction", "Volume", "Price",
                          "Commission", "Swap", "Profit", "Balance", "Comment"], *rows]
    xml_rows = []
    for number, values in enumerate(table, 1):
        cells = "".join(
            f'<c r="{chr(65 + col)}{number}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
            for col, value in enumerate(values)
        )
        xml_rows.append(f'<row r="{number}">{cells}</row>')
    xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           '<sheetData>' + "".join(xml_rows) + '</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/worksheets/sheet1.xml", xml)


@pytest.fixture
def scenario(tmp_path):
    ticks = tmp_path / "ticks"
    _ticks(ticks, [
        (START, 100.0, 100.2), (START + 249, 100.05, 100.25),
        (START + 250, 100.1, 100.3), (START + 251, 100.2, 100.4),
        (START + 300_000, 101.0, 101.2),  # request while BUY is open
        (START + 360_000, 102.0, 102.2),  # gap through BUY TP
        (START + 600_000, 100.0, 101.0),  # 100-point spread: rejected
        (START + 900_000, 100.0, 100.2),  # ATR makes volume below minimum
        (START + 1_200_000, 100.0, 100.2),
        (START + 1_200_200, 99.9, 100.1),
        (START + 1_200_251, 99.8, 100.0),  # excluded from 250ms fill window
        (START + 1_260_000, 97.8, 98.0),  # SELL exits at ask
    ])
    signals = tmp_path / "signals.csv"
    signals.write_text(
        "entry_utc,direction,m5_atr14\n"
        "2026-06-01T00:00:00Z,BUY,1\n"
        "2026-06-01T00:05:00Z,BUY,1\n"
        "2026-06-01T00:10:00Z,BUY,1\n"
        "2026-06-01T00:15:00Z,BUY,100\n"
        "2026-06-01T00:20:00Z,SELL,1\n", encoding="utf-8",
    )
    report = tmp_path / "report.xlsx"
    rows = [
        ["2026.06.01 00:00:00", "XAUUSD.r", "buy", "in", .06, 100.3, -.36, 0, 0, 4999.64, "SMA_RSI_EXACT_BUY"],
        ["2026.06.01 00:06:00", "XAUUSD.r", "sell", "out", .06, 102, 0, 0, 10.2, 5009.84, "tp 101.70"],
        ["2026.06.01 00:20:00", "XAUUSD.r", "sell", "in", .06, 99.9, -.36, 0, 0, 5009.48, "SMA_RSI_EXACT_SELL"],
        ["2026.06.01 00:21:00", "XAUUSD.r", "buy", "out", .06, 98, 0, 0, 11.4, 5020.88, "tp 98.50"],
    ]
    _report(report, rows)
    return dict(ticks=ticks, signals=signals, report=report, rows=rows,
                output=tmp_path / "replay.csv", summary=tmp_path / "summary.json")


def _run(scenario):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/replay_sma_rsi_v21_standalone.py"),
         "--python-signals", str(scenario["signals"]), "--ticks-root", str(scenario["ticks"]),
         "--mt5-report", str(scenario["report"]), "--output", str(scenario["output"]),
         "--summary", str(scenario["summary"])],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    with scenario["output"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, json.loads(scenario["summary"].read_text(encoding="utf-8"))


def test_synthetic_cli_replay_preserves_execution_and_is_deterministic(scenario):
    rows, summary = _run(scenario)
    assert summary["python_raw_signals"] == 5
    assert summary["standalone_executed_trades"] == 2
    assert summary["skip_counts"] == dict(SPREAD=1, POSITION=1, VOLUME=1, NO_TICK=0, AMBIGUOUS=0)
    assert summary["standalone_replay_parity_pass"] is True
    assert summary["standalone_final_balance"] == 5020.88
    assert [r["direction"] for r in rows] == ["BUY", "SELL"]
    assert [int(r["fill_msc"]) for r in rows] == [START + 250, START + 1_200_200]
    assert [float(r["fill_price"]) for r in rows] == [100.3, 99.9]
    assert [float(r["sl"]) for r in rows] == [98.7, 101.5]
    assert [float(r["tp"]) for r in rows] == [101.7, 98.5]
    assert [float(r["exit_price"]) for r in rows] == [102., 98.]
    assert [float(r["volume"]) for r in rows] == [.06, .06]
    assert [float(r["commission"]) for r in rows] == [-.36, -.36]
    assert [float(r["net"]) for r in rows] == [9.84, 11.04]
    assert list(rows[0]) == FIELDS
    before = (scenario["output"].read_bytes(), scenario["summary"].read_bytes())
    _run(scenario)
    assert before == (scenario["output"].read_bytes(), scenario["summary"].read_bytes())


def test_report_price_cannot_drive_replay_and_one_cent_still_fails_parity(scenario):
    _run(scenario)
    original = scenario["output"].read_bytes()
    scenario["rows"][1][5] = 102.01
    _report(scenario["report"], scenario["rows"])
    _, summary = _run(scenario)
    assert scenario["output"].read_bytes() == original
    assert summary["exit_price_mismatches"] == 1
    assert summary["standalone_replay_parity_pass"] is False


def test_zero_trade_run_replaces_stale_csv_with_header(scenario):
    _run(scenario)
    scenario["signals"].write_text("entry_utc,direction,m5_atr14\n", encoding="utf-8")
    rows, summary = _run(scenario)
    assert summary["standalone_executed_trades"] == 0
    assert rows == []
    assert scenario["output"].read_text().strip().split(",") == FIELDS


@pytest.mark.parametrize("raw,expected", [(.009, 0), (.01, .01), (.019, .01), (.02, .02), (101, 100)])
def test_volume_floor_boundaries(replay, raw, expected):
    assert replay._floor_volume(raw) == expected


def test_risk_compounds_only_with_current_flat_balance(replay):
    assert replay._expected_volume(5000, 100, 99) == .10
    assert replay._expected_volume(5500, 100, 99) == .11
    assert replay._expected_volume(5000, 100, 100) == 0
    assert replay._expected_volume(5000, 100, 0) == 0


@pytest.mark.parametrize("value,expected", [(1.005, 1.01), (-1.005, -1.01), (100.125, 100.13)])
def test_frozen_decimal_rounding(replay, value, expected):
    assert replay._round_price(value) == expected
    assert replay._round_money(value) == expected


@pytest.mark.parametrize("direction,entry,exit_,expected", [
    ("BUY", "2026-06-03T23:59:59", "2026-06-04T00:00:00", -1.74),
    ("SELL", "2026-06-03T23:59:59", "2026-06-04T00:00:00", 1.11),
    ("BUY", "2026-06-05T23:59:59", "2026-06-08T00:00:00", -.58),
    ("BUY", "2026-06-01T12:00:00", "2026-06-01T23:59:59", 0),
])
def test_swap_midnight_triple_and_weekend_contract(replay, direction, entry, exit_, expected):
    assert replay._swap_for_trade(direction, .01, datetime.fromisoformat(entry),
                                  datetime.fromisoformat(exit_)) == expected


def test_tick_window_boundaries_stable_ties_and_cache_eviction(replay, tmp_path):
    first = _ticks(tmp_path, [(START+2, 102., 102.2), (START, 100., 100.2),
                              (START+2, 103., 103.2)])
    second = _ticks(tmp_path, [(START+86_400_000, 104., 104.2)], "2026-06-02")
    store = replay.TickStore(tmp_path, max_cached_partitions=1)
    assert store.window(START, START+2)["bid"].to_list() == [100, 102, 103]
    assert store.last_tick_at_or_before(START, START+2)["bid"] == 103
    assert store.first_tick_from(START, START)["bid"] == 100
    assert store.first_tick_from(START+3, START+4) is None
    assert store.last_available_tick()["bid"] == 104
    assert list(store.cache) == [second]
    assert store.window(START, START+2)["bid"].to_list() == [100, 102, 103]
    assert list(store.cache) == [first]


@pytest.mark.parametrize("direction,sl,tp,quotes,reason", [
    ("BUY", 99, 101, [(100, 102), (98, 100)], "SL"),
    ("SELL", 101, 99, [(98, 100), (97, 98)], "TP"),
    ("BUY", 99, 101, [(100, 100.2), (102, 102.2)], "TP"),
    ("SELL", 101, 99, [(100, 100.2), (100.9, 101.1)], "SL"),
])
def test_exit_uses_correct_quote_side(replay, tmp_path, direction, sl, tp, quotes, reason):
    _ticks(tmp_path, [(START+i, *quote) for i, quote in enumerate(quotes)])
    found, tick = replay._first_exit_tick(replay.TickStore(tmp_path), direction, sl, tp,
                                          START-1, START+1)
    assert found == reason
    assert tick["time_msc"] == START+1


def test_standalone_timestamp_parser_normalizes_offsets(replay):
    assert replay._parse_python_time("2026-06-01T03:00:00+03:00") == datetime(2026, 6, 1)
    assert replay._parse_python_time("2026-06-01T00:00:00Z") == datetime(2026, 6, 1)


def test_frozen_execution_constants(replay):
    assert (replay.POINT, replay.DIGITS, replay.CONTRACT_SIZE) == (.01, 2, 100.)
    assert (replay.RISK_PERCENT, replay.VOLUME_MIN, replay.VOLUME_MAX, replay.VOLUME_STEP) == (.20, .01, 100., .01)
    assert (replay.MAX_SPREAD_POINTS, replay.ATR_MULTIPLIER, replay.DELAY_MS) == (80, 1.5, 250)
    assert (replay.COMMISSION_PER_LOT_ENTRY, replay.SWAP_LONG_POINTS, replay.SWAP_SHORT_POINTS) == (-6., -57.849, 36.963)


def test_csv_write_failure_preserves_prior_file_and_cleans_temp(replay, tmp_path):
    output = tmp_path / "replay.csv"
    output.write_text("previous complete artifact", encoding="utf-8")
    with pytest.raises(ValueError):
        replay._write_replay_csv(output, [{"unexpected_column": 1}])
    assert output.read_text() == "previous complete artifact"
    assert list(tmp_path.iterdir()) == [output]


def test_populated_csv_serialization_is_byte_identical(replay, tmp_path):
    row = dict.fromkeys(FIELDS, "")
    row.update(trade_index=1, direction="BUY", volume=.06, balance=5009.84)
    legacy = io.StringIO(newline="")
    writer = csv.DictWriter(legacy, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    path = tmp_path / "replay.csv"
    replay._write_replay_csv(path, [row])
    assert path.read_bytes() == legacy.getvalue().encode("utf-8")
