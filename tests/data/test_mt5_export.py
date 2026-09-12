from types import SimpleNamespace
import time as stdlib_time

import numpy as np
import polars as pl
import pytest

from xau_lab.data.mt5_export import export_all_m1
from xau_lab.data.schema import DataPaths


class FakeProvider:
    TIMEFRAME_M1 = 1

    def __init__(self):
        self.shutdown_called = False

    def initialize(self):
        return True

    def shutdown(self):
        self.shutdown_called = True

    def symbol_info(self, symbol):
        return SimpleNamespace(
            digits=2,
            point=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            currency_profit="USD",
        )

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if start_pos > 0:
            return np.array([], dtype=[])
        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i8"),
            ("real_volume", "i8"),
        ]
        return np.array(
            [
                (1767355200, 2000, 2002, 1999, 2001, 10, 35, 0),
                (1767355260, 2001, 2003, 2000, 2002, 12, 34, 0),
            ],
            dtype=dtype,
        )


class CappedHistoryProvider(FakeProvider):
    def terminal_info(self):
        return SimpleNamespace(maxbars=3)


class NoneAtNaturalEndProvider(FakeProvider):
    def terminal_info(self):
        return SimpleNamespace(maxbars=1000)

    def last_error(self):
        return (-4, "No history")

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if start_pos > 0:
            return None
        return super().copy_rates_from_pos(symbol, timeframe, start_pos, count)


class NoneAtCapProvider(FakeProvider):
    def terminal_info(self):
        return SimpleNamespace(maxbars=2)

    def last_error(self):
        return (-4, "No history")

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if start_pos > 0:
            return None
        return super().copy_rates_from_pos(symbol, timeframe, start_pos, count)


class StableShortFinalChunkProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def terminal_info(self):
        return SimpleNamespace(maxbars=1000)

    def last_error(self):
        return (-1, "Terminal: Call failed")

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.calls.append(start_pos)
        if start_pos == 0:
            return super().copy_rates_from_pos(symbol, timeframe, start_pos, count)
        return None


class GrowingShortChunkProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def terminal_info(self):
        return SimpleNamespace(maxbars=1000)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        assert start_pos == 0
        self.calls += 1
        base = super().copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if self.calls == 1:
            return base
        extra = np.array(
            [(1767355320, 2002, 2004, 2001, 2003, 9, 33, 0)],
            dtype=base.dtype,
        )
        return np.concatenate([base, extra])


def test_export_writes_canonical_raw_csv_and_metadata(tmp_path):
    provider = FakeProvider()
    paths = DataPaths(tmp_path)

    meta = export_all_m1(provider, "XAUUSD", paths, server_timezone="Etc/UTC")

    out = pl.read_csv(paths.raw_csv)
    assert out.height == 2
    assert out.columns == [
        "time", "open", "high", "low", "close",
        "tick_volume", "spread", "real_volume",
    ]
    assert meta.symbol == "XAUUSD"
    assert meta.exported_rows == 2
    assert meta.server_timezone == "Etc/UTC"
    assert provider.shutdown_called is True
    assert paths.metadata_json.exists()


def test_export_refuses_to_overwrite_raw_data_by_default(tmp_path):
    paths = DataPaths(tmp_path)
    paths.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_csv.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_all_m1(FakeProvider(), "XAUUSD", paths, server_timezone="Etc/UTC")


def test_export_rejects_invalid_timezone_before_connecting(tmp_path):
    provider = FakeProvider()
    with pytest.raises(ValueError):
        export_all_m1(provider, "XAUUSD", DataPaths(tmp_path), server_timezone="Not/AZone")
    assert provider.shutdown_called is False


def test_export_rejects_history_pinned_to_terminal_max_bars(tmp_path):
    provider = CappedHistoryProvider()
    paths = DataPaths(tmp_path)

    with pytest.raises(RuntimeError, match="Max bars in chart"):
        export_all_m1(provider, "XAUUSD", paths, server_timezone="Etc/UTC")

    assert provider.shutdown_called is True
    assert not paths.raw_csv.exists()
    assert not paths.metadata_json.exists()


def test_export_treats_not_found_after_data_as_natural_history_end(tmp_path):
    provider = NoneAtNaturalEndProvider()
    paths = DataPaths(tmp_path)

    meta = export_all_m1(
        provider,
        "XAUUSD",
        paths,
        server_timezone="Etc/UTC",
        chunk_size=2,
    )

    assert meta.exported_rows == 2
    assert provider.shutdown_called is True
    assert paths.raw_csv.exists()
    assert paths.metadata_json.exists()


def test_export_reports_cap_when_none_occurs_at_terminal_max_bars(tmp_path):
    provider = NoneAtCapProvider()
    paths = DataPaths(tmp_path)

    with pytest.raises(RuntimeError, match="Max bars in chart"):
        export_all_m1(
            provider,
            "XAUUSD",
            paths,
            server_timezone="Etc/UTC",
            chunk_size=2,
        )

    assert provider.shutdown_called is True
    assert not paths.raw_csv.exists()
    assert not paths.metadata_json.exists()


def test_export_accepts_only_a_stable_short_final_chunk(tmp_path, monkeypatch):
    provider = StableShortFinalChunkProvider()
    paths = DataPaths(tmp_path)
    monkeypatch.setattr(stdlib_time, "sleep", lambda _: None)

    meta = export_all_m1(
        provider,
        "XAUUSD",
        paths,
        server_timezone="Etc/UTC",
        chunk_size=100_000,
    )

    assert meta.exported_rows == 2
    assert provider.calls == [0, 0, 0, 2]
    assert provider.shutdown_called is True
    assert paths.raw_csv.exists()
    assert paths.metadata_json.exists()


def test_export_rejects_short_chunk_when_history_is_still_changing(tmp_path, monkeypatch):
    provider = GrowingShortChunkProvider()
    paths = DataPaths(tmp_path)
    monkeypatch.setattr(stdlib_time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="history changed during short-chunk stability check"):
        export_all_m1(
            provider,
            "XAUUSD",
            paths,
            server_timezone="Etc/UTC",
            chunk_size=100_000,
        )

    assert provider.calls == 2
    assert provider.shutdown_called is True
    assert not paths.raw_csv.exists()
    assert not paths.metadata_json.exists()
