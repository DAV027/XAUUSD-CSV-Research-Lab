import pytest

from xau_lab.backtest.models import Trade
from xau_lab.io.trades import TRADE_COLUMNS, trade_to_record, trades_to_frame


EXPECTED_COLUMNS = [
    "experiment_id",
    "parameter_set_id",
    "fingerprint",
    "signal_time",
    "entry_time",
    "direction",
    "signal_price",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "lot_size",
    "planned_risk_usd",
    "risk_R",
    "gross_pnl",
    "spread_cost",
    "commission",
    "slippage_cost",
    "net_pnl",
    "pnl_R",
    "hold_minutes",
    "broker_date",
    "session_asia",
    "session_london",
    "session_new_york",
    "session_overlap",
]


def _trade():
    return Trade(
        experiment_id="exp-1",
        parameter_set_id="set-1",
        fingerprint="abc123",
        signal_time=0,
        entry_time=60,
        direction=1,
        signal_price=2000.0,
        entry_price=2000.4,
        stop_price=1999.4,
        target_price=2002.4,
        exit_time=180,
        exit_price=2002.35,
        exit_reason="target",
        lot_size=0.02,
        planned_risk_usd=2.0,
        risk_R=1.0,
        gross_pnl=4.0,
        spread_cost=0.7,
        commission=0.12,
        slippage_cost=0.2,
        net_pnl=2.98,
        pnl_R=1.49,
        hold_minutes=2.0,
        broker_date="1970-01-01",
        session_asia=True,
        session_london=False,
        session_new_york=False,
        session_overlap=False,
        broker_timezone="UTC",
    )


def test_trade_record_uses_exact_ledger_order_and_iso_broker_timestamps():
    assert TRADE_COLUMNS == EXPECTED_COLUMNS
    record = trade_to_record(_trade())
    assert list(record) == EXPECTED_COLUMNS
    assert record["signal_time"] == "1970-01-01T00:00:00+00:00"
    assert record["entry_time"] == "1970-01-01T00:01:00+00:00"
    assert record["exit_time"] == "1970-01-01T00:03:00+00:00"
    assert record["broker_date"] == "1970-01-01"


def test_invalid_broker_timezone_fails_closed():
    trade = _trade()
    bad = Trade(**{**trade.__dict__, "broker_timezone": "Not/A_Timezone"})
    with pytest.raises(ValueError, match="invalid broker timezone"):
        trade_to_record(bad)


def test_trades_to_frame_uses_exact_column_order_when_polars_available():
    pl = pytest.importorskip("polars")
    frame = trades_to_frame([_trade()])
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == EXPECTED_COLUMNS
    assert frame.height == 1
