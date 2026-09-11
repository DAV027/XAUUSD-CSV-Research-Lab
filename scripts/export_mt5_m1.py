from pathlib import Path
import argparse

import MetaTrader5 as mt5

from xau_lab.data.mt5_export import export_all_m1
from xau_lab.data.schema import DataPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical XAUUSD M1 history from MT5")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--server-timezone", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = export_all_m1(
        mt5,
        args.symbol,
        DataPaths(args.root),
        args.server_timezone,
        overwrite=args.overwrite,
    )
    print(metadata)


if __name__ == "__main__":
    main()
