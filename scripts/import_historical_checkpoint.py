from __future__ import annotations

import argparse
from pathlib import Path

from xau_lab.experiments.history import import_historical_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Import historical MT5 research evidence from outputs.zip")
    parser.add_argument("archive", type=Path, help="Path to outputs.zip")
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    frame, manifest = import_historical_checkpoint(args.archive, args.result_root)
    print(f"Imported {frame.height} historical evidence rows")
    print(f"Source SHA-256: {manifest['source_sha256']}")
    print(args.result_root / "HISTORICAL_EVIDENCE.csv")


if __name__ == "__main__":
    main()
