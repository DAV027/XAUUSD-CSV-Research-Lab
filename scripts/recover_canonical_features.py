from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import polars as pl

from scripts.build_features import load_session_config
from xau_lab.data.features import build_shared_features, filter_research_window
from xau_lab.data.schema import DataPaths


CANONICAL_RESEARCH_ROWS = 3_403_685
CANONICAL_FEATURE_SHA256 = "5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF"


def select_research_prefix(frame: pl.DataFrame, *, expected_rows: int = CANONICAL_RESEARCH_ROWS) -> pl.DataFrame:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    research = filter_research_window(frame)
    if research.height < expected_rows:
        raise ValueError(
            f"expected at least {expected_rows} research rows from 2017-01-01, "
            f"found {research.height}"
        )
    return research.head(expected_rows)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_if_sha_matches(candidate: Path, canonical: Path, expected_sha256: str) -> str:
    candidate = Path(candidate)
    canonical = Path(canonical)
    observed = file_sha256(candidate)
    expected = expected_sha256.strip().lower()
    if observed.lower() != expected:
        raise RuntimeError(
            "candidate SHA256 mismatch; canonical artifact was not modified: "
            f"expected={expected} observed={observed} candidate={candidate}"
        )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    candidate.replace(canonical)
    return observed


def recover_canonical_features(
    root: Path,
    *,
    session_config: Path | None = None,
    expected_rows: int = CANONICAL_RESEARCH_ROWS,
    expected_sha256: str = CANONICAL_FEATURE_SHA256,
) -> str:
    paths = DataPaths(Path(root))
    if not paths.clean_csv.exists():
        raise FileNotFoundError(f"clean data not found: {paths.clean_csv}")

    clean = pl.read_csv(paths.clean_csv, try_parse_dates=False)
    research = select_research_prefix(clean, expected_rows=expected_rows)
    config = load_session_config(session_config)
    features = build_shared_features(research, session_config=config)

    if features.height != expected_rows:
        raise RuntimeError(
            f"recovery produced {features.height} rows; expected exactly {expected_rows}"
        )

    paths.features_parquet.parent.mkdir(parents=True, exist_ok=True)
    candidate = paths.features_parquet.with_name(paths.features_parquet.name + ".recovery_candidate")
    temp = candidate.with_name(candidate.name + ".tmp")
    features.write_parquet(temp, compression="zstd")
    temp.replace(candidate)

    return promote_if_sha_matches(candidate, paths.features_parquet, expected_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the frozen canonical XAUUSD feature artifact from clean MT5 history. "
            "The canonical target is replaced only when the rebuilt parquet matches the "
            "known SHA256 exactly."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session-config", type=Path)
    args = parser.parse_args()

    observed = recover_canonical_features(args.root, session_config=args.session_config)
    print(f"canonical feature recovery verified: SHA256={observed.upper()}")


if __name__ == "__main__":
    main()
