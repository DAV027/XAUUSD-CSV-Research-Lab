"""Profile explicitly selected experiments without writing campaign results."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import subprocess
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from xau_lab.runner import single
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.manifest import build_run_manifest
from xau_lab.strategies.base import StrategyDefinition


def peak_memory_bytes():
    if os.name != 'nt':
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--experiment-id', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    catalog = {e.experiment_id: e for e in _read_catalog(args.catalog)}
    experiments = [catalog[key] for key in args.experiment_id]
    manifest = build_run_manifest(feature_path=args.features, catalog_path=args.catalog,
        workers=1, campaign_seed=9_215_000, repo_root=Path(__file__).resolve().parents[1])
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([repo / 'pyproject.toml', *repo.glob('xau_lab/**/*.py'), *repo.glob('scripts/**/*.py')]):
        digest.update(path.relative_to(repo).as_posix().encode() + b'\0' + path.read_bytes())
    manifest['source_tree_sha256'] = digest.hexdigest()
    manifest['working_tree_dirty'] = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip())
    started = time.perf_counter()
    market = load_market_bundle(args.features)
    report = dict(manifest=manifest, market_rows=len(market.bars),
                  load_seconds=time.perf_counter()-started, experiments=[])
    generate = StrategyDefinition.generate
    backtest = single.run_fast_backtest
    summarize = single.summarize_trades
    for experiment in experiments:
        timings = {}
        def timed_generate(*a, **kw):
            start = time.perf_counter()
            result = generate(*a, **kw)
            timings['signal_generation_seconds'] = time.perf_counter()-start
            timings['nonzero_signals'] = int(np.count_nonzero(result))
            print(json.dumps(dict(experiment_id=experiment.experiment_id, stage='signals', **timings)), flush=True)
            return result
        def timed_backtest(*a, **kw):
            start = time.perf_counter()
            result = backtest(*a, **kw)
            timings['backtest_seconds'] = time.perf_counter()-start
            return result
        def timed_metrics(*a, **kw):
            start = time.perf_counter()
            result = summarize(*a, **kw)
            timings['metrics_seconds'] = time.perf_counter()-start
            return result
        start = time.perf_counter()
        with patch.object(StrategyDefinition, 'generate', timed_generate), \
             patch.object(single, 'run_fast_backtest', timed_backtest), \
             patch.object(single, 'summarize_trades', timed_metrics):
            outcome = single.run_experiment(experiment, market)
        timings['total_seconds'] = time.perf_counter()-start
        timings['process_lifetime_peak_working_set_bytes'] = peak_memory_bytes()
        row = dict(experiment_id=experiment.experiment_id, strategy=experiment.strategy_name,
                   allocation_bucket=experiment.allocation_bucket, timings=timings,
                   master_result=outcome.master_result)
        report['experiments'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(json.dumps(dict(experiment_id=experiment.experiment_id, **timings)), flush=True)


if __name__ == '__main__':
    main()
