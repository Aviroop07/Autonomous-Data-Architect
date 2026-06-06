"""
Smoke test for Stage 4 generated synthesis code.

Executes the generated Python script at reduced scale (10% by default),
monitors memory usage, collects output DataFrames from written CSVs,
and cleans up all temporary files.
"""

import re
import sys
import time
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import psutil

SCALE_FACTOR_PATTERN = re.compile(r'^SCALE_FACTOR\s*=\s*[\d.]+', re.MULTILINE)
MEMORY_LIMIT_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
TIMEOUT_SECONDS = 120


def run_smoke_test(
    generated_code: str,
    scale_factor: float = 0.1,
) -> Tuple[bool, Dict[str, pd.DataFrame], List[str]]:
    """
    Execute the generated synthesis code at reduced scale for validation.

    Injects SCALE_FACTOR = <scale_factor>, redirects CSV output to a
    temporary directory, monitors memory (kills if > 1 GB), enforces a
    120-second timeout, and loads any written CSVs into DataFrames.

    Args:
        generated_code: The complete Python synthesis script as a string.
        scale_factor:   Fractional scale to use (default 0.1 = 10%).

    Returns:
        (success, dataframes, logs)
        - success:    True if the subprocess exited with returncode 0.
        - dataframes: Dict mapping table name -> DataFrame for each CSV written.
        - logs:       List of human-readable status / error messages.
    """
    logs: List[str] = []
    dataframes: Dict[str, pd.DataFrame] = {}

    # 1. Inject scale factor -- replace existing assignment or prepend
    code = SCALE_FACTOR_PATTERN.sub(
        f'SCALE_FACTOR = {scale_factor}', generated_code
    )
    if f'SCALE_FACTOR = {scale_factor}' not in code:
        code = f'SCALE_FACTOR = {scale_factor}\n' + code

    with tempfile.TemporaryDirectory(prefix='scribbledb_smoke_') as tmpdir:
        script_path = Path(tmpdir) / 'synthesis.py'

        # 2. Redirect all CSV writes to tmpdir by injecting os.chdir at top.
        #    The generated code uses bare filenames:
        #      for name, df in BUFFERS.items(): df.to_csv(f'{name}.csv', ...)
        #    so a chdir before execution is sufficient.
        patched_code = f'import os\nos.chdir({repr(tmpdir)})\n' + code
        script_path.write_text(patched_code, encoding='utf-8')

        logs.append(f'[SmokeTest] Script written to {script_path}')
        logs.append(f'[SmokeTest] Scale factor: {scale_factor}')

        # 3. Launch subprocess
        start = time.time()
        killed = False
        mem_exceeded = False
        proc = None

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                text=True,
                encoding='utf-8',
            )

            # 4. Monitor memory in a daemon thread -- kill if RSS > 1 GB
            def _monitor_memory() -> None:
                nonlocal killed, mem_exceeded
                try:
                    p = psutil.Process(proc.pid)
                    while proc.poll() is None:
                        try:
                            mem = p.memory_info().rss
                            if mem > MEMORY_LIMIT_BYTES:
                                mem_exceeded = True
                                proc.kill()
                                killed = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            break
                        time.sleep(0.5)
                except Exception:
                    pass

            monitor_thread = threading.Thread(target=_monitor_memory, daemon=True)
            monitor_thread.start()

            # 5. Wait for completion with hard timeout
            try:
                stdout, stderr = proc.communicate(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                killed = True
                logs.append(
                    '[SmokeTest] TIMEOUT: Process killed after 120 seconds'
                )

            monitor_thread.join(timeout=2.0)
            elapsed = time.time() - start

            if stdout:
                logs.append(f'[SmokeTest] STDOUT:\n{stdout[:2000]}')
            if stderr:
                logs.append(f'[SmokeTest] STDERR:\n{stderr[:2000]}')

            if mem_exceeded:
                logs.append(
                    '[SmokeTest] MEMORY EXCEEDED: Process killed (> 1 GB)'
                )

            success = proc.returncode == 0 and not killed

            if success:
                logs.append(
                    f'[SmokeTest] Execution SUCCESS in {elapsed:.1f}s'
                )
            else:
                logs.append(
                    f'[SmokeTest] Execution FAILED'
                    f' (returncode={proc.returncode}) in {elapsed:.1f}s'
                )

            # 6. Collect CSVs written by the synthesis script
            csv_files = list(Path(tmpdir).glob('*.csv'))
            for csv_path in csv_files:
                try:
                    table_name = csv_path.stem
                    df = pd.read_csv(csv_path)
                    dataframes[table_name] = df
                    logs.append(
                        f'[SmokeTest] Loaded {table_name}:'
                        f' {len(df)} rows x {len(df.columns)} cols'
                    )
                except Exception as e:
                    logs.append(
                        f'[SmokeTest] Failed to load {csv_path.name}: {e}'
                    )

            # 7. CSVs are deleted automatically when the tmpdir context exits.

        except Exception as e:
            success = False
            logs.append(f'[SmokeTest] Unexpected error: {e}')

    return success, dataframes, logs
