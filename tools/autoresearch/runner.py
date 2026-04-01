"""
VOLCARCH AutoResearch Runner v0.1
==================================
Autonomous experiment execution framework.
Reads a program.md, executes experiments, evaluates results, logs outcomes.

Inspired by Karpathy's autoresearch but adapted for multi-domain science:
- One metric PER PROGRAM (not one global metric)
- Human checkpoint every N experiments
- Safety rails: never modify raw data, never delete experiments

Usage:
    python tools/autoresearch/runner.py program_robustness.md
    python tools/autoresearch/runner.py program_colonialmine.md

The runner is a SCAFFOLD — Claude Code provides the intelligence,
the runner provides the structure and logging.
"""

import argparse
import json
import csv
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("D:/documents/volcarch-repo")
RESULTS_DIR = REPO_ROOT / "tools" / "autoresearch" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_program(program_path):
    """Parse a program.md into structured instructions."""
    with open(program_path, 'r', encoding='utf-8') as f:
        content = f.read()

    program = {
        'name': '',
        'goal': '',
        'metric': '',
        'keep_threshold': '',
        'scope': '',
        'constraints': [],
        'time_budget_minutes': 5,
        'max_experiments': 10,
        'raw_content': content,
    }

    # Parse frontmatter-style fields
    for line in content.split('\n'):
        if line.startswith('**Goal:**'):
            program['goal'] = line.replace('**Goal:**', '').strip()
        elif line.startswith('**Metric:**'):
            program['metric'] = line.replace('**Metric:**', '').strip()
        elif line.startswith('**Keep if:**'):
            program['keep_threshold'] = line.replace('**Keep if:**', '').strip()
        elif line.startswith('**Scope:**'):
            program['scope'] = line.replace('**Scope:**', '').strip()
        elif line.startswith('**Time budget:**'):
            try:
                program['time_budget_minutes'] = int(re.search(r'\d+', line).group())
            except:
                pass
        elif line.startswith('**Max experiments:**'):
            try:
                program['max_experiments'] = int(re.search(r'\d+', line).group())
            except:
                pass
        elif line.startswith('# '):
            program['name'] = line.replace('# ', '').strip()

    return program


def log_result(program_name, experiment_id, metric_value, verdict, notes=""):
    """Append result to TSV log."""
    log_path = RESULTS_DIR / f"{program_name}_results.tsv"

    row = {
        'timestamp': datetime.now().isoformat(),
        'experiment': experiment_id,
        'metric_value': metric_value,
        'verdict': verdict,
        'notes': notes,
    }

    write_header = not log_path.exists()
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter='\t')
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return log_path


def run_experiment(script_path, timeout_seconds=300):
    """Run a Python experiment script and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', str(script_path)],
            capture_output=True, text=True, timeout=timeout_seconds,
            cwd=str(REPO_ROOT)
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout[-5000:] if result.stdout else '',  # last 5K chars
            'stderr': result.stderr[-2000:] if result.stderr else '',
            'success': result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'TIMEOUT',
            'success': False,
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False,
        }


def evaluate_result(output, metric_name, keep_threshold):
    """Extract metric value from experiment output and evaluate."""
    # Simple extraction: look for metric keywords in output
    lines = output.get('stdout', '').split('\n')

    metric_value = None
    for line in lines:
        if 'VERDICT' in line.upper() or 'RESULT' in line.upper():
            if 'ROBUST' in line.upper():
                metric_value = 1.0
            elif 'FRAGILE' in line.upper():
                metric_value = 0.0
            elif 'SUCCESS' in line.upper():
                metric_value = 1.0
            elif 'FAIL' in line.upper():
                metric_value = 0.0

    verdict = 'KEEP' if metric_value and metric_value > 0.5 else 'DISCARD'
    if not output['success']:
        verdict = 'ERROR'
        metric_value = None

    return metric_value, verdict


def main():
    parser = argparse.ArgumentParser(description='VOLCARCH AutoResearch Runner')
    parser.add_argument('program', help='Path to program.md file')
    parser.add_argument('--dry-run', action='store_true', help='Parse program only, do not execute')
    parser.add_argument('--max', type=int, default=None, help='Override max experiments')
    args = parser.parse_args()

    program_path = Path(args.program)
    if not program_path.exists():
        print(f"ERROR: Program file not found: {program_path}")
        sys.exit(1)

    program = load_program(program_path)

    print("=" * 70)
    print(f"VOLCARCH AutoResearch Runner v0.1")
    print(f"=" * 70)
    print(f"  Program: {program['name']}")
    print(f"  Goal: {program['goal']}")
    print(f"  Metric: {program['metric']}")
    print(f"  Keep threshold: {program['keep_threshold']}")
    print(f"  Time budget: {program['time_budget_minutes']} min/experiment")
    print(f"  Max experiments: {args.max or program['max_experiments']}")

    if args.dry_run:
        print(f"\n  DRY RUN — program parsed successfully. Not executing.")
        return

    print(f"\n  Ready to execute. Start experiments...")
    print(f"  (In autonomous mode, Claude Code reads this program and executes)")
    print(f"  (The runner provides structure; Claude provides intelligence)")


if __name__ == '__main__':
    main()
