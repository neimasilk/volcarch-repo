# VOLCARCH AutoResearch Framework v0.1

Autonomous experiment execution framework inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How It Works

1. **Human writes `program.md`** — defines goal, metric, scope, constraints
2. **Claude Code reads program** — designs and executes experiments
3. **Runner logs results** — TSV file with timestamps and verdicts
4. **Human reviews in morning** — check results, adjust program if needed

## Available Programs

| Program | Status | Goal | Estimated Time |
|---------|--------|------|----------------|
| `program_robustness.md` | READY | Stress-test 65 FDR-surviving findings | ~5 hours |
| `program_colonialmine.md` | PLANNED | NLP pipeline for Delpher colonial data | ~2 days |
| `program_cascade.md` | DONE (E120) | Cascade sensitivity analysis | 1 hour |

## Usage

```bash
# Dry run (parse program, don't execute)
python tools/autoresearch/runner.py program_robustness.md --dry-run

# Full run (Claude Code executes)
python tools/autoresearch/runner.py program_robustness.md
```

## Architecture

```
Human                    Claude Code              Runner
  |                         |                       |
  |-- writes program.md --> |                       |
  |                         |-- reads program -----> |
  |                         |<-- parsed structure ---|
  |                         |                       |
  |                         |-- executes exp 1 ---> |
  |                         |<-- result + log ------|
  |                         |-- executes exp 2 ---> |
  |                         |<-- result + log ------|
  |                         |      ...              |
  |                         |                       |
  |<-- reviews results -----|                       |
```

## Key Principles

1. **Kreativitas agent berbanding lurus dengan kejelasan evaluasi** — clearer metrics = better autonomous work
2. **Human writes the program, not the code** — "programming the research org"
3. **Keep/discard is binary** — every experiment has a clear verdict
4. **Safety rails** — never modify raw data, never delete experiments, flag contradictions
