# VOLCARCH AutoResearch — Konsep Integrasi

**Tanggal:** 2026-03-30
**Status:** 🅿 SUPERSEDED 2026-08-11 — belum pernah diimplementasikan (`tools/autoresearch/results/`
kosong); prioritasnya digantikan manifesto v5.0 §2; Program 3 (cascade) objeknya sudah pensiun.
Diaramkan sebagai arsip konsep; jangan dibangun pipeline controller (keputusan kritik sistem).
**Sebelumnya:** PROPOSAL — perlu keputusan Pak Amien
**Inspirasi:** [Karpathy autoresearch](https://github.com/karpathy/autoresearch) (Maret 2026)

---

## Prinsip Kunci dari Karpathy

> "You're not touching Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that set up your iterative research pipeline."

Pola Karpathy:
1. **Satu metrik** (val_bpb) — objektif, terukur, comparable
2. **Satu file yang dimodifikasi** (train.py) — constrains search space
3. **Budget waktu tetap** (5 menit) — membuat eksperimen comparable
4. **Keep/discard binary** berdasarkan metrik
5. **`program.md`** = "research org code" yang ditulis manusia
6. **Iterasi pipeline berjalan tanpa interaksi aktif peneliti**

Insight Pak Amien yang tepat: *"Pipeline yang diberi ruang eksplorasi luas menghasilkan hipotesis lebih beragam, asalkan manusia memberikan evaluasi dan tujuan yang jelas."*

Ini bisa diformulasi: **Kualitas output pipeline berbanding lurus dengan kejelasan evaluasi.**

---

## Mengapa VOLCARCH Cocok

VOLCARCH sudah punya semua komponen yang dibutuhkan:

| Karpathy | VOLCARCH | Sudah Ada? |
|----------|----------|:---:|
| `program.md` (tujuan + aturan) | `docs/drafts/manifesto.md` + `docs/EVAL.md` | **YA** |
| `train.py` (file yg dimodifikasi) | Python scripts per eksperimen | **YA** |
| `val_bpb` (metrik evaluasi) | Falsification criteria per layer | **YA** |
| `results.tsv` (logging) | Experiment README + JOURNAL.md | **YA** |
| Keep/discard logic | SUCCESS / FAILED / INCONCLUSIVE | **YA** |
| Fixed budget | Setiap eksperimen self-contained | **YA** |

Yang belum ada: **pipeline controller** dan **program.md khusus per research program**.

---

## Adaptasi: Dari Satu Metrik ke Research Programs

Karpathy mengoptimasi SATU metrik. Riset ilmiah punya BANYAK pertanyaan. Solusi: **pecah jadi research programs**, masing-masing dengan metrik sendiri.

### Research Program = Unit Pipeline

Setiap program punya:
- **Goal** (satu kalimat)
- **Metric** (bagaimana mengukur sukses/gagal)
- **Scope** (dataset + eksperimen yang boleh diakses)
- **Constraints** (apa yang TIDAK boleh dilakukan)
- **Time budget** per eksperimen
- **Keep/discard criteria**

---

## 5 Research Programs yang Bisa Dijalankan

### Program 1: "Robustness Battery" — PALING AMAN, MULAI DARI SINI

**Goal:** Untuk setiap klaim statistik yang survive FDR (30 dari 41, E068), jalankan battery robustness test.

**Metrik per eksperimen:** Apakah kesimpulan berubah? (binary: ROBUST / FRAGILE)

**Loop:**
```
FOR each of 30 FDR-surviving experiments:
  1. Read README.md → extract statistical claim
  2. Write robustness script:
     a. Bootstrap 1000× → confidence interval
     b. Jackknife leave-one-out → stability
     c. Permutation 10K shuffles → p-value
     d. Parameter sensitivity ±20% → threshold
  3. Run (budget: 5 min per test, 20 min per experiment)
  4. Evaluate: if ALL pass → ROBUST; if ANY fail → FRAGILE (flag)
  5. Write README_robustness.md in experiment dir
  6. Log to results.tsv
```

**Estimasi:** 30 eksperimen × 4 tests × 5 min = ~10 jam compute. Bisa jalan overnight.

**Nilai:** Sebelum submit paper, bisa bilang "semua klaim di-stress-test secara otomatis." Reviewer-proof.

**Risiko:** Rendah. Tidak mengubah klaim, hanya menguji.

---

### Program 2: "ColonialMine NLP Pipeline" — P21, NLP-HEAVY

**Goal:** Bangun NER pipeline untuk ekstrasi data taphonomic dari corpus kolonial Delpher.nl.

**Metrik:** F1 score pada test set manual (target: F1 > 0.70)

**Loop:**
```
WHILE F1 < target:
  1. Download batch dari Delpher API
  2. Preprocess (historical Dutch normalization)
  3. Run NER (BERT fine-tune atau rule-based)
  4. Evaluate pada manual annotations
  5. If F1 improved → keep; else → discard
  6. Try: different models, rules, features, thresholds
```

**Estimasi:** Pipeline setup 1 minggu, then iterative fine-tuning cycles. GPU: 2-4 jam per siklus.

**Nilai:** Dataset genuinely independent dari DHARMA/ABVD. Extends E083/E091. Langsung bisa jadi revision support material untuk P1.

---

### Program 3: "Cascade Stress Test" — CRITICAL

**Goal:** Temukan faktor terlemah dalam 5-factor cascade (E110).

**Metrik:** Magnitude of conclusion change when varying each factor.

**Loop:**
```
FOR each of 5 cascade factors:
  1. Vary parameter from 0.5× to 2.0× baseline (10 steps)
  2. Recompute cascade product
  3. Check: does visibility still match observed 0.031%?
  4. Find: at what threshold does cascade break?
  5. Identify: which factor, when removed, changes conclusion most?
```

**Estimasi:** 5 × 10 × 1 min = ~1 jam. Sangat cepat.

**Nilai:** Menjawab pertanyaan reviewer: "Which factor drives your model? How sensitive is it?" dengan data, bukan retorika.

---

### Program 4: "TobaSim" — P20, GPU-HEAVY, LONG-TERM

**Goal:** Rekonstruksi dispersal abu Toba 74ka dengan FALL3D. Minimize prediction error vs 57 deposit aktual.

**Metrik:** RMSE antara prediksi ketebalan abu vs pengukuran aktual (Costa et al. 2014 supplementary).

**Loop:**
```
1. Setup FALL3D + paleotopography (manual, 2 minggu)
2. Run baseline simulation → establish RMSE_baseline
WHILE RMSE can improve:
  3. Vary: wind field, source parameters, grain size
  4. Run simulation (2-8 jam per run, RTX 4080)
  5. Compare vs known deposits
  6. If RMSE improved → keep; else → discard
  7. After convergence: extract Java + Sulawesi differential
```

**Estimasi:** 2 minggu setup + 1-2 minggu compute (100 simulations).

**Nilai:** Extends VOLCARCH ke timescale geologis. Nature-level paper potential. Tapi needs geologist co-author.

---

### Program 5: "Anomaly Refinement" — EXTENDS E097

**Goal:** Improve anomaly detection overlap dengan E080 fieldwork candidates dari 65% → >80%.

**Metrik:** Overlap percentage dengan E080 targets.

**Loop:**
```
WHILE overlap < 80%:
  1. Modify: features (TRI, slope, distance, burial depth)
  2. Modify: algorithm (Isolation Forest, LOF, DBSCAN, AutoEncoder)
  3. Run anomaly detection on E. Java grid
  4. Compute overlap with E080 top-20
  5. If improved → keep; else → discard
```

**Estimasi:** 5-10 min per iteration, dozens of iterations.

**Nilai:** Stronger fieldwork targeting = better use of expensive GPR time.

---

## Implementasi Teknis

### Struktur Direktori

```
tools/autoresearch/
├── program_robustness.md       ← Research program 1
├── program_colonialmine.md     ← Research program 2
├── program_cascade.md          ← Research program 3
├── runner.py                   ← Loop runner (generic)
├── evaluator.py                ← Result evaluator
├── results/
│   ├── robustness_results.tsv
│   ├── colonialmine_results.tsv
│   └── ...
└── README.md                   ← How to run
```

### Loop Runner (Konsep)

```python
"""
VOLCARCH AutoResearch Runner
Inspired by Karpathy's autoresearch

Usage: Tell Claude Code to "run program_robustness.md"
Pipeline reads program file, runs training cycle, logs results.
"""

# Human writes program.md → defines goal, metric, scope
# Agent reads program.md → designs experiments → runs → evaluates
# Results logged to TSV → human reviews in morning

# Key difference from Karpathy:
# - Karpathy: one metric, one file, iterative refinement cycle
# - VOLCARCH: one metric PER PROGRAM, multiple scripts,
#   iterative cycle with human checkpoints
```

### Perbedaan Kunci vs Karpathy

| Aspek | Karpathy | VOLCARCH |
|-------|----------|----------|
| Metrik | 1 (val_bpb) | 1 per program |
| File yg diubah | 1 (train.py) | 1 script per eksperimen |
| Domain | ML training | Multi-domain science |
| Evaluasi | Otomatis | Otomatis untuk robustness, human-in-loop untuk novelty |
| Output | Better model | Paper-ready evidence + new findings |
| Risk | Low (just training) | Medium (wrong conclusions possible) |

### Safety Rails

1. **Tidak boleh mengubah data raw.** `data/raw/` = read-only.
2. **Tidak boleh menghapus eksperimen.** FAILED = documented, not deleted.
3. **Contradict manifesto = IMMEDIATE FLAG.** Agent harus berhenti dan lapor jika temuan menolak hipotesis.
4. **Human checkpoint setiap N eksperimen** (N = 10 default, adjustable).
5. **Semua kode di-commit per eksperimen.** Reproducibility > speed.
6. **Budget timeout per eksperimen.** Prevent runaway compute.

---

## Roadmap Implementasi

### Phase 1: Proof of Concept (Minggu ini — 1 sesi)
- Jalankan **Program 3 (Cascade Stress Test)** secara manual dalam 1 sesi Claude Code
- Budget: ~1 jam
- Outcome: Apakah pola autoresearch bekerja untuk VOLCARCH?

### Phase 2: Robustness Battery (Minggu depan — overnight run)
- Setup Program 1 sebagai script yang bisa dijalankan otomatis
- Target: 30 eksperimen × 4 robustness tests = 120 checks
- Claude Code jalan overnight, Pak Amien review pagi

### Phase 3: ColonialMine Sprint (April 2026)
- Setup Delpher API access + Dutch NLP pipeline
- Iterative NER pipeline
- Target: >100 extracted colonial finds → revision support material untuk P1 di EGQSJ

### Phase 4: Full Integration (Mei 2026+)
- Multiple research programs running concurrently across 4 GPU
- Programs feed into each other (ColonialMine → Cascade → Paper revision)
- Pipeline proposes new programs berdasarkan results

---

## Hubungan dengan 3 Proposal Mudik

| Proposal | Program | Kapan Mulai | Blocker |
|----------|---------|-------------|---------|
| P20 TobaSim | Program 4 | Setelah Phase 2 berhasil | FALL3D setup + geologist |
| P21 ColonialMine | Program 2 | Phase 3 (April) | Delpher API testing |
| P22 JavaTephroChron | Depends on P20 | Setelah P20 FALL3D ready | P20 infrastructure |

P21 (ColonialMine) paling cocok dimulai duluan karena:
1. Skills match (NLP, sudah ada pengalaman E091)
2. Data free (Delpher.nl public domain)
3. Tidak butuh co-author untuk memulai
4. Hasilnya langsung jadi revision support material

---

## Pertanyaan untuk Pak Amien

1. **Mau mulai dari mana?** Rekomendasi: Program 3 (Cascade, 1 jam) sebagai proof of concept.
2. **Overnight runs OK?** Claude Code jalan semalaman di kampus = ~100 micro-experiments.
3. **4× RTX 4080 — mana yang available?** Program 4 (TobaSim) butuh dedicated GPU.
4. **Risk appetite:** Apakah temuan otomatis yang MENOLAK hipotesis harus langsung di-flag, atau boleh dilanjut pipeline?
5. **autoresearch/ folder di inBox** — ini project terpisah dari Karpathy. Delete dari inBox setelah kita paham konsepnya? Atau simpan sebagai reference di `tools/`?

---

*"Pertanyaannya bukan: apakah pipeline bisa berjalan otomatis. Pertanyaannya: apakah kita bisa memberikan evaluasi yang cukup jelas agar hasilnya produktif, bukan menyesatkan."*
