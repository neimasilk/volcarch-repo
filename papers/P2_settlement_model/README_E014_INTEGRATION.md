# E014 Sudah Terintegrasi di Paper 2

**File PDF:** `submission_remote_sensing_FINAL.pdf` (16 halaman, 4.2 MB)

---

## Di Mana E014 Muncul di Paper?

### 1. **Abstract (Halaman 1)**
> "...temporal split validation confirming generalization to 'undiscovered' sites (**AUC = 0.755**)."

### 2. **Table 2 — Enhanced Tautology Test Suite (Halaman 8)**
Baris ke-4 tabel:
| T4: Temporal Split | **PASS** | **AUC = 0.755** (temporal) vs 0.785 (spatial) |

### 3. **Section 3.2 — Test 4 Description (Halaman 8)**
Paragraf lengkap menjelaskan:
- Metode: train on easy-access sites, test on hard-access sites
- Hasil: **AUC = 0.755**
- Interpretasi: "model generalizes to sites that are harder to discover"

### 4. **Verdict Update (Halaman 8)**
> "Overall, the enhanced tautology test suite returns a **PASS** verdict: the **temporal validation (T4)** and stratified CV (T3) provide robust evidence against tautology..."

### 5. **Code Availability Statement (Halaman 15)**
```
experiments/E014_temporal_validation/01_temporal_split_test.py 
— Temporal validation (E014)
```

---

## Perubahan Penting karena E014

| Sebelum E014 | Setelah E014 |
|--------------|--------------|
| Verdict: GREY_ZONE | Verdict: **PASS** |
| Claim: "tautology-mitigated" | Claim: **"tautology-resistant"** |
| 3 tests | **4 tests** (T1-T4) |

---

## File PDF Final

**Lokasi:** `papers/P2_settlement_model/submission_remote_sensing_FINAL.pdf`

**Cara cek:**
1. Buka PDF
2. Search (Ctrl+F): "0.755" atau "temporal" atau "E014"
3. Atau langsung ke halaman 8 (Table 2 dan Section 3.2)

---

## Status: ✅ SUDAH LENGKAP

E014 temporal validation sudah sepenuhnya terintegrasi dalam Paper 2 dan menjadi bukti utama untuk claim "tautology-resistant".
