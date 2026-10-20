"""Correct references.bib from Crossref / publisher metadata (ledger C076; JCAA round 2, R1-6). 2026-10-20.

Every value below was checked by the orchestrator against a fresh Crossref call the same day
(see REFERENCE_FIX_20261020.md). Entries are replaced whole; nothing else in the file changes.
Run: python papers/P2_settlement_model/fix_references_20261020.py
"""
import re
from pathlib import Path

BIB = Path(__file__).resolve().parent / "references.bib"
t = BIB.read_text(encoding="utf-8")

NEW = {
    "castiello2021": r"""@article{castiello2021,
  author  = {Castiello, Maria Elena and Tonini, Marj},
  title   = {An explorative application of {R}andom {F}orest algorithm for archaeological predictive modeling. {A} {S}wiss case study},
  journal = {Journal of Computer Applications in Archaeology},
  year    = {2021},
  volume  = {4},
  number  = {1},
  pages   = {110--125},
  doi     = {10.5334/jcaa.71}
}""",
    "wang2023": r"""@article{wang2023,
  author  = {Wang, Yuan and Shi, Xiaodan and Oguchi, Takashi},
  title   = {Archaeological predictive modeling using machine learning and statistical methods for {J}apan and {C}hina},
  journal = {ISPRS International Journal of Geo-Information},
  year    = {2023},
  volume  = {12},
  number  = {6},
  pages   = {238},
  doi     = {10.3390/ijgi12060238}
}""",
    "comer2023": r"""@article{comer2023,
  author  = {Comer, Jacob A. and Comer, Douglas C. and Dumitru, Ioana A. and Priebe, Carey E. and Patsolic, Jesse L.},
  title   = {Sampling methods for archaeological predictive modeling: {S}patial autocorrelation and model performance},
  journal = {Journal of Archaeological Science: Reports},
  year    = {2023},
  volume  = {48},
  pages   = {103824},
  doi     = {10.1016/j.jasrep.2022.103824}
}""",
}


def entry_span(text, key):
    m = re.search(r"@\w+\{" + re.escape(key) + r",", text)
    assert m, key
    depth, i = 0, text.index("{", m.start())
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return m.start(), i + 1
        i += 1


log = []
for key, body in NEW.items():
    s, e = entry_span(t, key)
    t = t[:s] + body + t[e:]
    log.append(f"replaced {key}")

# Field-level fixes.
s, e = entry_span(t, "senay2013")
blk = t[s:e]
assert "Senay, Sabrina D." in blk
t = t[:s] + blk.replace("Senay, Sabrina D.", "Senay, Senait D.") + t[e:]
log.append("senay2013: Sabrina D. -> Senait D. (Crossref 10.1371/journal.pone.0071218)")

for key, doi in (("kvamme2006", "10.1201/9780203563359.sec1"), ("franklin2009", "10.1017/CBO9780511810602")):
    s, e = entry_span(t, key)
    blk = t[s:e]
    if "doi" not in blk:
        blk = blk[:-1].rstrip().rstrip(",") + f",\n  doi       = {{{doi}}}\n}}"
        t = t[:s] + blk + t[e:]
        log.append(f"{key}: doi {doi} added")

BIB.write_text(t, encoding="utf-8")
print("\n".join(log))
