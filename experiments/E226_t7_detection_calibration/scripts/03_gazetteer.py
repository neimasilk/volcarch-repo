"""E226 step 2a: modern desa/kelurahan list for the frame and buffer kabupaten (I2 uniqueness test).

Source: Kepmendagri 300.2.2-2430/2025 via cahyadsn/wilayah (data/sources.md). Names only, no coordinates;
hamlet (dusun) names are NOT in this list, which makes I2 conservative.
"""
import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "data", "raw", "gazetteer", "kemendagri_wilayah_2025_cahyadsn.sql")
OUT = os.path.join(ROOT, "data", "processed", "gazetteer", "desa_kedu_mataram_buffer_2025.csv")
FRAME = {"Kabupaten Magelang", "Kota Magelang", "Kabupaten Temanggung", "Kabupaten Wonosobo",
         "Kabupaten Purworejo", "Kabupaten Sleman", "Kabupaten Bantul", "Kota Yogyakarta",
         "Kabupaten Klaten", "Kabupaten Gunungkidul", "Kabupaten Gunung Kidul", "Kabupaten Kulon Progo"}
BUFFER = {"Kabupaten Boyolali", "Kabupaten Kendal", "Kabupaten Semarang", "Kabupaten Banjarnegara",
          "Kabupaten Kebumen"}

rows = re.findall(r"\('(\d{2}(?:\.\d{2}){0,2}(?:\.\d{4})?)','([^']*(?:''[^']*)*)'\)",
                  open(SRC, encoding="utf-8").read())
name = {c: n.replace("''", "'") for c, n in rows}
out = []
for c, n in name.items():
    if c.count(".") != 3 or c[:2] not in ("33", "34"):
        continue
    kab = name.get(c[:5], "")
    if kab in FRAME or kab in BUFFER:
        out.append(dict(kode=c, desa=n, kecamatan=name.get(c[:8], ""), kabupaten=kab,
                        zone="frame" if kab in FRAME else "buffer", kel_or_desa="kelurahan" if c[9] == "1" else "desa"))
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
from collections import Counter
print(len(out), Counter(r["kabupaten"] for r in out))
