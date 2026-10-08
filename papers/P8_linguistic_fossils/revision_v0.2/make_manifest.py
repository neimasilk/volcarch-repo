"""Print the upload manifest of the P8 resubmission (file, portal slot, size, SHA-256) as a Markdown table.

usage: python -X utf8 make_manifest.py [--check]
  --check   also run the anonymity check on the three files that can reach a reviewer and compare Figures 2 and 5
            with the hashes of the files uploaded on 2026-10-07 (they must be unchanged)

Run it again whenever a file is rebuilt: G12 (download every uploaded file back and compare) uses these hashes.
"""
import hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[2] / "experiments" / "E229_p8_revision_tables" / "results"
ROWS = [
    ("Article File", HERE / "P8_revision_v0.3_anonymous.docx", "replaces P8_revision_v0.2_anonymous.docx"),
    ("Figure 1", FIG / "F1_input_importance.tif", "replaces the file of 7 Oct (legend)"),
    ("Figure 2", FIG / "F2_workflow.tif", "unchanged — leave the portal's file"),
    ("Figure 3", FIG / "F3_pmp_classes.tif", "replaces the file of 7 Oct (labels)"),
    ("Figure 4", FIG / "F4_heldout_auc.tif", "replaces the file of 7 Oct (legend)"),
    ("Figure 5", FIG / "F5_glottal_by_list.tif", "unchanged — leave the portal's file"),
    ("Author Cover Letter", HERE / "COVER_LETTER_v0.4.docx", "replaces COVER_AND_RESPONSE_v0.2.docx"),
    ("Revision Summary", HERE / "RESPONSE_TO_REVIEWERS_v0.4.docx", "replaces COVER_AND_RESPONSE_v0.2.docx"),
    ("Supplemental Material", HERE / "P8_revision_v0.3_tracked_changes.docx", "new (ninth file)"),
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
print("| Portal slot | File | Bytes | SHA-256 | Action |")
print("|---|---|---|---|---|")
for slot, path, action in ROWS:
    print(f"| {slot} | `{path.name}` | {path.stat().st_size:,} | `{sha(path)}` | {action} |")

if "--check" in sys.argv:
    old = {}
    for line in (FIG / "as_uploaded_20261007" / "SHA256.txt").read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1); old[name.lstrip("*")] = h
    for name in ("F2_workflow.tif", "F5_glottal_by_list.tif"):
        print(f"{name} identical to the file uploaded on 2026-10-07: {sha(FIG / name) == old[name]}")
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "tracked_build" / "check_anonymity.py"),
                        str(HERE / "P8_revision_v0.3_anonymous.docx"), str(HERE / "P8_revision_v0.3_tracked_changes.docx"),
                        str(HERE / "RESPONSE_TO_REVIEWERS_v0.4.docx")], capture_output=True)
    print(r.stdout.decode("utf-8", errors="replace"))
    sys.exit(r.returncode)
