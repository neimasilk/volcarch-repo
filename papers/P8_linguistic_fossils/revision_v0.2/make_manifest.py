"""Print the upload manifest of the P8 resubmission (file, portal slot, size, SHA-256) as a Markdown table.

usage: python -X utf8 make_manifest.py [--check]
  --check   also run the anonymity check on the three files that can reach a reviewer, and confirm that Figures 2 and 5
            are, below their white top band, pixel for pixel the files uploaded on 2026-10-07

Run it again whenever a file is rebuilt: G12 (download every uploaded file back and compare) uses these hashes.
The hashes of the files as they sat in the portal on 2026-10-08 are in PORTAL_STATE_20261008.md.
"""
import hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[2] / "experiments" / "E229_p8_revision_tables" / "results"
ROWS = [
    ("Article File", HERE / "P8_revision_v0.3_anonymous.docx", "replaces P8_revision_v0.2_anonymous.docx"),
    ("Figure 1", FIG / "F1_input_importance.tif", "replaces the file of 7 Oct (legend; top band)"),
    ("Figure 2", FIG / "F2_workflow.tif", "replaces the file of 7 Oct (top band only)"),
    ("Figure 3", FIG / "F3_pmp_classes.tif", "replaces the file of 7 Oct (labels; top band)"),
    ("Figure 4", FIG / "F4_heldout_auc.tif", "replaces the file of 7 Oct (legend; top band)"),
    ("Figure 5", FIG / "F5_glottal_by_list.tif", "replaces the file of 7 Oct (top band only)"),
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
    import numpy as np
    from PIL import Image
    for name in ("F2_workflow.tif", "F5_glottal_by_list.tif"):
        new = np.asarray(Image.open(FIG / name)); old = np.asarray(Image.open(FIG / "as_uploaded_20261007" / name))
        band = new.shape[0] - old.shape[0]
        ok = band > 0 and int(new[:band].min()) == 255 and np.array_equal(new[band:], old)
        print(f"{name}: below a white band of {band} px, identical to the file uploaded on 2026-10-07: {ok}")
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "tracked_build" / "check_anonymity.py"),
                        str(HERE / "P8_revision_v0.3_anonymous.docx"), str(HERE / "P8_revision_v0.3_tracked_changes.docx"),
                        str(HERE / "RESPONSE_TO_REVIEWERS_v0.4.docx")], capture_output=True)
    print(r.stdout.decode("utf-8", errors="replace"))
    sys.exit(r.returncode)
