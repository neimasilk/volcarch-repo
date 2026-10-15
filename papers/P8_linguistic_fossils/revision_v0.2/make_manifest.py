"""Print the upload manifest of the P8 resubmission (file, portal slot, size, SHA-256) as a Markdown table.

usage: python -X utf8 make_manifest.py [--check]
  --check   also run the anonymity check on the three files that can reach a reviewer, and confirm that the seven files
            not replaced on 2026-10-15 are byte for byte the files in the portal since 2026-10-12

Run it again whenever a file is rebuilt: G12 (download every uploaded file back and compare) uses these hashes.
State of 2026-10-15: article v0.4, five figures and tracked copy as uploaded on 2026-10-12; both letters v0.6 (the editor's
reply of 13 October). --check confirms that the seven files not replaced on 2026-10-15 are byte for byte the files in the
portal since 2026-10-12 (PORTAL_STATE_20261012.md, section 2).
"""
import hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parents[2] / "experiments" / "E229_p8_revision_tables" / "results"
ROWS = [
    ("Article File", HERE / "P8_revision_v0.4_anonymous.docx", "in the portal since 12 Oct"),
    ("Figure 1", FIG / "F1_input_importance.tif", "in the portal since 12 Oct"),
    ("Figure 2", FIG / "F2_workflow.tif", "in the portal since 12 Oct"),
    ("Figure 3", FIG / "F3_pmp_classes.tif", "unchanged since 8 Oct"),
    ("Figure 4", FIG / "F4_heldout_auc.tif", "unchanged since 8 Oct"),
    ("Figure 5", FIG / "F5_glottal_by_list.tif", "unchanged since 8 Oct"),
    ("Author Cover Letter", HERE / "COVER_LETTER_v0.6.docx", "replaces COVER_LETTER_v0.5.docx (editor's reply of 13 Oct)"),
    ("Revision Summary", HERE / "RESPONSE_TO_REVIEWERS_v0.6.docx", "replaces RESPONSE_TO_REVIEWERS_v0.5.docx (note for the reviewers)"),
    ("Supplemental Material", HERE / "P8_revision_v0.4_tracked_changes.docx", "in the portal since 12 Oct"),
]
# as hashed in the browser from the portal draft on 2026-10-12 after the upload (G12, PORTAL_STATE_20261012.md section 2)
IN_PORTAL_20261012 = {
    "P8_revision_v0.4_anonymous.docx": "6edc2a081827cc06620afff58d2b80ca094aaa4e79b145e4f531d55970ce5d71",
    "F1_input_importance.tif": "cc04efdfeb8690dc7974dc215131f37cfe1350bcf83030f1d3149b6fa535a8aa",
    "F2_workflow.tif": "7a9af08488f6ba7b55cd3fe5ca28bff3fcc4f6767fd7ebb7bb851674a1d01012",
    "F3_pmp_classes.tif": "39ea2cf7af3b8f58d0cb8e9da94c3a9896099284bf5cdd99ab20d1eafb249a84",
    "F4_heldout_auc.tif": "c1e4cf044315696c46077bb34a816fb367e9b8afd7ef5505089e49ebe76f52d3",
    "F5_glottal_by_list.tif": "3b78cb6d079fea44e855b4106ba078cc1d42930768a11992f99cc2c6efc75a36",
    "P8_revision_v0.4_tracked_changes.docx": "b90b010d2c1ff011c6fa872fbe2f584c9f0cfef857922ccd6043bda7fd3428ba",
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
print("| Portal slot | File | Bytes | SHA-256 | Action |")
print("|---|---|---|---|---|")
for slot, path, action in ROWS:
    print(f"| {slot} | `{path.name}` | {path.stat().st_size:,} | `{sha(path)}` | {action} |")

if "--check" in sys.argv:
    bad = 0
    for name, want in IN_PORTAL_20261012.items():
        ok = sha((HERE if name.endswith(".docx") else FIG) / name) == want
        bad += not ok
        print(f"{name}: identical to the file in the portal since 2026-10-12: {ok}")
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "tracked_build" / "check_anonymity.py"),
                        str(HERE / "P8_revision_v0.4_anonymous.docx"), str(HERE / "P8_revision_v0.4_tracked_changes.docx"),
                        str(HERE / "RESPONSE_TO_REVIEWERS_v0.6.docx")], capture_output=True)
    print(r.stdout.decode("utf-8", errors="replace"))
    sys.exit(1 if (bad or r.returncode) else 0)
