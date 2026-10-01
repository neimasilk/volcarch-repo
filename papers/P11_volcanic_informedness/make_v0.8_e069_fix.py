"""P11 v0.7 -> v0.8: remove the E069 survey-control claim (sign was misread), 2026-10-01.

The E069 coefficient (beta = -0.831) is on DISTANCE to the nearest volcano, so it means MORE
recorded sites near volcanoes after survey control, not a deficit. v0.7 said "The deficit is
real, not a survey artefact." The regression is also run on a 666-site OSM/Wikipedia inventory
with almost no period information that may include modern monuments, so it is dropped from the
paper altogether rather than re-cited with the corrected sign; the survey-effort objection is
stated as open. This script rewrites exactly the dependent passages, in both the LaTeX source and
the SPAFA Word file (removing footnote 56), and leaves v0.7 untouched. Every replacement asserts
an exact match so a silent no-op is impossible.
See experiments/E069_adversarial_comparanda/adv3_survey_intensity/README.md (CORRECTION) and
docs/CRITIQUE_LEDGER.md C023.
"""
import shutil
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
TEX_IN, TEX_OUT = HERE / "draft_v0.7_spafa.tex", HERE / "draft_v0.8_spafa.tex"
DOCX_IN = HERE / "spafa_assets" / "P11_submission_v0.7.docx"
DOCX_OUT = HERE / "spafa_assets" / "P11_submission_v0.8.docx"

# ---- new wording (shared by both formats) ------------------------------------------------
S_OBJECTION = ("A potential objection is that the scarcity of settlement sites near volcanoes reflects "
               "differential survey effort rather than burial.")
S_OPEN = ("Our data cannot yet settle this: settlements make up barely one per cent of the recorded "
          "inventory (see below), too few for a survey-controlled test, and separating burial from survey "
          "effort will require excavation-effort records that have not yet been compiled.")
S_LIMIT = ("Survey effort remains an unresolved alternative explanation for the scarcity of settlement "
           "sites (see the Discussion).")
S_SILENCE = ("The silence in the record is not, by itself, evidence of absence; at Liangan, Sambisari and "
             "Kimpulan it is demonstrably the work of burial.")


def tex_fix():
    t = TEX_IN.read_text(encoding="utf-8")
    reps = [
        (", the Liangan discovery, and the survey-intensity control--points toward a single conclusion",
         ", and the Liangan discovery--points toward a single conclusion"),
        ("A potential objection is that the archaeological gap near volcanoes reflects differential survey "
         "effort rather than genuine burial.\n"
         "We tested this using robustness regression controlling for three survey-intensity proxies (road "
         "distance, proximity to heritage offices, university proximity).\n"
         "After controlling for all three, volcanic proximity still explains additional variance in site "
         "counts.\\footnote{Quasi-Poisson GLM on the canonical 30-volcano Java inventory: volcanic proximity "
         "$\\beta = -0.831$, likelihood-ratio $p = 2.9 \\times 10^{-7}$ (dispersion-adjusted quasi-likelihood). "
         "The signal strengthens relative to the earlier 7-volcano run ($\\beta = -0.477$, $p = 0.0015$).}\n"
         "The deficit is real, not a survey artefact.",
         S_OBJECTION + "\n" + S_OPEN),
        ("All evidence remains indirect--no subsurface geophysical validation has yet been conducted.\n",
         "All evidence remains indirect--no subsurface geophysical validation has yet been conducted.\n"
         + S_LIMIT + "\n"),
        ("The silence in the record is not evidence of absence; it is evidence of burial.", S_SILENCE),
    ]
    for old, new in reps:
        n = t.count(old)
        assert n == 1, f"tex: expected 1 match, found {n}: {old[:70]!r}"
        t = t.replace(old, new)
    TEX_OUT.write_text(t, encoding="utf-8")
    print("wrote", TEX_OUT.name)


def _set_run_text(run_el, text):
    ts = run_el.findall(qn("w:t"))
    assert len(ts) == 1, "expected exactly one w:t in run"
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def _run_text(run_el):
    return "".join(t.text or "" for t in run_el.iter(qn("w:t")))


def docx_body_fix():
    shutil.copyfile(DOCX_IN, DOCX_OUT)
    doc = Document(DOCX_OUT)
    paras = doc.paragraphs

    # P60: drop the survey-control item from the evidence list
    p = paras[60]._p
    runs = p.findall(qn("w:r"))
    old = ", the Liangan discovery, and the survey-intensity control–points"
    txt = _run_text(runs[0])
    assert txt.count(old) == 1, "P60 phrase not found"
    _set_run_text(runs[0], txt.replace(old, ", and the Liangan discovery–points"))

    # P61: objection paragraph; the regression sentence and its footnote reference are removed
    p = paras[61]._p
    runs = p.findall(qn("w:r"))
    assert len(runs) == 8, f"P61: expected 8 runs, found {len(runs)}"
    assert _run_text(runs[0]).startswith("A potential objection is that the archaeological gap")
    fref = runs[5].find(qn("w:footnoteReference"))
    assert fref is not None and fref.get(qn("w:id")) == "56"
    assert _run_text(runs[7]) == "The deficit is real, not a survey artefact."
    _set_run_text(runs[0], S_OBJECTION)
    _set_run_text(runs[2], S_OPEN)
    for r in runs[3:]:
        p.remove(r)

    # P65: add the survey-effort limitation after the "indirect" sentence
    p = paras[65]._p
    runs = p.findall(qn("w:r"))
    assert _run_text(runs[2]) == ("All evidence remains indirect–no subsurface geophysical validation "
                                  "has yet been conducted.")
    _set_run_text(runs[2], _run_text(runs[2]) + " " + S_LIMIT)

    # P69: downgrade the closing claim
    p = paras[69]._p
    runs = p.findall(qn("w:r"))
    assert _run_text(runs[7]) == "The silence in the record is not evidence of absence; it is evidence of burial."
    _set_run_text(runs[7], S_SILENCE)
    doc.save(DOCX_OUT)
    print("body edits saved")


def docx_drop_footnote_56():
    tmp = DOCX_OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(DOCX_OUT) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/footnotes.xml":
                root = etree.fromstring(data)
                fn = [f for f in root.findall(qn("w:footnote")) if f.get(qn("w:id")) == "56"]
                assert len(fn) == 1, "footnote 56 not found"
                root.remove(fn[0])
                data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            if item.filename == "word/document.xml":
                assert b'w:footnoteReference w:id="56"' not in data, "reference to footnote 56 still present"
            zout.writestr(item, data)
    tmp.replace(DOCX_OUT)
    print("footnote 56 removed")


if __name__ == "__main__":
    tex_fix()
    docx_body_fix()
    docx_drop_footnote_56()
