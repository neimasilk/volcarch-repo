# Post-process the pandoc conversion of the SUBMITTED text so that citation wording matches what the
# revision uses ("and", not "&"; serial comma for three authors); content is not touched.
# Output = baseline.docx (no author, no personal properties).
import docx, re, sys
src, out = sys.argv[1], sys.argv[2]
d = docx.Document(src)
n = 0
SERIAL = re.compile(r"([A-Z][A-Za-z'’-]+, [A-Z][A-Za-z'’-]+) and ([A-Z])")

def fix_par(p):
    global n
    for r in p.runs:
        if " & " in r.text:
            r.text = r.text.replace(" & ", " and ")
            n += 1
        new = SERIAL.sub(lambda m: m.group(1) + ", and " + m.group(2), r.text)
        if new != r.text:
            r.text = new
            n += 1

for p in d.paragraphs:
    fix_par(p)
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                fix_par(p)
cp = d.core_properties
cp.author = ""
cp.last_modified_by = ""
cp.title = "Submitted version (converted for comparison)"
cp.comments = ""
d.save(out)
print("runs changed:", n)
