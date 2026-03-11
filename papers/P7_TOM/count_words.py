"""Quick word count for Antiquity submission."""
import re
from pathlib import Path

tex = Path(__file__).parent / "submission_antiquity_v0.1.tex"
text = tex.read_text(encoding="utf-8")

body = text.split(r"\begin{document}")[1].split(r"\end{document}")[0]

# Remove comments
body = re.sub(r"%.*", "", body)

# Remove LaTeX environments/commands roughly
body = re.sub(r"\\begin\{[^}]*\}", "", body)
body = re.sub(r"\\end\{[^}]*\}", "", body)
body = re.sub(r"\\textbf\{([^}]*)\}", r"\1", body)
body = re.sub(r"\\textit\{([^}]*)\}", r"\1", body)
body = re.sub(r"\\textsuperscript\{[^}]*\}", "", body)
body = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", body)
body = re.sub(r"\\cite\w*\{[^}]*\}", "CITE", body)
body = re.sub(r"\\bibitem.*?\{[^}]*\}", "", body)
body = re.sub(r"\\[a-zA-Z]+", " ", body)
body = re.sub(r"[{}$~]", " ", body)

words = body.split()
print(f"Approximate word count: {len(words)}")

# Break down sections
sections = {
    "Title+Author": "Spatial segregation",
    "Body text": "The taphonomic problem",
    "Table": "Median distance",
    "Acknowledgements": "Acknowledgements",
    "References": "thebibliography",
    "Figure captions": "Figure captions",
}
print(f"\nTotal words in document body (incl. refs, captions, table): {len(words)}")
