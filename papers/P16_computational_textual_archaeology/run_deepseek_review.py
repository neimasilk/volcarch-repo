"""G9 cross-model critical review of P16 (Wacana submission) via DeepSeek."""
import os
from pathlib import Path
from openai import OpenAI

key = os.environ.get("DEEPSEEK_API")
if not key:
    envp = Path(__file__).resolve().parents[2] / ".env"
    for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("DEEPSEEK_API") and "=" in line:
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
assert key, "DEEPSEEK_API not found in env or .env"
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

tex = Path("submission_wacana_v1.0.tex").read_text(encoding="utf-8")
body = tex[tex.find("\\title"):]  # drop preamble to save tokens

prompt = f"""You are a rigorous, skeptical peer reviewer for *Wacana, Journal of the Humanities of Indonesia* (Scopus Q2; Indonesian studies — linguistics, philology, archaeology). Review the manuscript below CRITICALLY and decide whether it should be accepted.

Identify the most serious weaknesses. For EACH weakness give: (a) the critique, (b) severity [FATAL / MAJOR / MINOR], (c) whether it is fixable by revision or is structural/unfixable. Focus especially on:
- methodological validity (Sentence-BERT trained on modern English applied to translated ancient texts; Monte Carlo convergence with researcher-chosen concept-group tags; n=46 dated inscriptions);
- over-claiming vs. what the evidence supports;
- circularity or assuming the conclusion;
- reproducibility;
- equifinality (are there simpler explanations for "volcanic silence" and the 929 CE shift?);
- scope fit for a humanities/Indonesian-studies journal (is the contribution too purely computational?);
- whether the stated conclusions actually follow from the results.

Be harsh and specific; quote the exact claims you critique. Then end with:
RECOMMENDATION: [Accept / Minor revision / Major revision / Reject]
SINGLE MOST IMPORTANT FIX: ...

MANUSCRIPT:
{body}
"""

def run(model):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=8000, stream=False,
    )
    return r.choices[0].message.content, r.usage

try:
    out, usage = run("deepseek-reasoner")
    model_used = "deepseek-reasoner"
except Exception as e:
    print("reasoner failed:", e, "-> falling back to deepseek-chat")
    out, usage = run("deepseek-chat")
    model_used = "deepseek-chat"

Path("external_reviews").mkdir(exist_ok=True)
Path("external_reviews/critical_deepseek_p16_wacana_R1_20260610.md").write_text(
    f"# DeepSeek critical review — P16 (Wacana submission, R1-revised draft) — 2026-06-10\nModel: {model_used}\n\n{out}\n",
    encoding="utf-8")
print(f"=== MODEL: {model_used} | tokens: {usage.total_tokens} | output chars: {len(out)} ===")
print("(full review saved to external_reviews/critical_deepseek_p16_wacana_20260608.md)")
