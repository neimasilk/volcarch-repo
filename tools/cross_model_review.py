"""
Cross-Model Critical Review — DeepSeek API caller

Purpose: Break the Claude-Claude echo chamber by running critical peer review
prompts through a non-Claude model. Addresses Mata Elang #15 §6B.

Usage:
    # Set API key first:
    # export DEEPSEEK_API_KEY=sk-...   (or put in .env)

    # Review P1-core:
    python tools/cross_model_review.py \\
        --paper papers/P1_taphonomic_framework/submission_jasrep_v3.0.tex \\
        --target P1 \\
        --out papers/P1_taphonomic_framework/external_reviews/critical_deepseek_$(date +%Y%m%d).md

    # Review P0:
    python tools/cross_model_review.py \\
        --paper papers/P0_invisible_civilization/draft_v0.1.tex \\
        --target P0 \\
        --out papers/P0_invisible_civilization/external_reviews/critical_deepseek_$(date +%Y%m%d).md

    # Use R1 (reasoning model, ~2x cost, deeper critique):
    python tools/cross_model_review.py --paper ... --model deepseek-reasoner

Budget: ~$0.50-$2 per review with deepseek-chat; ~$2-5 with deepseek-reasoner.

Output: Markdown file with review + metadata header.

Reads API key from: DEEPSEEK_API_KEY env var, or .env file in repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


CRITICAL_PROMPT_FILE = Path(__file__).parent / "critical_reviewer_prompt.md"
REPO_ROOT = Path(__file__).parent.parent
ENV_FILE = REPO_ROOT / ".env"


def load_env() -> None:
    """Minimal .env loader — no external dependency."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_paper(path: Path) -> str:
    """Read paper source. Supports .tex, .md, .txt."""
    if not path.exists():
        sys.exit(f"Paper not found: {path}")
    if path.suffix.lower() in {".tex", ".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    sys.exit(f"Unsupported paper format: {path.suffix}. Convert to .tex/.md/.txt first.")


def extract_prompt(prompt_file: Path, target: str) -> str:
    """Extract the core prompt + target-specific addendum from the prompt file."""
    text = prompt_file.read_text(encoding="utf-8")

    # Extract main prompt block between first ``` and second ```
    main_match = re.search(r"## The Prompt\s*```\s*(.*?)```", text, re.DOTALL)
    if not main_match:
        sys.exit("Could not parse main prompt from critical_reviewer_prompt.md")
    main_prompt = main_match.group(1).strip()

    # Extract target-specific addendum
    if target.upper() == "P1":
        addendum_match = re.search(
            r"## Specific Instruction Addenda for P1-core\s*```\s*(.*?)```",
            text, re.DOTALL,
        )
    elif target.upper() == "P0":
        addendum_match = re.search(
            r"## Specific Instruction Addenda for P0\s*```\s*(.*?)```",
            text, re.DOTALL,
        )
    else:
        addendum_match = None

    addendum = addendum_match.group(1).strip() if addendum_match else ""
    return f"{main_prompt}\n\n{addendum}" if addendum else main_prompt


def call_deepseek(api_key: str, model: str, system_prompt: str, user_content: str,
                  max_tokens: int = 8000) -> dict:
    """Call DeepSeek chat completions API via `requests`, streaming mode."""
    import requests

    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    content_parts = []
    usage = {}
    finish_reason = None

    with requests.post(url, json=payload, headers=headers, timeout=600,
                       stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]

    elapsed = time.time() - t0
    content = "".join(content_parts)
    if not content:
        raise RuntimeError("Empty response from stream.")

    return {
        "choices": [{"message": {"content": content},
                     "finish_reason": finish_reason}],
        "usage": usage,
        "_elapsed_seconds": round(elapsed, 1),
    }


def call_gemini(api_key: str, model: str, system_prompt: str, user_content: str,
                max_tokens: int = 8000) -> dict:
    """Call Google Gemini API via `requests`, streaming SSE mode.

    Gemini API format differs from OpenAI/DeepSeek:
    - URL contains model + `:streamGenerateContent` method
    - API key in header `x-goog-api-key`
    - Body uses `contents` / `systemInstruction` rather than `messages`
    - Response shape: `candidates[0].content.parts[0].text`
    """
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:streamGenerateContent?alt=sse"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    t0 = time.time()
    content_parts = []
    usage = {}
    finish_reason = None

    with requests.post(url, json=payload, headers=headers, timeout=600,
                       stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidates = chunk.get("candidates") or []
            if candidates:
                cand = candidates[0]
                parts = (cand.get("content") or {}).get("parts") or []
                for part in parts:
                    text = part.get("text")
                    if text:
                        content_parts.append(text)
                if cand.get("finishReason"):
                    finish_reason = cand["finishReason"]
            if chunk.get("usageMetadata"):
                um = chunk["usageMetadata"]
                usage = {
                    "prompt_tokens": um.get("promptTokenCount", 0),
                    "completion_tokens": um.get("candidatesTokenCount", 0),
                    "total_tokens": um.get("totalTokenCount", 0),
                }

    elapsed = time.time() - t0
    content = "".join(content_parts)
    if not content:
        raise RuntimeError("Empty response from Gemini stream.")

    return {
        "choices": [{"message": {"content": content},
                     "finish_reason": finish_reason}],
        "usage": usage,
        "_elapsed_seconds": round(elapsed, 1),
    }


def format_output(response: dict, paper_path: Path, prompt: str,
                  model: str, target: str) -> str:
    """Produce markdown output with header + review body."""
    choice = response["choices"][0]
    review_text = choice["message"]["content"]
    usage = response.get("usage", {})
    elapsed = response.get("_elapsed_seconds", "?")

    header = [
        f"# Critical Cross-Model Review — {target} — {model}",
        "",
        f"**Paper:** `{paper_path}`",
        f"**Model:** `{model}`",
        f"**Target addendum:** `{target}`",
        f"**Prompt tokens:** {usage.get('prompt_tokens', '?')}",
        f"**Completion tokens:** {usage.get('completion_tokens', '?')}",
        f"**Elapsed seconds:** {elapsed}",
        "",
        "## Review",
        "",
    ]
    return "\n".join(header) + review_text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model critical review via DeepSeek API."
    )
    parser.add_argument("--paper", type=Path, required=True,
                        help="Path to paper source (.tex/.md/.txt).")
    parser.add_argument("--target", choices=["P0", "P1", "generic"], default="generic",
                        help="Target-specific prompt addendum.")
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "gemini"],
                        help="API provider. 'deepseek' or 'gemini'.")
    parser.add_argument("--model", default=None,
                        help=("Model name. Defaults: deepseek-chat (deepseek) / "
                              "gemini-2.5-pro (gemini)."))
    parser.add_argument("--out", type=Path, required=True,
                        help="Output markdown file path.")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max response tokens.")
    parser.add_argument("--prompt-file", type=Path, default=CRITICAL_PROMPT_FILE,
                        help="Alternative critical reviewer prompt file.")
    args = parser.parse_args()

    load_env()

    # Select provider + default model + resolve API key
    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")
        if not api_key:
            sys.exit(
                "DEEPSEEK_API_KEY (or DEEPSEEK_API) not found in .env.\n"
                "Obtain key at https://platform.deepseek.com/api_keys"
            )
        model = args.model or "deepseek-chat"
        caller = call_deepseek
    elif args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit(
                "GEMINI_API_KEY not found in .env.\n"
                "Obtain free key at https://aistudio.google.com/apikey"
            )
        model = args.model or "gemini-2.5-pro"
        caller = call_gemini
    else:
        sys.exit(f"Unknown provider: {args.provider}")

    prompt = extract_prompt(args.prompt_file, args.target)
    paper_text = read_paper(args.paper)

    # System prompt = role. User content = paper.
    system_prompt = prompt
    user_content = (
        f"The manuscript follows. It is LaTeX source; treat the LaTeX markup as "
        f"peripheral (tables and equations matter; formatting commands do not).\n\n"
        f"---\n\n{paper_text}"
    )

    print(f"Calling {args.provider} ({model})...", file=sys.stderr)
    try:
        response = caller(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        sys.exit(f"API call failed: {e}")

    output = format_output(response, args.paper, prompt, model, args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")

    usage = response.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    # DeepSeek-chat: ~$0.14/M input, $0.28/M output (approx; check current)
    # Rough ballpark for budgeting display:
    est_cost_usd = round(total_tokens * 0.20 / 1_000_000, 4)
    print(f"Done. Output: {args.out}", file=sys.stderr)
    print(f"Tokens: {total_tokens:,}  (rough estimate ${est_cost_usd})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
