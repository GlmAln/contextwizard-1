# backend/main.py
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional, Literal, Callable, TypeVar
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import json
import sys
import time
import random
import re

import anyio
from google import genai

types = genai.types  # alias for convenience
app = FastAPI()

# ----------------------------
# Retry config (tune here)
# ----------------------------
# You asked for 0.5s waits. In practice, starting a bit lower often clears transient 503s
# faster while still being gentle on the API.
RETRY_INITIAL_DELAY_SEC = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "0.35"))  # start slightly < 0.5
RETRY_MAX_DELAY_SEC = float(os.getenv("GEMINI_RETRY_MAX_DELAY", "2.0"))
RETRY_MAX_ATTEMPTS = int(os.getenv("GEMINI_RETRY_MAX_ATTEMPTS", "12"))
RETRY_JITTER_SEC = float(os.getenv("GEMINI_RETRY_JITTER_SEC", "0.10"))  # small jitter to avoid thundering herd


# ----------------------------
# Payload models
# ----------------------------
class FileInfo(BaseModel):
    filename: str
    status: Optional[str] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None
    changes: Optional[int] = None
    patch: Optional[str] = None  # unified diff string


class ReviewCommentInfo(BaseModel):
    id: int
    body: str
    path: Optional[str] = None
    diff_hunk: Optional[str] = None
    position: Optional[int] = None
    line: Optional[int] = None
    original_line: Optional[int] = None
    user_login: Optional[str] = None


class ReviewPayload(BaseModel):
    kind: str  # "review" or "review_comment"

    # review-level fields
    review_body: Optional[str] = None
    review_state: Optional[str] = None

    # inline-comment-level fields
    comment_body: Optional[str] = None
    comment_path: Optional[str] = None
    comment_diff_hunk: Optional[str] = None
    comment_position: Optional[int] = None
    comment_id: Optional[int] = None

    reviewer_login: Optional[str] = None
    pr_number: int
    pr_title: Optional[str] = None
    pr_body: Optional[str] = None
    pr_author_login: Optional[str] = None
    repo_full_name: str
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None

    files: Optional[List[FileInfo]] = None

    # all inline comments that belong to this finished review
    review_comments: Optional[List[ReviewCommentInfo]] = None


class BackendResponse(BaseModel):
    comment: str


# ----------------------------
# Gemini structured output models
# ----------------------------
Category = Literal[
    "PRAISE",
    "GOOD_CHANGE",
    "BAD_CHANGE",
    "GOOD_QUESTION",
    "BAD_QUESTION",
    "UNKNOWN",
]


class Classification(BaseModel):
    category: Category
    needs_reply: bool = Field(..., description="True only for GOOD_CHANGE, BAD_CHANGE, BAD_QUESTION.")
    needs_clarification: bool = Field(..., description="True only for BAD_CHANGE or BAD_QUESTION.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str = Field(..., description="One short sentence. No chain-of-thought.")


class ClarifiedQuestion(BaseModel):
    clarified_question: str = Field(..., description="A rewritten, clarified version of the original question.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str = Field(..., description="One short sentence on what was ambiguous / what you clarified.")


class ClarifiedChange(BaseModel):
    clarified_request: str = Field(
        ...,
        description="A rewritten, clarified change request. Must be actionable but may contain placeholders like <which function?>.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str = Field(..., description="One short sentence on what was unclear / what you clarified.")


# ----------------------------
# Helpers
# ----------------------------
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def clip(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "\n…(truncated)…"


def build_llm_context(payload: ReviewPayload) -> str:
    pr_title = payload.pr_title or ""
    pr_body = clip(payload.pr_body, 1200)

    base = f"""
Repo: {payload.repo_full_name}
PR: #{payload.pr_number} — {pr_title}
PR author: {payload.pr_author_login}

PR description (truncated):
{pr_body}
""".strip()

    if payload.kind == "review_comment":
        comment_text = payload.comment_body or ""
        path = payload.comment_path or ""
        hunk = clip(payload.comment_diff_hunk, 1200)

        base += f"""

Event: inline review comment
Reviewer: {payload.reviewer_login}
File path: {path}
Original comment:
{clip(comment_text, 1500)}

Diff hunk (truncated):
{hunk}
""".rstrip()
    else:
        review_text = payload.review_body or ""
        base += f"""

Event: review submitted
Reviewer: {payload.reviewer_login}
State: {payload.review_state}
Review body:
{clip(review_text, 2000)}
""".rstrip()

        if payload.review_comments:
            base += "\n\nInline comments in this review (showing up to 5):\n"
            for c in payload.review_comments[:5]:
                base += (
                    f"- id={c.id} file={c.path} line={c.line or c.position} "
                    f"by {c.user_login}: {clip(c.body, 400)}\n"
                )

    files = payload.files or []
    if files:
        base += f"\n\nChanged files: {len(files)} (showing up to 6 patches, truncated)\n"
        for f in files[:6]:
            base += (
                f"\n---\nFILE: {f.filename}\nSTATUS: {f.status} "
                f"(+{f.additions}/-{f.deletions}, changes={f.changes})\n"
                f"PATCH:\n{clip(f.patch, 1200)}\n"
            )

    return base.strip()


def extract_first_fenced_code_block(text: str) -> str:
    """
    Return ONLY the first fenced code block (```...```).
    If none found, wrap whole text in a plain ``` block as a fallback.
    """
    if not text:
        return "```diff\n```"

    m = re.search(r"```[a-zA-Z0-9_-]*\n.*?\n```", text, flags=re.DOTALL)
    if m:
        return m.group(0).strip()

    # fallback: if it contains ``` but weirdly formatted, return from first fence onward
    if "```" in text:
        first = text.find("```")
        return text[first:].strip()

    return f"```\n{text.strip()}\n```"


# ----------------------------
# Gemini retry wrapper (sync)
# ----------------------------
T = TypeVar("T")


def _is_transient_gemini_error(exc: Exception) -> bool:
    """
    Best-effort transient detection for overload / rate limit / gateway issues.
    Gemini errors can surface with different exception types depending on runtime;
    we rely on message heuristics + common status codes.
    """
    msg = (str(exc) or "").lower()

    # common transient signals
    transient_markers = [
        "503",
        "overloaded",
        "unavailable",
        "resource exhausted",
        "rate limit",
        "quota",
        "429",
        "timeout",
        "timed out",
        "deadline exceeded",
        "connection reset",
        "connection aborted",
        "bad gateway",
        "502",
        "gateway timeout",
        "504",
        "internal error",
        "500",
        "temporarily",
        "try again",
    ]
    return any(m in msg for m in transient_markers)


def gemini_call_with_retry(
    call_name: str,
    fn: Callable[[], T],
    *,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    initial_delay: float = RETRY_INITIAL_DELAY_SEC,
    max_delay: float = RETRY_MAX_DELAY_SEC,
    jitter: float = RETRY_JITTER_SEC,
) -> T:
    """
    Retries transient Gemini failures (e.g., 503 overloaded).
    Prints attempts to stderr so you can track retries in logs.
    """
    attempt = 1
    delay = max(0.0, initial_delay)

    while True:
        try:
            print(f"[gemini] {call_name}: attempt {attempt}/{max_attempts}", file=sys.stderr)
            return fn()
        except Exception as e:
            transient = _is_transient_gemini_error(e)
            print(
                f"[gemini] {call_name}: attempt {attempt} failed "
                f"(transient={transient}) -> {type(e).__name__}: {str(e)[:220]}",
                file=sys.stderr,
            )

            # If it's not transient, fail fast
            if not transient:
                raise

            # If we've exhausted attempts, re-raise
            if attempt >= max_attempts:
                raise

            # Sleep then retry
            # add a tiny jitter to avoid synchronized retries across concurrent requests
            sleep_for = min(max_delay, delay) + random.uniform(0.0, max(0.0, jitter))
            print(f"[gemini] {call_name}: sleeping {sleep_for:.2f}s before retry", file=sys.stderr)
            time.sleep(sleep_for)

            # Gentle backoff
            delay = min(max_delay, max(delay, 0.05) * 1.5)
            attempt += 1


# ----------------------------
# Gemini calls (sync)
# ----------------------------
def classify_with_gemini(payload: ReviewPayload) -> Classification:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system_instructions = """
You are a code review assistant that classifies a GitHub PR inline review comment
into exactly ONE category.

Decision priority:
1) Determine intent: praise / question / request change
2) Determine clarity: good / bad

Categories: PRAISE, GOOD_CHANGE, BAD_CHANGE, GOOD_QUESTION, BAD_QUESTION

Rules:
- "bad" = unclear/underspecified (not rude)
- needs_reply true ONLY for: GOOD_CHANGE, BAD_CHANGE, BAD_QUESTION
- needs_clarification true ONLY for: BAD_CHANGE, BAD_QUESTION
- Unknown intent -> UNKNOWN with low confidence

Return ONLY valid JSON for the schema.
""".strip()

    ctx = build_llm_context(payload)

    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"{system_instructions}\n\nCONTEXT:\n{ctx}")],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Classification,
                temperature=0.2,
            ),
        )

        data = getattr(resp, "parsed", None)
        if data is None:
            # sometimes resp.text may be empty on failures; let that raise
            data = json.loads(resp.text)

        return Classification.model_validate(data)

    return gemini_call_with_retry("classify_with_gemini", _call)


def clarify_bad_question(payload: ReviewPayload, cls: Classification) -> ClarifiedQuestion:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system_instructions = """
Rewrite an unclear PR question into a clarified question.

Rules:
- Output must match the JSON schema.
- 1–2 short sentences max, end with "?".
- Do NOT answer. Do NOT invent facts.
- Use placeholders if missing: "<which file?>", "<which function?>", "<expected behavior?>"
""".strip()

    ctx = build_llm_context(payload)

    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"{system_instructions}\n\nCONTEXT:\n{ctx}")],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClarifiedQuestion,
                temperature=0.2,
            ),
        )

        data = getattr(resp, "parsed", None)
        if data is None:
            data = json.loads(resp.text)

        return ClarifiedQuestion.model_validate(data)

    return gemini_call_with_retry("clarify_bad_question", _call)


def clarify_bad_change(payload: ReviewPayload, cls: Classification) -> ClarifiedChange:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system_instructions = """
Rewrite an unclear PR change request into a clarified, actionable request.

Rules:
- Output must match the JSON schema.
- Do NOT propose code. Do NOT invent facts.
- "clarified_request" must be 1–2 short sentences max.
- Use placeholders if missing: "<which file?>", "<which function?>", "<acceptance criteria?>"
""".strip()

    ctx = build_llm_context(payload)

    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"{system_instructions}\n\nCONTEXT:\n{ctx}")],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClarifiedChange,
                temperature=0.2,
            ),
        )

        data = getattr(resp, "parsed", None)
        if data is None:
            data = json.loads(resp.text)

        return ClarifiedChange.model_validate(data)

    return gemini_call_with_retry("clarify_bad_change", _call)


def generate_code_suggestion(
    payload: ReviewPayload,
    cls: Classification,
    reviewer_comment_override: Optional[str] = None,
) -> str:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    reviewer_comment = (reviewer_comment_override or payload.comment_body or payload.review_body or "").strip()
    ctx = build_llm_context(payload)

    system_instructions = f"""
You are a GitHub code review assistant.

Goal: produce a SHORT, STRICT code suggestion for the requested change.

Hard rules:
- Output MUST be ONLY ONE fenced code block and NOTHING else.
- The code block language MUST be either:
  1) ```diff  (preferred)
  2) ```suggestion  (only if diff isn't possible)
- Keep it minimal: change ONLY the smallest relevant lines.
- Do NOT rewrite whole files. Do NOT include unrelated context.
- If unsure, output a SMALL diff that adds TODOs/placeholders rather than guessing.

Comment to satisfy (source of truth):
{reviewer_comment}
""".strip()

    prompt = f"""
{system_instructions}

CONTEXT (reference only):
---
{ctx}
---

Return ONLY the single fenced code block now.
""".strip()

    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return extract_first_fenced_code_block((resp.text or "").strip())

    return gemini_call_with_retry("generate_code_suggestion", _call)


# ----------------------------
# Formatting helpers
# ----------------------------
def format_debug_comment(payload: ReviewPayload, cls: Classification) -> str:
    where = "review" if payload.kind == "review" else "inline comment"
    original_text = payload.review_body if payload.kind == "review" else payload.comment_body
    original_text = (original_text or "").strip()

    lines = [
        "🧠 **ContextWizard (debug: classification only)**",
        f"- event: `{where}`",
        f"- category: **{cls.category}**",
        f"- confidence: `{cls.confidence:.2f}`",
        f"- needs_reply: `{cls.needs_reply}`",
        f"- needs_clarification: `{cls.needs_clarification}`",
        f"- reason: {cls.short_reason}",
        "",
        "**Original text:**",
        f"> {(original_text[:500] + '…') if len(original_text) > 500 else original_text}".replace("\n", "\n> "),
        "",
        "_(classification only; no follow-up action taken)_",
    ]
    return "\n".join(lines).strip()


def format_clarification_question_comment(payload: ReviewPayload, cls: Classification, cq: ClarifiedQuestion) -> str:
    original_text = (payload.comment_body or payload.review_body or "").strip()

    lines = [
        "❓ **ContextWizard (clarified question)**",
        f"- category: **{cls.category}**",
        f"- classification_confidence: `{cls.confidence:.2f}`",
        f"- rewrite_confidence: `{cq.confidence:.2f}`",
        f"- reason: {cls.short_reason}",
        f"- rewrite_note: {cq.short_reason}",
        "",
        "**Original question:**",
        f"> {(original_text[:800] + '…') if len(original_text) > 800 else original_text}".replace("\n", "\n> "),
        "",
        "**Proposed clarified version:**",
        f"> {cq.clarified_question}".replace("\n", "\n> "),
    ]
    return "\n".join(lines).strip()


def format_bad_change_with_suggestion_comment(
    cls: Classification,
    clarified_request: str,
    suggestion_block: str,
) -> str:
    return "\n".join(
        [
            f"1- **clarified version:** {clarified_request}",
            "2- **suggested code change:**",
            suggestion_block.strip(),
        ]
    ).strip()


# ----------------------------
# FastAPI route
# ----------------------------
@app.post("/analyze-review", response_model=BackendResponse)
async def analyze_review(payload: ReviewPayload):
    print(f"Processing kind: {payload.kind} for PR #{payload.pr_number}", file=sys.stderr)
    if payload.kind == "wizard_review_command":
        try:
            suggestions = await anyio.to_thread.run_sync(run_wizard_full_review, payload)
            return BackendResponse(comment=f"🧙‍♂️ **Wizard Review Suggestions**\n\n{suggestions}")
        except Exception as e:
            return BackendResponse(comment=f"❌ Error during Wizard Review: {str(e)[:100]}")
        
    print("==== Incoming payload ====", file=sys.stderr)
    try:
        print(json.dumps(payload.model_dump(), indent=2), file=sys.stderr)
    except Exception:
        print(json.dumps(payload.dict(), indent=2), file=sys.stderr)
    print("==========================", file=sys.stderr)

    # 1) Classify
    print("Classifying with Gemini...", file=sys.stderr)
    try:
        cls = await anyio.to_thread.run_sync(classify_with_gemini, payload)
    except Exception as e:
        cls = Classification(
            category="UNKNOWN",
            needs_reply=True,
            needs_clarification=False,
            confidence=0.0,
            short_reason=f"Gemini classification failed: {type(e).__name__}: {str(e)[:160]}",
        )
        return BackendResponse(comment=format_debug_comment(payload, cls))

    # Only do follow-up actions for inline comments
    if payload.kind != "review_comment":
        return BackendResponse(comment=format_debug_comment(payload, cls))

    # 2) GOOD_CHANGE -> strict short code suggestion (diff/suggestion only)
    if cls.category == "GOOD_CHANGE" and cls.confidence >= 0.7:
        print("Generating good change with Gemini...", file=sys.stderr)
        try:
            suggestion_block = await anyio.to_thread.run_sync(generate_code_suggestion, payload, cls, None)
            return BackendResponse(comment=suggestion_block)
        except Exception as e:
            fallback = Classification(
                category="UNKNOWN",
                needs_reply=True,
                needs_clarification=False,
                confidence=0.0,
                short_reason=f"Suggestion generation failed: {type(e).__name__}: {str(e)[:160]}",
            )
            return BackendResponse(comment=format_debug_comment(payload, fallback))

    # 3) BAD_QUESTION -> clarified question message
    if cls.category == "BAD_QUESTION" and cls.confidence >= 0.55:
        print("Clarifying bad question with Gemini...", file=sys.stderr)
        try:
            cq = await anyio.to_thread.run_sync(clarify_bad_question, payload, cls)
            return BackendResponse(comment=format_clarification_question_comment(payload, cls, cq))
        except Exception as e:
            fallback = Classification(
                category="UNKNOWN",
                needs_reply=True,
                needs_clarification=False,
                confidence=0.0,
                short_reason=f"Question clarification failed: {type(e).__name__}: {str(e)[:160]}",
            )
            return BackendResponse(comment=format_debug_comment(payload, fallback))

    # 4) BAD_CHANGE -> clarify -> code suggestion -> reply includes BOTH
    if cls.category == "BAD_CHANGE" and cls.confidence >= 0.55:
        print("Clarifying bad change and generating suggestion with Gemini...", file=sys.stderr)
        try:
            cc = await anyio.to_thread.run_sync(clarify_bad_change, payload, cls)
            suggestion_block = await anyio.to_thread.run_sync(
                generate_code_suggestion,
                payload,
                cls,
                cc.clarified_request,
            )
            body = format_bad_change_with_suggestion_comment(cls, cc.clarified_request, suggestion_block)
            return BackendResponse(comment=body)
        except Exception as e:
            fallback = Classification(
                category="UNKNOWN",
                needs_reply=True,
                needs_clarification=False,
                confidence=0.0,
                short_reason=f"BAD_CHANGE clarification/suggestion failed: {type(e).__name__}: {str(e)[:160]}",
            )
            return BackendResponse(comment=format_debug_comment(payload, fallback))

    # 5) Default: classification debug comment
    return BackendResponse(comment=format_debug_comment(payload, cls))


def run_wizard_full_review(payload: ReviewPayload) -> str:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system_instructions = """
    You are the 'ContextWizard' AI Reviewer. 
    Your goal is to perform an autonomous code review of the provided changes.
    
    Rules:
    1. Analyze the Diff Hunks and Changed Files provided in the context.
    2. Identify bugs, security risks, or performance issues.
    3. For each issue, provide:
       - ### [TITLE]: A short, descriptive title of the problem.
       - **Description**: A clear explanation of what is wrong and how to fix it.
    4. Be concise and professional.
    """
    
    ctx = build_llm_context(payload)
    
    def _call():
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user", 
                    parts=[types.Part(text=f"{system_instructions}\n\nCONTEXT:\n{ctx}")]
                )
            ],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return resp.text

    return gemini_call_with_retry("wizard_review", _call)