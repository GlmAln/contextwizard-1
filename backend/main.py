# backend/main.py
from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import json
import sys

from google import genai
types = genai.types  # alias for convenience

app = FastAPI()


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
# Gemini structured output model
# ----------------------------
Category = Literal[
    "PRAISE",
    "GOOD_CHANGE",
    "BAD_CHANGE",
    "GOOD_QUESTION",
    "BAD_QUESTION",
    "UNKNOWN",  # safety fallback
]


class Classification(BaseModel):
    category: Category
    needs_reply: bool = Field(..., description="True only for GOOD_CHANGE, BAD_CHANGE, BAD_QUESTION.")
    needs_clarification: bool = Field(..., description="True only for BAD_CHANGE or BAD_QUESTION.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    short_reason: str = Field(..., description="One short sentence. No chain-of-thought.")


# ----------------------------
# Helpers
# ----------------------------
def get_client() -> genai.Client:
    """Returns the synchronous Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def clip(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "\n…(truncated)…"


def build_llm_context(payload: ReviewPayload) -> str:
    """
    Compact, high-signal context for classification and suggestion.
    """
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

    # Include changed files + patches (truncated)
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

# NOTE: This function is SYNCHRONOUS (no 'async')
def classify_with_gemini(payload: ReviewPayload) -> Classification:
    client = get_client() # Get synchronous client
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # The system instructions string for classification
    system_instructions = """
You are a code review assistant that classifies a GitHub PR inline review comment
into exactly ONE category.

Decision priority:
1) First determine the INTENT of the comment:
    - praise
    - question
    - request to change code
2) Then determine whether the intent is CLEAR (good) or UNCLEAR (bad).

Categories:
1) PRAISE:
    - Only positive reaction.
    - No actionable request.

2) GOOD_CHANGE:
    - Clear, actionable request to change code.

3) BAD_CHANGE:
    - A request to change code, but unclear, underspecified, or poorly explained.
    - Needs clarification before code can be suggested.

4) GOOD_QUESTION:
    - A clear question.
    - No action required by the bot.

5) BAD_QUESTION:
    - A question, but unclear, ambiguous, or underspecified.
    - Bot should ask clarifying questions.

Important notes:
- "bad" means unclear, ambiguous, or underspecified — NOT rude or toxic.
- Informal language, slang, sarcasm, emojis, or non-English text
    does NOT automatically make a comment bad.
- Short comments are allowed to be GOOD if intent is still clear.
- needs_reply must be true ONLY for:
    GOOD_CHANGE, BAD_CHANGE, BAD_QUESTION.
- needs_clarification must be true ONLY for:
    BAD_CHANGE, BAD_QUESTION.
- If intent cannot be determined, use category=UNKNOWN with low confidence.

Return ONLY valid JSON that matches the provided schema.
""".strip()

    ctx = build_llm_context(payload)    

    # NOTE: No 'await' keyword here.
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"{system_instructions}\n\nCONTEXT:\n{ctx}"
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Classification,
            temperature=0.2,
        ),
    )

    # Structured output parsing (safe fallback)
    data = getattr(resp, "parsed", None)
    if data is None:
        data = json.loads(resp.text)

    return Classification.model_validate(data)

# NOTE: This function is SYNCHRONOUS (no 'async')
def generate_code_suggestion(payload: ReviewPayload, cls: Classification) -> str:
    client = get_client() # Get synchronous client
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") # Use a powerful model for code

    # The system instructions string for code suggestion
    system_instructions = f"""
You are an expert GitHub code review assistant. Your task is to provide a helpful, actionable code suggestion in response to a peer review comment.

Constraints:
1. **Analyze the original comment** (provided below) and the surrounding **diff hunk**.
2. **Focus ONLY on the requested change.**
3. **If possible, provide the entire suggested file content or a complete function/class block that includes the fix.**
4. **Your final output MUST be a clean, direct Markdown code block.** Do not include any introductory or explanatory text outside the code block.

Reviewer's Intent (from Classification): {cls.category} - {cls.short_reason}
""".strip()

    # Reuse the context builder, but focus the prompt on the necessary fix
    ctx = build_llm_context(payload)

    # The full prompt string for code suggestion
    prompt = f"""
{system_instructions}

CONTEXT:
---
{ctx}
---

Your task: Based on the "Original comment" and the "Diff hunk" in the context above, provide the corrected/suggested code.

Return ONLY the code block in Markdown format.
"""

    # NOTE: No 'await' keyword here.
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt)
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
        ),
    )
    
    # Strip any potential leading/trailing text outside the code block
    # and return the generated text.
    return resp.text.strip()


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


# ----------------------------
# FastAPI route
# ----------------------------
@app.post("/analyze-review", response_model=BackendResponse)
async def analyze_review(payload: ReviewPayload):
    # Print payload for debug (keep this)
    print("==== Incoming payload ====", file=sys.stderr)
    try:
        print(json.dumps(payload.model_dump(), indent=2), file=sys.stderr)
    except Exception:
        print(json.dumps(payload.dict(), indent=2), file=sys.stderr) 
    print("==========================", file=sys.stderr)

    # 1. Classify the comment/review
    try:
        # MUST use 'await' here to run the synchronous helper in a thread pool
        cls = classify_with_gemini(payload)
    except Exception as e:
        # Fallback debug comment on classification failure
        cls = Classification(
            category="UNKNOWN",
            needs_reply=True,
            needs_clarification=False,
            confidence=0.0,
            short_reason=f"Gemini classification failed: {type(e).__name__}: {str(e)[:160]}",
        )
        # If classification fails, return the error immediately
        final_comment_body = format_debug_comment(payload, cls)
        return BackendResponse(comment=final_comment_body.strip())

    final_comment_body = ""

    # 2. Check if a code suggestion is warranted
    # Only attempt code suggestion for inline comments, NOT full reviews.
    if payload.kind == "review_comment" and cls.category == "GOOD_CHANGE" and cls.confidence >= 0.7:
        try:
            # MUST use 'await' here to run the synchronous helper in a thread pool
            suggestion = generate_code_suggestion(payload, cls)
            
            # Format the final comment body for GitHub
            final_comment_body = f"""
**🤖 Code Suggestion ({cls.category} - {cls.confidence:.2f})**

_Reviewer intent: {cls.short_reason}_

Here is a suggested implementation for the requested change:

{suggestion}

---
_(Generated by Gemini)_
"""
        except Exception as e:
            # Fallback if code generation fails
            final_comment_body = format_debug_comment(
                payload, 
                Classification(
                    category="UNKNOWN",
                    needs_reply=True,
                    needs_clarification=False,
                    confidence=0.0,
                    short_reason=f"Suggestion generation failed: {type(e).__name__}: {str(e)[:160]}",
                )
            )
    else:
        # For full reviews, or other categories/low confidence inline comments, return the classification debug comment
        final_comment_body = format_debug_comment(payload, cls)


    return BackendResponse(comment=final_comment_body.strip())