// probot-app/index.js
const axios = require("axios");

/**
 * Env
 */
function getBackendUrl(context) {
  const url = process.env.BACKEND_URL;
  if (!url) {
    context.log.error("BACKEND_URL is not set in environment variables");
    return null;
  }
  return url;
}

/**
 * Ignore events from bots (your app, dependabot, etc.)
 */
function isFromBot(context) {
  const sender = context.payload.sender;
  if (!sender) return false;
  if (sender.type === "Bot") return true;
  if (sender.login && sender.login.endsWith("[bot]")) return true;
  return false;
}

/**
 * Call backend: POST payload -> expects { comment: string }
 */
async function callBackend(context, payloadForBackend) {
  const backendUrl = getBackendUrl(context);
  if (!backendUrl) return null;

  context.log.info(
    { kind: payloadForBackend.kind, pr: payloadForBackend.pr_number },
    "Sending payload to backend"
  );

  try {
    const res = await axios.post(backendUrl, payloadForBackend, {
      headers: { "Content-Type": "application/json" },
      timeout: 30_000
    });

    const commentBody = res?.data?.comment;
    if (!commentBody || !commentBody.trim()) {
      context.log.info("Backend returned empty comment, skipping.");
      return null;
    }
    return commentBody;
  } catch (err) {
    context.log.error({ err }, "Error calling backend");
    return null;
  }
}

/**
 * Helper: fetch changed files for a PR (includes unified diff patch when available)
 */
async function getPrFiles(context, owner, repo, prNumber) {
  const files = [];
  let page = 1;

  while (true) {
    const res = await context.octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: prNumber,
      per_page: 100,
      page
    });

    if (!res.data.length) break;

    for (const f of res.data) {
      files.push({
        filename: f.filename,
        status: f.status,
        additions: f.additions,
        deletions: f.deletions,
        changes: f.changes,
        patch: f.patch
      });
    }

    if (res.data.length < 100) break;
    page += 1;
  }

  return files;
}

/**
 * Build backend payload for a single inline review comment event
 */
async function buildReviewCommentPayload(context) {
  const comment = context.payload.comment;
  const pr = context.payload.pull_request;
  const repo = context.payload.repository;

  const commentBodyOriginal = (comment.body || "").trim();
  if (!commentBodyOriginal) return null;

  const owner = repo.owner.login;
  const repoName = repo.name;
  const prNumber = pr.number;

  const files = await getPrFiles(context, owner, repoName, prNumber);

  return {
    kind: "review_comment",

    // review-level fields (unused for this kind)
    review_body: null,
    review_state: null,

    // inline comment fields
    comment_body: commentBodyOriginal,
    comment_path: comment.path,
    comment_diff_hunk: comment.diff_hunk,
    comment_position: comment.position,
    comment_id: comment.id,

    // shared metadata
    reviewer_login: comment.user && comment.user.login,
    pr_number: prNumber,
    pr_title: pr.title,
    pr_body: pr.body,
    pr_author_login: pr.user && pr.user.login,
    repo_full_name: repo.full_name,
    repo_owner: owner,
    repo_name: repoName,

    // diff context
    files,

    // not included for single inline comment
    review_comments: null
  };
}


/**
 * Build backend payload for a submitted review event (top-level comment)
 */
async function buildReviewPayload(context) {
  const review = context.payload.review;
  const pr = context.payload.pull_request;
  const repo = context.payload.repository;

  const reviewBodyOriginal = (review.body || "").trim();

  // A review with no body is usually just an APPROVED status, which we can ignore
  if (!reviewBodyOriginal) return null;

  const owner = repo.owner.login;
  const repoName = repo.name;
  const prNumber = pr.number;

  const files = await getPrFiles(context, owner, repoName, prNumber);

  return {
    kind: "review",

    // review-level fields
    review_body: reviewBodyOriginal,
    review_state: review.state,

    // inline comment fields (unused for this kind)
    comment_body: null,
    comment_path: null,
    comment_diff_hunk: null,
    comment_position: null,
    comment_id: null,

    // shared metadata
    reviewer_login: review.user && review.user.login,
    pr_number: prNumber,
    pr_title: pr.title,
    pr_body: pr.body,
    pr_author_login: pr.user && pr.user.login,
    repo_full_name: repo.full_name,
    repo_owner: owner,
    repo_name: repoName,

    // diff context
    files,

    review_comments: null
  };
}


/**
 * Post reply to the inline comment thread
 */
async function replyToInlineComment(context, owner, repoName, prNumber, commentId, body) {
  await context.octokit.pulls.createReplyForReviewComment({
    owner,
    repo: repoName,
    pull_number: prNumber,
    comment_id: commentId,
    body
  });
}

/**
 * Post reply to the top-level PR thread (for reviews)
 */
async function replyToPrThread(context, owner, repoName, prNumber, body) {
  await context.octokit.issues.createComment({
    owner,
    repo: repoName,
    issue_number: prNumber,
    body
  });
}

/**
 * Main Probot app
 */
module.exports = (app) => {
  // ----------------------------------------------
  // 1. Handle single inline review comment
  // ----------------------------------------------
  app.on("pull_request_review_comment.created", async (context) => {
    try {
      console.log("Received pull_request_review_comment.created event");
      if (isFromBot(context)) return;

      const commentBody = (context.payload.comment.body || "").trim();
      const isWizardCmd = commentBody.startsWith("/wizard-review");

      const payloadForBackend = await buildReviewCommentPayload(context);
      if (!payloadForBackend) return;

      if (isWizardCmd) {
        console.log("Detected /wizard-review command in inline comment");
        payloadForBackend.kind = "wizard_review_command";
      }

      const replyBody = await callBackend(context, payloadForBackend);
      if (!replyBody) return;

      const owner = payloadForBackend.repo_owner;
      const repoName = payloadForBackend.repo_name;
      const prNumber = payloadForBackend.pr_number;

      await replyToInlineComment(context, owner, repoName, prNumber, context.payload.comment.id, replyBody);

    } catch (err) {
      context.log.error({ err }, "Error in wizard-review trigger");
    }
  });

  // ----------------------------------------------
  // 2. Handle submitted Pull Request Review (top-level comment)
  // ----------------------------------------------
  app.on("pull_request_review.submitted", async (context) => {
    try {
      if (isFromBot(context)) {
        context.log.info("Skipping event from bot sender.");
        return;
      }

      const reviewBody = context.payload.review.body;
      if (!reviewBody || reviewBody.trim() === "") {
        context.log.info("Skipping review with empty body.");
        return;
      }

      const payloadForBackend = await buildReviewPayload(context);
      if (!payloadForBackend) {
        context.log.info("Review payload construction failed, skipping.");
        return;
      }

      const replyBody = await callBackend(context, payloadForBackend);
      if (!replyBody) return;

      const repo = context.payload.repository;
      const pr = context.payload.pull_request;

      const owner = repo.owner.login;
      const repoName = repo.name;
      const prNumber = pr.number;

      // Post the reply as a new top-level comment on the PR thread
      await replyToPrThread(context, owner, repoName, prNumber, replyBody);

      context.log.info(
        { pr: prNumber, review_id: context.payload.review.id },
        "Replied to submitted review."
      );
    } catch (err) {
      context.log.error({ err }, "Error while handling pull_request_review.submitted");
    }
  });
};