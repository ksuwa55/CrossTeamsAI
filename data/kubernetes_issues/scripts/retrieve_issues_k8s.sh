#!/usr/bin/env bash
# Retrieve raw evidence for the Golden Dataset issues:
#   raw/issue_<id>.json          issue body, comments, final state + stateReason
#   raw/timeline_<id>.json       full REST Timeline API (closed/reopened, cross-references, labels, ...)
#   raw/closed_events_<id>.json  GraphQL Closed/Reopened events incl. `closer` (the PR/commit that closed it)
#   raw/prs/pr_<owner>_<repo>_<n>.json  every PR linked to the issue (mergedAt, mergeCommit, ...)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_OWNER="kubernetes"
REPO_NAME="kubernetes"
IDS=(109553 129652 115362 103587 113627)

CLOSED_EVENTS_QUERY='
query($owner: String!, $repo: String!, $num: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issue(number: $num) {
      timelineItems(first: 100, after: $cursor, itemTypes: [CLOSED_EVENT, REOPENED_EVENT]) {
        pageInfo { hasNextPage endCursor }
        nodes {
          __typename
          ... on ClosedEvent {
            id createdAt stateReason url
            actor { login }
            closer {
              __typename
              ... on PullRequest { number url repository { nameWithOwner } }
              ... on Commit { oid url }
            }
          }
          ... on ReopenedEvent { id createdAt actor { login } }
        }
      }
    }
  }
}'

mkdir -p raw/prs

for id in "${IDS[@]}"; do
    echo "Issue #${id}"

    gh issue view "$id" \
        --repo "${REPO_OWNER}/${REPO_NAME}" \
        --json number,title,body,state,stateReason,createdAt,closedAt,author,labels,comments,url,closedByPullRequestsReferences \
        > "raw/issue_${id}.json"

    # REST Timeline API; --slurp + jq flattens the paginated pages into one array
    gh api "repos/${REPO_OWNER}/${REPO_NAME}/issues/${id}/timeline?per_page=100" \
        --paginate --slurp \
        | jq 'add // []' > "raw/timeline_${id}.json"

    gh api graphql --paginate --slurp \
        -F owner="$REPO_OWNER" -F repo="$REPO_NAME" -F num="$id" \
        -f query="$CLOSED_EVENTS_QUERY" \
        | jq '[.[].data.repository.issue.timelineItems.nodes[]]' > "raw/closed_events_${id}.json"

    # Linked PRs = cross-referenced PRs + PRs that closed the issue + closing references
    prs=$(jq -rn \
        --slurpfile tl "raw/timeline_${id}.json" \
        --slurpfile ce "raw/closed_events_${id}.json" \
        --slurpfile iss "raw/issue_${id}.json" '
        (
          ($tl[0][] | select(.event == "cross-referenced" and .source.issue.pull_request != null)
                    | "\(.source.issue.repository.full_name) \(.source.issue.number)"),
          ($ce[0][] | select(.closer.__typename? == "PullRequest")
                    | "\(.closer.repository.nameWithOwner) \(.closer.number)"),
          ($iss[0].closedByPullRequestsReferences[]?
                    | "\(.repository.owner.login)/\(.repository.name) \(.number)")
        )' | sort -u)

    while read -r pr_repo pr_num; do
        [ -z "${pr_repo:-}" ] && continue
        out="raw/prs/pr_${pr_repo//\//_}_${pr_num}.json"
        [ -f "$out" ] && continue
        echo "  PR ${pr_repo}#${pr_num}"
        gh pr view "$pr_num" --repo "$pr_repo" \
            --json number,title,state,author,createdAt,mergedAt,closedAt,mergedBy,mergeCommit,url,baseRefName,isDraft \
            > "$out" || { echo "  WARN: could not fetch ${pr_repo}#${pr_num}"; rm -f "$out"; }
    done <<< "$prs"
done
