# Kubernetes Issues — Golden Dataset Source Data

Raw GitHub data and preprocessed transcripts for the five Golden Dataset issues
(103587, 109553, 113627, 115362, 129652). The pipeline only collects and joins
evidence; it does **not** annotate decisions or compute latencies (Steps 3–4 are manual).

## Pipeline

```
scripts/retrieve_issues_k8s.sh   →  raw/        (download from GitHub)
scripts/preprocess_k8s.py        →  processed/  (normalize + join)
```

## Raw files (`raw/`)

| File | Source | Contents |
|---|---|---|
| `issue_<id>.json` | `gh issue view` | Title, body, comments, labels, `state`, `stateReason`, `closedByPullRequestsReferences` |
| `timeline_<id>.json` | REST Timeline API `GET /repos/{owner}/{repo}/issues/{id}/timeline` | All events: closed/reopened, cross-referenced, labeled, assigned, renamed, ... |
| `closed_events_<id>.json` | GraphQL `timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])` | Close/reopen events incl. `closer` (the PR or commit that closed the issue) |
| `prs/pr_<owner>_<repo>_<n>.json` | `gh pr view` | One file per linked PR: `state`, `createdAt`, `mergedAt`, `mergedBy`, `mergeCommit` |

**Why GraphQL is needed:** in Kubernetes, issues are closed by `k8s-ci-robot` after a
PR merges, so the REST `closed` event has `commit_id: null` and does not say *what*
closed the issue. Only GraphQL `ClosedEvent.closer` exposes the closing PR.

`raw/prs/` is shared across all issues (24 PRs in total). Many are abandoned or
unmerged attempts — a linked PR is a *candidate*, not proof of implementation.

## Processed files (`processed/transcript_<id>.json`)

| Section | Contents |
|---|---|
| `metadata` | Issue info + `state_reason` |
| `transcript` | Issue description + all comments (incl. bot comments) |
| `timeline` | Lifecycle events, time-ordered (comments/subscribed/unsubscribed/mentioned excluded) |
| `linked_prs` | PR facts for every PR linked to the issue |
| `outcome_evidence` | Summary of final close; `implementation_evidence` is an automatic hint, not ground truth |

## Join logic: Issue ↔ PR (important)

**Join key:** `owner/repo#number`, e.g. `kubernetes/kubernetes#131837`.

The same key is read from each raw file:

```
timeline_<id>.json        source.issue.repository.full_name + source.issue.number   ┐
closed_events_<id>.json   closer.repository.nameWithOwner  + closer.number          │
issue_<id>.json           closedByPullRequestsReferences[].repository               ├─→ owner/repo#number
                            .owner.login/.name + .number                            │
prs/pr_*.json             owner/repo parsed from "url" + "number"                   ┘
```

Example (issue 103587, PR #131837):

| Raw file | Field | Value |
|---|---|---|
| `timeline_103587.json` | `cross-referenced` → `source.issue` | `kubernetes/kubernetes`, `131837`, linked 2025-05-18 |
| `closed_events_103587.json` | `ClosedEvent.closer` | `kubernetes/kubernetes`, `131837`, closed 2025-06-26T19:18:31Z |
| `prs/pr_kubernetes_kubernetes_131837.json` | `url`, `number` | merged 2025-06-26T19:18:30Z |

### Where it happens in `scripts/preprocess_k8s.py`

1. **Build key** — `pr_key()`: `f"{repo}#{number}"`.
2. **Index PR files** — `load_prs()`: parses owner/repo from each PR's `url`;
   called once in `main()`.
3. **Extract keys from issue data** — `convert_timeline_event()`:
   - `cross-referenced` event → `details.source` (only when `source.issue.pull_request` exists)
   - `closed` event → `details.closer.pr`
4. **Collect + look up** — `build_lifecycle()`:
   - union of cross-referenced PRs, closer PRs and `closedByPullRequestsReferences`
     (`linked_keys`)
   - `prs_index[key]` → `convert_pr()` → `linked_prs`
   - flags: `closed_this_issue` (key was a closer), `is_closing_reference`
     (key in `closedByPullRequestsReferences`); both false = only mentioned

### Second join: REST `closed` event ↔ GraphQL `ClosedEvent`

The REST timeline and GraphQL use different IDs, so close/reopen events are matched on
**(event type, timestamp)** via `closed_events_by_time` in `build_lifecycle()`. The
GraphQL id is kept as `graphql_id` and the closer as `details.closer`.

## Traceability (raw → processed)

| Raw | Processed | `evidence_id` |
|---|---|---|
| `issue_<id>.json` comments | `transcript` | GitHub comment node id; description = `issue_<id>_description` |
| `timeline_<id>.json` events | `timeline` | `event_<id>` (REST event `id`) |
| `timeline_<id>.json` cross-references | `timeline` | `xref_<owner>_<repo>_<n>` (GitHub gives no id; built from source) |
| `closed_events_<id>.json` | `timeline` (closed/reopened) | REST `event_<id>` + `graphql_id` |
| `prs/pr_*.json` | `linked_prs` | `pr_<owner>_<repo>_<n>` |

## Known limitations

- `commented`, `subscribed`, `unsubscribed`, `mentioned` timeline events are dropped
  (comments are already in `transcript`).
- `project_v2_item_status_changed` has no from/to status in the REST API (`details` is empty).
- "Linked" ≠ "implements the decision": cross-references include any PR that merely
  mentioned the issue. Which PR (if any) implements a decision is an annotation judgment.
- A final close with `closer: null` (manual or `/close`) gives no PR evidence of implementation.
