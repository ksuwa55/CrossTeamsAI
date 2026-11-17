import csv
import json
import glob
import os

def clean_comment_body(body: str) -> str:
    """Clean the comment body:
    - Only strip leading/trailing whitespace.
    - Do not remove GitHub bot commands (lines starting with '/') or code blocks.
    """
    if not body:
        return ""
    # Keep commands and code blocks as required by guidelines
    return body.strip()

def convert_issue_to_transcript(raw_json_path: str) -> dict:
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract issue-level metadata
    issue_meta = {
        "issue_id": data.get("number"),
        "title": data.get("title"),
        "state": data.get("state"),
        "url": data.get("url"),
        "created_at": data.get("createdAt"),
        "closed_at": data.get("closedAt"),
        "state_reason": data.get("stateReason"),
        "labels":[label.get("name") for label in data.get("labels", [])],
    }

    transcript = []

    # 1. Add the issue description as the first entry in the transcript
    author_info = data.get("author") or {}
    author_login = author_info.get("login", "unknown")
    body = clean_comment_body(data.get("body", ""))
    
    if body:
        transcript.append({
            "evidence_id": f"issue_{data.get('number')}_description",
            "speaker": author_login,
            "is_bot": False,  # Flag for bot accounts
            "timestamp": data.get("createdAt"),
            "text": f"[Issue Description]\n{body}",
            "url": data.get("url")
        })

    # 2. Add all comments (keeping bot activities and commands without filtering them out)
    BOT_ACCOUNTS = {"k8s-ci-robot", "k8s-bot", "k8s-triage-robot", "fejta-bot", "linux-foundation-easycla"}

    for comment in data.get("comments", []):
        c_author_info = comment.get("author") or {}
        c_author_login = c_author_info.get("login", "unknown")
        is_bot = c_author_login in BOT_ACCOUNTS

        c_text = clean_comment_body(comment.get("body", ""))
        if not c_text:
            continue

        comment_id = comment.get("id", "unknown_comment_id")
        c_time = comment.get("createdAt", "")
        c_url = comment.get("url", "")

        transcript.append({
            "evidence_id": comment_id,  # Unique ID for traceability back to raw GitHub source
            "speaker": c_author_login,
            "is_bot": is_bot,           # Retain bot data while marking the flag
            "timestamp": c_time,
            "text": c_text,
            "url": c_url
        })

    return {
        "metadata": issue_meta,
        "transcript": transcript
    }

# Timeline events that carry no lifecycle information (comments are already in the transcript)
SKIP_TIMELINE_EVENTS = {"commented", "subscribed", "unsubscribed", "mentioned"}

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def pr_key(repo: str, number) -> str:
    return f"{repo}#{number}"

def load_prs(prs_dir: str) -> dict:
    """Index raw/prs/pr_<owner>_<repo>_<n>.json by 'owner/repo#n'."""
    prs = {}
    for path in glob.glob(f"{prs_dir}/pr_*.json"):
        pr = load_json(path, {})
        # url: https://github.com/<owner>/<repo>/pull/<n>
        parts = pr.get("url", "").split("/")
        if len(parts) >= 7:
            prs[pr_key(f"{parts[3]}/{parts[4]}", pr.get("number"))] = pr
    return prs

def convert_pr(pr: dict) -> dict:
    repo = "/".join(pr["url"].split("/")[3:5])
    return {
        "evidence_id": f"pr_{repo.replace('/', '_')}_{pr.get('number')}",
        "repo": repo,
        "number": pr.get("number"),
        "title": pr.get("title"),
        "author": (pr.get("author") or {}).get("login"),
        "state": pr.get("state"),            # OPEN / CLOSED / MERGED
        "is_draft": pr.get("isDraft"),
        "base_ref": pr.get("baseRefName"),
        "created_at": pr.get("createdAt"),
        "merged_at": pr.get("mergedAt"),
        "closed_at": pr.get("closedAt"),
        "merged_by": (pr.get("mergedBy") or {}).get("login"),
        "merge_commit": (pr.get("mergeCommit") or {}).get("oid"),
        "url": pr.get("url"),
    }

def convert_timeline_event(ev: dict, issue_id, closed_events_by_time: dict):
    """Normalize one REST Timeline API event; returns None for non-lifecycle noise."""
    event_type = ev.get("event")
    if event_type in SKIP_TIMELINE_EVENTS:
        return None

    entry = {
        "evidence_id": f"event_{ev['id']}" if ev.get("id") else None,
        "event_type": event_type,
        "actor": (ev.get("actor") or {}).get("login", "unknown"),
        "timestamp": ev.get("created_at"),
        "details": {},
        "url": f"https://github.com/kubernetes/kubernetes/issues/{issue_id}#event-{ev['id']}" if ev.get("id") else None,
    }
    details = entry["details"]

    if event_type in ("closed", "reopened"):
        details["state_reason"] = ev.get("state_reason")
        details["commit_id"] = ev.get("commit_id")
        # The REST API does not expose what closed the issue; take `closer` from the GraphQL event
        gql = closed_events_by_time.get((event_type, ev.get("created_at")))
        if gql:
            entry["graphql_id"] = gql.get("id")
            closer = gql.get("closer")
            if closer and closer.get("__typename") == "PullRequest":
                details["closer"] = {"type": "PullRequest",
                                     "pr": pr_key(closer["repository"]["nameWithOwner"], closer["number"]),
                                     "url": closer.get("url")}
            elif closer and closer.get("__typename") == "Commit":
                details["closer"] = {"type": "Commit", "oid": closer.get("oid"), "url": closer.get("url")}
            elif event_type == "closed":
                details["closer"] = None
    elif event_type == "cross-referenced":
        src = ev.get("source", {}).get("issue", {})
        repo = src.get("repository", {}).get("full_name")
        is_pr = src.get("pull_request") is not None
        entry["evidence_id"] = f"xref_{(repo or '').replace('/', '_')}_{src.get('number')}"
        entry["url"] = src.get("html_url")
        details.update({
            "source_type": "pull_request" if is_pr else "issue",
            "source": pr_key(repo, src.get("number")),
            "source_title": src.get("title"),
            "source_state": src.get("state"),
        })
        if is_pr:
            details["source_merged_at"] = src["pull_request"].get("merged_at")
    elif event_type in ("labeled", "unlabeled"):
        details["label"] = (ev.get("label") or {}).get("name")
    elif event_type in ("assigned", "unassigned"):
        details["assignee"] = (ev.get("assignee") or {}).get("login")
    elif event_type in ("milestoned", "demilestoned"):
        details["milestone"] = (ev.get("milestone") or {}).get("title")
    elif event_type == "renamed":
        details["from"] = (ev.get("rename") or {}).get("from")
        details["to"] = (ev.get("rename") or {}).get("to")
    elif event_type == "referenced":
        details["commit_id"] = ev.get("commit_id")
        details["commit_url"] = ev.get("commit_url")
        entry["url"] = ev.get("commit_url") or entry["url"]

    if entry["evidence_id"] is None:
        entry["evidence_id"] = f"event_{issue_id}_{event_type}_{ev.get('created_at')}"
    return entry

def build_lifecycle(raw_dir: str, issue_id: str, data: dict, prs_index: dict) -> dict:
    timeline_raw = load_json(f"{raw_dir}/timeline_{issue_id}.json", [])
    closed_raw = load_json(f"{raw_dir}/closed_events_{issue_id}.json", [])

    closed_events_by_time = {
        ("closed" if ce["__typename"] == "ClosedEvent" else "reopened", ce.get("createdAt")): ce
        for ce in closed_raw
    }

    timeline = []
    for ev in timeline_raw:
        entry = convert_timeline_event(ev, issue_id, closed_events_by_time)
        if entry:
            timeline.append(entry)
    timeline.sort(key=lambda e: e["timestamp"] or "")

    # Linked PRs: cross-referenced PRs, PRs that closed the issue, closing references
    linked_keys = set()
    closer_keys = set()
    for e in timeline:
        if e["event_type"] == "cross-referenced" and e["details"]["source_type"] == "pull_request":
            linked_keys.add(e["details"]["source"])
        closer = e["details"].get("closer")
        if closer and closer["type"] == "PullRequest":
            linked_keys.add(closer["pr"])
            closer_keys.add(closer["pr"])
    closing_ref_keys = {
        pr_key(f"{r['repository']['owner']['login']}/{r['repository']['name']}", r["number"])
        for r in data.get("closedByPullRequestsReferences", [])
    }
    linked_keys |= closing_ref_keys

    linked_prs = []
    for key in sorted(linked_keys):
        if key not in prs_index:
            print(f"  WARN: {key} linked to #{issue_id} but missing in raw/prs/")
            continue
        pr = convert_pr(prs_index[key])
        pr["closed_this_issue"] = key in closer_keys
        pr["is_closing_reference"] = key in closing_ref_keys
        linked_prs.append(pr)
    linked_prs.sort(key=lambda p: p["created_at"] or "")

    # Outcome evidence: observable facts only, not an interpretation of the decision outcome
    close_events = [e for e in timeline if e["event_type"] == "closed"]
    final_close = close_events[-1] if data.get("state") == "CLOSED" and close_events else None
    final_closer = (final_close or {}).get("details", {}).get("closer")
    closer_pr = None
    if final_closer and final_closer["type"] == "PullRequest":
        closer_pr = next((p for p in linked_prs if pr_key(p["repo"], p["number"]) == final_closer["pr"]), None)

    if data.get("state") != "CLOSED":
        implementation_evidence = "open"
    elif closer_pr and closer_pr["merged_at"]:
        implementation_evidence = "closed_by_merged_pr"
    elif any(p["merged_at"] for p in linked_prs):
        implementation_evidence = "closed_with_merged_linked_pr"   # merged PR exists but did not close the issue
    else:
        implementation_evidence = "closure_only"                   # insufficient as implementation evidence

    outcome = {
        "final_state": data.get("state"),
        "state_reason": data.get("stateReason"),
        "closed_at": data.get("closedAt"),
        "close_count": len(close_events),
        "reopen_count": sum(1 for e in timeline if e["event_type"] == "reopened"),
        "final_close_evidence_id": (final_close or {}).get("evidence_id"),
        "final_close_actor": (final_close or {}).get("actor"),
        "final_closer": final_closer,
        "final_closer_pr_merged_at": (closer_pr or {}).get("merged_at"),
        "merged_linked_prs": [p["evidence_id"] for p in linked_prs if p["merged_at"]],
        "implementation_evidence": implementation_evidence,
    }
    return {"timeline": timeline, "linked_prs": linked_prs, "outcome_evidence": outcome}

# CSV columns = keys of each "transcript" entry in the JSON output
TRANSCRIPT_CSV_COLUMNS = ["evidence_id", "speaker", "is_bot", "timestamp", "text", "url"]

def write_transcript_csv(transcript: list, out_path: str):
    # utf-8-sig so Excel displays non-ASCII characters correctly
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRANSCRIPT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(transcript)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    raw_dir = f"{base_dir}/raw"
    json_dir = f"{base_dir}/processed_json"
    csv_dir = f"{base_dir}/processed_csv"

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    raw_files = sorted(glob.glob(f"{raw_dir}/issue_*.json"))
    
    if not raw_files:
        print(f"No raw JSON files found in {raw_dir}/")
        return

    prs_index = load_prs(f"{raw_dir}/prs")

    for rf in raw_files:
        issue_id = os.path.basename(rf).replace("issue_", "").replace(".json", "")
        processed_data = convert_issue_to_transcript(rf)
        processed_data.update(build_lifecycle(raw_dir, issue_id, load_json(rf, {}), prs_index))

        out_path = f"{json_dir}/transcript_{issue_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        csv_path = f"{csv_dir}/transcript_{issue_id}.csv"
        write_transcript_csv(processed_data["transcript"], csv_path)

        print(f"Processed: {rf} -> {out_path}, {csv_path} ({len(processed_data['transcript'])} entries, "
              f"{len(processed_data['timeline'])} timeline events, {len(processed_data['linked_prs'])} linked PRs, "
              f"{processed_data['outcome_evidence']['implementation_evidence']})")

if __name__ == "__main__":
    main()