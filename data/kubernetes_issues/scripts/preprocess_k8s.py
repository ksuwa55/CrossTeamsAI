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
        "labels": [label.get("name") for label in data.get("labels", [])],
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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    raw_dir = f"{base_dir}/raw"
    processed_dir = f"{base_dir}/processed"

    os.makedirs(processed_dir, exist_ok=True)
    raw_files = sorted(glob.glob(f"{raw_dir}/issue_*.json"))
    
    if not raw_files:
        print(f"No raw JSON files found in {raw_dir}/")
        return

    for rf in raw_files:
        issue_id = os.path.basename(rf).replace("issue_", "").replace(".json", "")
        processed_data = convert_issue_to_transcript(rf)
        
        out_path = f"{processed_dir}/transcript_{issue_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed: {rf} -> {out_path} ({len(processed_data['transcript'])} entries)")

if __name__ == "__main__":
    main()