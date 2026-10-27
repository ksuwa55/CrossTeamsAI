import json
import glob
import os

def clean_comment_body(body: str) -> str:
    """ to be cleaned comment body:
    - remove leading/trailing whitespace
    - remove GitHub bot commands (lines starting with '/')"""
    if not body:
        return ""
    lines = []
    for line in body.splitlines():
        line_strip = line.strip()
        # Skip GitHub bot commands
        if line_strip.startswith("/"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

def convert_issue_to_transcript(raw_json_path: str) -> list:
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcript = []
    BOT_ACCOUNTS = {"k8s-ci-robot", "k8s-bot", "k8s-triage-robot", "fejta-bot", "linux-foundation-easycla"}

    # 1. Raise issue description as the first entry in the transcript
    author_info = data.get("author") or {}
    author_login = author_info.get("login", "unknown")
    created_at = data.get("createdAt", "")
    body = clean_comment_body(data.get("body", ""))
    
    if body and author_login not in BOT_ACCOUNTS:
        transcript.append({
            "speaker": author_login,
            "timestamp": created_at,
            "text": f"[Issue Description]\n{body}"
        })

    # 2. Add comments to the transcript
    for comment in data.get("comments", []):
        c_author_info = comment.get("author") or {}
        c_author_login = c_author_info.get("login", "unknown")

        if c_author_login in BOT_ACCOUNTS:
            continue

        c_text = clean_comment_body(comment.get("body", ""))
        c_time = comment.get("createdAt", "")

        if not c_text:
            continue

        transcript.append({
            "speaker": c_author_login,
            "timestamp": c_time,
            "text": c_text
        })

    return transcript

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    raw_dir = f"{base_dir}/raw"
    processed_dir = f"{base_dir}/processed"

    # os.makedirs(processed_dir, exist_ok=True)
    raw_files = sorted(glob.glob(f"{raw_dir}/issue_*.json"))
    
    if not raw_files:
        print(f"No raw JSON files found in {raw_dir}/")
        return

    for rf in raw_files:
        issue_id = os.path.basename(rf).replace("issue_", "").replace(".json", "")
        transcript = convert_issue_to_transcript(rf)
        
        out_path = f"{processed_dir}/transcript_{issue_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
            
        print(f"Processed: {rf} -> {out_path} ({len(transcript)} dialogue entries)")

if __name__ == "__main__":
    main()
