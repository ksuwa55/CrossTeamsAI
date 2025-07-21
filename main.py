import argparse
from summarizer import MeetingSummarizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["general", "decision", "blocker", "query"], default="general")
    parser.add_argument("--query", type=str, help="Optional query for 'query' mode")
    parser.add_argument("--input", default="data/zoom_transcript_sample.json", help="Path to input transcript")
    parser.add_argument("--output", default="output/summary.json", help="Path to output summary file")

    args = parser.parse_args()

    summarizer = MeetingSummarizer()
    transcript = summarizer.load_and_preprocess_transcript(args.input)
    prompt = summarizer.build_prompt(transcript, mode=args.mode, query=args.query)
    raw_output = summarizer.run_summarizer(prompt)
    parsed = summarizer.parse_output(raw_output)
    summarizer.save_summary(parsed, args.output)
    summarizer.log_experiment({"transcript": transcript, "prompt": prompt}, parsed, {"model": "gpt-3.5-turbo"})
