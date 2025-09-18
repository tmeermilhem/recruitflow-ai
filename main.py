# main.py
from __future__ import annotations
import sys
from agent.agent import build_agent

INTRO = """
RecruitFlow AI – Your Recruitment Assistant
-------------------------------------------

I help HR teams quickly find and evaluate the best candidates.

Here’s how it works:
• Paste a job listing below.
• I’ll scan through the available resumes and pull out the most relevant candidates.
• Each candidate will receive a personalized questionnaire to clarify their experience and skills.
• Once answers are in, I’ll score and rank them, with short explanations for transparency.
• Finally, I’ll present the top 10 candidates you can confidently invite for interviews.

Please paste your job listing now (end input with Ctrl-D / Ctrl-Z).
""".strip()


def main() -> None:
    # Print intro for HR users
    print(INTRO)

    # Read job listing until EOF (Ctrl-D on mac/Linux, Ctrl-Z on Windows)
    raw = sys.stdin.read().strip()
    if not raw:
        print("\nNo job listing provided. Exiting.")
        return

    # Build the autonomous agent
    agent = build_agent()

    # Run the pipeline
    print("\n> Running RecruitFlow AI pipeline...\n")
    result = agent.invoke({"input": raw, "chat_history": []})

    # Print final result (paths to outputs)
    print("\n=== RecruitFlow AI Finished ===")
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print(result)


if __name__ == "__main__":
    main()