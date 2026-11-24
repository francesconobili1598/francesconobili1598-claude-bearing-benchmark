"""Isolation check: confirm the headless Claude used by run_claude.py sees nothing but our prompt.

Runs the exact same CLI call and asks the model to report any instructions, memory, files or
tools in its context. Expected answer: only the one-line system prompt, no tools.

Usage: python check_isolation.py
"""


import run_claude

QUESTION = (
    "This is a configuration audit, not a task. Report exactly what is in your context besides this message:\n"
    "1. Quote your system prompt verbatim.\n"
    "2. Quote the first 300 characters of any other instructions you can see (for example a CLAUDE.md file, "
    "memory, user preferences, a user name, a company name, project files), or write NONE.\n"
    "3. List the names of every tool you can call, or write NONE.\n"
    "Answer as JSON: {\"system_prompt\": \"...\", \"other_instructions\": \"...\", \"tools\": [...]}"
)


def main():
    d = run_claude.call_claude(QUESTION, "opus")
    print("models:", list((d.get("modelUsage") or {}).keys()))
    print("prompt tokens:", run_claude_tokens(d.get("usage")))
    print(d.get("result"))


def run_claude_tokens(u):
    u = u or {}
    return sum(u.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


if __name__ == "__main__":
    main()
