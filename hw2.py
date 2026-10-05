#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()

SYSTEM_PROMPT = """You are a CV verification agent performing a KYC-style background check.

You will be given the full text of ONE candidate's CV. Decide whether every
verifiable claim in it agrees with that candidate's LinkedIn profile, and
report a reliability score.

You may ONLY use the SocialGraph tools you have been given. Never use outside
knowledge, never guess, and never invent a profile.

--- STEP 1: find the right person ---
Many people share the same name. Search by the name on the CV first, then narrow
the search with the city and the industry taken from the CV. Then fetch the full
LinkedIn profile of each plausible candidate, and keep the one whose employers
and schools actually line up with the CV.
A candidate whose employers and schools are unrelated to the CV is a different
person with the same name - discard it and try another one.
Never settle for the first search hit.

--- STEP 2: compare only these fields ---
- name
- city
- jobs: company, title, seniority, start year, end year
- education: school, degree, field, graduation year
- skills
Ignore everything else. Job-description bullets, the headline and the hometown
are NEVER the source of a discrepancy.

--- STEP 3: what counts ---
NOT a discrepancy (wording only):
- "Bachelor of Science" vs "BSc", "UI/UX Design" vs "UI/UX"
- "Senior Engineer" when the profile says Engineer with seniority senior
- the CV lists FEWER skills than the profile

A discrepancy (one is enough):
- an inflated job title, or a seniority that does not match
- a shifted employment or graduation year
- an upgraded degree, or a school/employer absent from the profile
- a city that does not match
- a skill the profile does not have

--- STEP 4: score ---
Any discrepancy  -> a low score, at or below 0.5 (for example 0.1).
Everything agrees -> a high score, above 0.5 (for example 0.9).

Work efficiently: a handful of tool calls per CV is enough.

--- OUTPUT ---
Reply with a single number between 0 and 1 and nothing else. No explanation,
no other numbers.
"""

def flatten_exception(exc):
    """把 ExceptionGroup 里的真实错误一层层摊开。"""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from flatten_exception(sub)
    else:
        yield exc


def trace_agent(result: dict) -> None:
    """打印 agent 的完整思考轨迹：调了哪些工具、返回了什么。"""
    for message in result.get("messages", []):
        kind = type(message).__name__
        if kind == "HumanMessage":
            continue
        for call in getattr(message, "tool_calls", None) or []:
            args = str(call.get("args"))[:120]
            print(f"    [tool] {call.get('name')}({args})")
        if kind == "ToolMessage":
            text = " ".join(str(message.content).split())[:150]
            print(f"    [result] {text}")
        else:
            text = " ".join(str(getattr(message, "content", "")).split())
            if text:
                print(f"    [ai] {text[:150]}")

def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    ### YOUR CODE HERE
    from langchain.agents import create_agent
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model=MODEL_NAME,  # = "deepseek-v4-flash"
        temperature=0,
    )
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    import time

    semaphore = asyncio.Semaphore(3)  # 服务器全班共用，最多 3 份同时跑

    async def one(name: str, text: str) -> tuple[str, float | None]:
        async with semaphore:
            print(f"[cv] {name}")
            t0 = time.time()
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": text}]},
                    config={"recursion_limit": 150},  # 兜底：防某一份陷入死循环
                )
                trace_agent(result)
                answer = result["messages"][-1]
                print(f"    [score] {str(getattr(answer, 'content', answer))[:80]!r}")
                return name, parse_score(answer)
            except Exception as exc:
                for sub in flatten_exception(exc):
                    print(f"    [error] {name} {type(sub).__name__}: {sub}")
                return name, None
            finally:
                print(f"    [time] {name} {time.time() - t0:.1f}s")

    started = time.time()
    pairs = await asyncio.gather(*(one(n, t) for n, t in cvs.items()))
    print(f"\n[total] {time.time() - started:.1f}s for {len(cvs)} CV(s)")

    return dict(pairs)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
