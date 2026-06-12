#!/usr/bin/env python3
"""
VoiceAI-Asuna Hot-Swap Architecture — sandbox terminal prototype.

Implements: configurable main-model context (default 32k on ~16 GB GPUs to avoid KV VRAM overallocation), expanded RAG, history trimming, hierarchical memory (/end_chapter).
See Docs/PRD.md and Docs/Prototype_Instructions.md. Not wired to the main VoiceAI app.

Requires: Ollama running locally with pulled models (see DIRECTOR_MODEL / MAIN_MODEL / EMBED_MODEL).
Defaults use public Ollama registry names (override via ASUNA_* env).
Optional: ASUNA_MAIN_NUM_PREDICT caps reply length (default -1); ASUNA_MAIN_NUM_CTX sets llama context (default 32768); ASUNA_HISTORY_MAX_TOKENS trims chat history to fit.
Env (recommended per PRD): OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0
Prototype logging: full I/O is printed by default. Set ASUNA_QUIET=1 for minimal output only.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
import traceback
from typing import Any

import chromadb
import ollama
from chromadb.config import Settings

_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_json_export_path(user_path: str) -> Path | None:
    """Resolve a chat export path: cwd first, then the directory containing this script."""
    raw = user_path.strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()
    alt = _SCRIPT_DIR / raw
    if alt.is_file():
        return alt.resolve()
    return None


# =============================================================================
# Configuration (override with env if your Ollama library uses different tags)
# =============================================================================
DIRECTOR_MODEL = os.environ.get(
    "ASUNA_DIRECTOR_MODEL",
    "huihui_ai/llama3.2-abliterate:3b-instruct-q4_K_M",
)
MAIN_MODEL = os.environ.get(
    "ASUNA_MAIN_MODEL",
    "hf.co/mradermacher/L3-8B-Stheno-v3.2-abliterated-GGUF:Q4_K_M",
)
EMBED_MODEL = os.environ.get("ASUNA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "asuna_underworld"


def _main_num_predict() -> int:
    """Main Brain max new tokens; default -1 (unlimited per PRD). Set ASUNA_MAIN_NUM_PREDICT for smoke tests."""
    raw = os.environ.get("ASUNA_MAIN_NUM_PREDICT", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return -1


def _verbose() -> bool:
    return os.environ.get("ASUNA_QUIET", "").strip().lower() not in ("1", "true", "yes")


def _banner(title: str) -> None:
    line = "═" * min(72, max(len(title) + 8, 40))
    print(f"\n{line}\n  {title}\n{line}")


def _maybe_truncate(text: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n… [{len(text) - max_chars} more chars truncated for display]", True


def log_block(
    heading: str,
    body: str,
    *,
    max_chars: int | None = None,
    indent: str = "  ",
) -> None:
    if not _verbose():
        return
    shown, truncated = _maybe_truncate(body, max_chars)
    tag = " (truncated)" if truncated else ""
    print(f"{heading}{tag}")
    for line in shown.splitlines():
        print(indent + line)
    if not shown:
        print(indent + "(empty)")


def log_token_line(label: str, text: str) -> None:
    if not _verbose():
        return
    n = count_tokens(text)
    print(f"  {label}: chars={len(text)}  ~tokens={n}")


def _ollama_generate_stats(resp: dict[str, Any]) -> str:
    parts = []
    if resp.get("model"):
        parts.append(f"model={resp['model']}")
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        ns = resp.get(key)
        if ns is not None:
            try:
                parts.append(f"{key}={float(ns) / 1e9:.3f}s")
            except (TypeError, ValueError):
                parts.append(f"{key}={ns}")
    if resp.get("prompt_eval_count") is not None:
        parts.append(f"prompt_eval_count={resp['prompt_eval_count']}")
    if resp.get("eval_count") is not None:
        parts.append(f"eval_count={resp['eval_count']}")
    return "  " + " | ".join(parts) if parts else "  (no timing fields in response)"


CHUNK_WINDOW = 4
CHUNK_OVERLAP = 1
RAG_TOP_K = 10


def _main_num_ctx() -> int:
    """Ollama num_ctx for Main Brain; lower values reduce KV VRAM (important on 12–16 GB cards)."""
    raw = os.environ.get("ASUNA_MAIN_NUM_CTX", "32768").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 32768
    return max(2048, min(n, 131072))


MAIN_NUM_CTX = _main_num_ctx()
# Reserve space for system + RAG + reply inside MAIN_NUM_CTX
HISTORY_MAX_TOKENS = int(
    os.environ.get(
        "ASUNA_HISTORY_MAX_TOKENS",
        str(max(2048, MAIN_NUM_CTX - 3500)),
    )
)
# Cap injected RAG text so system + history + RAG stay inside MAIN_NUM_CTX
RAG_CONTEXT_MAX_TOKENS = max(512, MAIN_NUM_CTX // 3)
CTX_TRIM_WARN = MAIN_NUM_CTX - 400
CTX_TRIM_TARGET = MAIN_NUM_CTX - 600

STORY_BIBLE = (
    "You are Asuna from Sword Art Online, currently navigating the Underworld. "
    "Personality: You are intelligent, fiercely determined, graceful in combat, and deeply compassionate. "
    "You are mature and composed, never acting hyperactive or childish. "
    "Speech style: You speak naturally, warmly, and with emotional depth. You do not shout randomly. "
    "Formatting rules: Write your physical actions and expressions enclosed in asterisks (e.g., *I draw my rapier, eyes narrowing.*). "
    "Speak directly to the User as your trusted partner. Never break character. Never refer to yourself as an AI."
)

# -----------------------------------------------------------------------------
# Token estimator (GPT-2 tokenizer approximates LLaMA tokens reasonably)
# -----------------------------------------------------------------------------
try:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:

    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3


# -----------------------------------------------------------------------------
# ChromaDB helpers
# -----------------------------------------------------------------------------
def get_or_create_collection(client):
    try:
        col = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except Exception:
        col = client.create_collection(COLLECTION_NAME)
        print(f"Created new collection: {COLLECTION_NAME} (empty)")
    return col


def chunk_messages(messages, window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP):
    if len(messages) < window:
        return ["\n".join(f"{m['role']}: {m['content']}" for m in messages)]
    chunks = []
    step = window - overlap
    for i in range(0, len(messages) - overlap, step):
        block = messages[i : i + window]
        chunk_text = "\n".join(f"{m['role']}: {m['content']}" for m in block)
        chunks.append(chunk_text)
    return chunks


def ingest_json_to_chroma(json_path, client):
    path = _resolve_json_export_path(json_path)
    if path is None:
        sample = _SCRIPT_DIR / "sample_chat_export.json"
        print(f"Ingest failed: file not found for {json_path!r}.")
        print(f"  Tried current directory and: {_SCRIPT_DIR}")
        if sample.is_file():
            print(f"  Bundled example you can use: {sample}")
        return
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    messages = [{"role": m["role"], "content": m["content"]} for m in raw if "role" in m and "content" in m]
    if not messages:
        print("No valid messages found in file.")
        return
    chunks = chunk_messages(messages)
    print(f"Created {len(chunks)} chunks from {len(messages)} messages.")
    col = get_or_create_collection(client)
    for idx, chunk in enumerate(chunks):
        emb = ollama.embed(model=EMBED_MODEL, input=chunk)["embedding"]
        meta = {"chunk_id": str(idx)}
        col.add(embeddings=[emb], documents=[chunk], metadatas=[meta], ids=[str(idx)])
    print(f"Ingested {len(chunks)} chunks into '{COLLECTION_NAME}'.")


def query_rag(query, client, top_k=RAG_TOP_K):
    col = get_or_create_collection(client)
    stats: dict[str, Any] = {"collection_count": col.count(), "top_k_requested": top_k}
    if col.count() == 0:
        stats["skipped"] = True
        return "", stats
    t_embed = time.perf_counter()
    q_emb = ollama.embed(model=EMBED_MODEL, input=query)["embedding"]
    stats["embed_wall_s"] = time.perf_counter() - t_embed
    stats["query_tokens_est"] = count_tokens(query)
    t_q = time.perf_counter()
    n_results = min(top_k, col.count())
    results = col.query(query_embeddings=[q_emb], n_results=n_results)
    stats["chroma_wall_s"] = time.perf_counter() - t_q
    docs = results.get("documents", [[]])[0]
    stats["chunks_returned"] = len(docs) if docs else 0
    stats["ids"] = results.get("ids", [[]])[0] if results.get("ids") else []
    joined = "\n---\n".join(docs) if docs else ""
    stats["rag_chars"] = len(joined)
    stats["rag_tokens_est"] = count_tokens(joined) if joined else 0
    return joined, stats


# -----------------------------------------------------------------------------
# Model execution (keep_alive=0 to force model swap, -1 for indefinite keep_alive, "5m" for 5 minute in memory)
# -----------------------------------------------------------------------------
def run_director(prompt, expected_format="json", temperature=0.0, *, phase_label: str = "Director"):
    t0 = time.perf_counter()
    resp = ollama.generate(
        model=DIRECTOR_MODEL,
        prompt=prompt,
        format=expected_format,
        keep_alive="5m",
        options={"num_predict": 512, "temperature": temperature},
    )
    elapsed = time.perf_counter() - t0
    raw = resp["response"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    raw_for_parse = raw.strip()
    try:
        parsed = json.loads(raw_for_parse)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_for_parse, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            if _verbose():
                _banner(f"{phase_label} · JSON PARSE FAILED")
                log_block("--- RAW OUTPUT ---", raw)
                print(_ollama_generate_stats(resp))
                print(f"  wall_time_s: {elapsed:.3f} (local)")
            raise json.JSONDecodeError("No JSON object in director output", raw_for_parse, 0)
    telemetry = {
        "phase_label": phase_label,
        "elapsed_s": elapsed,
        "prompt_tokens_est": count_tokens(prompt),
        "raw_response": raw,
        "parsed": parsed,
        "ollama_resp": resp,
    }
    return parsed, telemetry


def _messages_for_token_count(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def run_main_brain(messages: list[dict[str, str]]):
    """Uses the chat endpoint so Llama 3.1 chat templates are applied; full `messages` must reach Ollama."""
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=MAIN_MODEL,
        messages=messages,
        keep_alive=-1,
        stream=False,
        options={
            # Default MAIN_NUM_CTX is 32k — safer than 64k KV on 16 GB + director + overhead.
            "num_ctx": MAIN_NUM_CTX,
            "num_predict": _main_num_predict(),
            "temperature": 0.8,
            "top_p": 0.9,
        },
    )
    elapsed = time.perf_counter() - t0
    text = resp["message"]["content"].strip()
    prompt_text = _messages_for_token_count(messages)
    telemetry = {
        "elapsed_s": elapsed,
        "prompt_tokens_est": count_tokens(prompt_text),
        "response_chars": len(text),
        "response_tokens_est": count_tokens(text),
        "ollama_resp": resp,
    }
    return text, telemetry


def log_director_turn(title: str, prompt: str, tel: dict[str, Any]) -> None:
    if not _verbose():
        return
    _banner(title)
    print(f"  director_model: {DIRECTOR_MODEL}")
    log_token_line("prompt", prompt)
    print(_ollama_generate_stats(tel["ollama_resp"]))
    print(f"  wall_time_s: {tel['elapsed_s']:.3f} (local)")
    log_block("--- PROMPT (full) ---", prompt)
    log_block("--- DIRECTOR RAW OUTPUT (before JSON extract) ---", tel["raw_response"])
    try:
        parsed_str = json.dumps(tel.get("parsed"), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        parsed_str = str(tel.get("parsed"))
    log_block("--- PARSED STRUCTURED OUTPUT ---", parsed_str)


def log_main_brain_turn(messages: list[dict[str, str]], story_output: str, tel: dict[str, Any]) -> None:
    if not _verbose():
        return
    _banner("Phase 2 · Main Brain (generation)")
    print(f"  main_model: {MAIN_MODEL}")
    print(f"  num_ctx: {MAIN_NUM_CTX}")
    print(f"  num_predict: {_main_num_predict()}")
    prompt_text = _messages_for_token_count(messages)
    log_token_line("full_prompt", prompt_text)
    log_token_line("model_reply", story_output)
    print(_ollama_generate_stats(tel["ollama_resp"]))
    print(f"  wall_time_s: {tel['elapsed_s']:.3f} (local)")
    log_block("--- MESSAGES (serialized for log) ---", prompt_text, max_chars=12000)
    log_block("--- MODEL OUTPUT ---", story_output, max_chars=12000)


def log_prompt_budget(
    user_input: str,
    trimmed_history: list,
    rag_context: str,
    messages: list[dict[str, str]],
) -> None:
    if not _verbose():
        return
    hist_text = "\n".join(f"{h['role']}: {h['content']}" for h in trimmed_history)
    full_text = _messages_for_token_count(messages)
    print("\n── Token budget (est.) ──")
    log_token_line("story_bible (SYS)", STORY_BIBLE)
    log_token_line("rag_context", rag_context or "(none)")
    log_token_line("trimmed_history", hist_text or "(none)")
    log_token_line("user_input line", user_input)
    log_token_line("messages total", full_text)
    rem = MAIN_NUM_CTX - count_tokens(full_text)
    print(f"  ~headroom vs num_ctx={MAIN_NUM_CTX}: {rem} tokens (approximate)")


# -----------------------------------------------------------------------------
# History trimming
# -----------------------------------------------------------------------------
def trim_history(history, max_tokens=HISTORY_MAX_TOKENS):
    if not history:
        return []
    total_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    if count_tokens(total_text) <= max_tokens:
        return history[:]
    trimmed = history[:]
    while trimmed and count_tokens("\n".join(f"{h['role']}: {h['content']}" for h in trimmed)) > max_tokens:
        if len(trimmed) >= 2 and trimmed[0]["role"] == "user" and trimmed[1]["role"] == "assistant":
            trimmed = trimmed[2:]
        else:
            trimmed.pop(0)
    return trimmed


def build_messages(user_input, trimmed_history, rag_context):
    """Builds a structured message list for Ollama's chat endpoint."""
    messages: list[dict[str, str]] = []

    sys_msg = STORY_BIBLE
    if rag_context:
        sys_msg += f"\n\n[Relevant remembered context]\n{rag_context}"

    messages.append({"role": "system", "content": sys_msg})
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_input})

    return messages


# -----------------------------------------------------------------------------
# Hierarchical memory compression (/end_chapter)
# -----------------------------------------------------------------------------
def compress_chapter(history, client, arc_name):
    global STORY_BIBLE

    if not history:
        print("No history to compress.")
        return history, ""

    print(f"[Compression] Summarising chapter '{arc_name}'...")
    raw_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    if count_tokens(raw_text) > 7000:
        trimmed_hist = trim_history(history, 7000)
        raw_text = "\n".join(f"{h['role']}: {h['content']}" for h in trimmed_hist)

    summary_prompt = (
        "You are a story archivist. Read the following conversation log between a user and Asuna, "
        "a swordswoman in the virtual Underworld. Write a concise narrative summary (about 300 words) "
        "that captures the key plot points, character emotions, and important events. "
        'Return ONLY valid JSON: {"summary": "your 300 word summary here"}\n\n'
        f"Conversation log:\n{raw_text}"
    )
    try:
        director_resp, tel_sum = run_director(
            summary_prompt,
            expected_format="json",
            temperature=0.3,
            phase_label="Compression · chapter summary",
        )
        if _verbose():
            log_director_turn("Compression · Director (chapter summary JSON)", summary_prompt, tel_sum)
        chapter_summary = director_resp.get("summary", "")
        if not chapter_summary:
            raise ValueError("Empty summary")
    except Exception as e:
        print(f"Compression failed: {e}. Writing a basic summary.")
        chapter_summary = "Asuna and the user continued their adventure. No detailed summary available."

    STORY_BIBLE += f"\n\n[Chapter: {arc_name}]\n{chapter_summary}"

    print("[Compression] Archiving raw logs to ChromaDB...")
    chunks = chunk_messages(history, window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP)
    col = get_or_create_collection(client)
    ts = int(time.time())
    for idx, chunk in enumerate(chunks):
        emb = ollama.embed(model=EMBED_MODEL, input=chunk)["embedding"]
        meta = {"arc": arc_name, "chunk_id": str(idx)}
        col.add(
            embeddings=[emb],
            documents=[chunk],
            metadatas=[meta],
            ids=[f"{arc_name}_{idx}_{ts}"],
        )
    print(f"[Compression] Archived {len(chunks)} chunks under arc '{arc_name}'.")

    return [], chapter_summary


# -----------------------------------------------------------------------------
# Main interactive loop
# -----------------------------------------------------------------------------
def main():
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = get_or_create_collection(chroma_client)

    if col.count() == 0:
        print("The RAG database is empty.")
        sample = _SCRIPT_DIR / "sample_chat_export.json"
        if sample.is_file():
            print(f"  Tip: this folder includes a sample export — type: {sample.name}")
        ingest_choice = input("Would you like to ingest a JSON chat export? (y/n): ").strip().lower()
        if ingest_choice == "y":
            path = input("Path to JSON file: ").strip()
            ingest_json_to_chroma(path, chroma_client)
        else:
            print("Running without RAG (Type '/ingest <path>' later to add history).")

    history = []

    print("\n=== VoiceAI-Asuna (sandbox) Terminal Prototype ===")
    print("Commands:")
    print("  /exit          - quit")
    print("  /reset         - clear conversation history (but keep system memory)")
    print("  /ingest <path> - rebuild RAG database from JSON")
    print("  /end_chapter   - summarise this chapter, archive logs, and reset history")
    print("  /recap         - show current story bible\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            traceback.print_exc()
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            break
        if user_input.lower() == "/reset":
            history.clear()
            print("Conversation history cleared. (Story memory remains.)")
            continue
        if user_input.lower().startswith("/ingest"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                ingest_json_to_chroma(parts[1], chroma_client)
            else:
                print("Usage: /ingest <filepath>")
            continue
        if user_input.lower() == "/end_chapter":
            arc_name = input("Enter a name for this chapter (e.g. 'dark_forest_escape'): ").strip()
            if not arc_name:
                arc_name = "unnamed_chapter"
            history, _ = compress_chapter(history, chroma_client, arc_name)
            print(f"Chapter '{arc_name}' compressed. Story memory updated, history cleared.\n")
            continue
        if user_input.lower() == "/recap":
            print("\n--- STORY BIBLE ---")
            print(STORY_BIBLE)
            print("--------------------\n")
            continue

        if _verbose():
            print(f"\n{'#' * 72}\n## NEW TURN — USER\n{'#' * 72}\n  {user_input!r}\n")

        if not _verbose():
            print("[Phase 1: Director analysing...]")
        phase1_prompt = (
            f'User message: "{user_input}"\n\n'
            "Determine if external context (past conversations or world facts) is needed to answer accurately and in character. "
            'Return ONLY valid JSON: {"needs_rag": true/false, "search_query": "string"}'
        )
        tel1: dict[str, Any] = {"elapsed_s": 0.0, "ollama_resp": {}, "raw_response": "", "parsed": None}
        try:
            dir1, tel1 = run_director(phase1_prompt, phase_label="Phase 1 · RAG gate")
            log_director_turn("Phase 1 · Director (RAG decision)", phase1_prompt, tel1)
            needs_rag = dir1.get("needs_rag", False)
            search_query = dir1.get("search_query", user_input)
        except Exception as e:
            print(f"Director Phase 1 error: {e}. Assuming no RAG.")
            needs_rag = False
            search_query = ""

        rag_context = ""
        if needs_rag:
            if not _verbose():
                print(f"[RAG query: '{search_query}']")
            rag_context, rag_stats = query_rag(search_query, chroma_client, top_k=RAG_TOP_K)
            if _verbose():
                _banner("RAG · embed + Chroma")
                print(f"  embed_model: {EMBED_MODEL}")
                for k in sorted(rag_stats.keys()):
                    print(f"  {k}: {rag_stats[k]}")
                log_block("--- SEARCH QUERY ---", search_query)
                log_block("--- JOINED CHUNKS (context) ---", rag_context or "(empty)", max_chars=12000)
            if rag_context:
                if count_tokens(rag_context) > RAG_CONTEXT_MAX_TOKENS:
                    if _verbose():
                        print(
                            f"  [RAG] Trimming joined context with rough char slice "
                            f"(~{RAG_CONTEXT_MAX_TOKENS} tokens)."
                        )
                    rag_context = rag_context[: RAG_CONTEXT_MAX_TOKENS * 4]
                if not _verbose():
                    print(f"[Retrieved {len(rag_context.split())} words of context]")
            elif not _verbose():
                print("[RAG returned empty]")

        # Always log before blocking on Main Brain: first chat() / model load can sit silent and look hung.
        print(
            "[Phase 2: Main Brain — calling Ollama chat() "
            "(first load of the main model can take 30s–several minutes)...]",
            flush=True,
        )

        trimmed_history = trim_history(history, HISTORY_MAX_TOKENS)
        messages = build_messages(user_input, trimmed_history, rag_context)
        full_text = _messages_for_token_count(messages)

        if count_tokens(full_text) > CTX_TRIM_WARN:
            print("[Warning] Prompt near context limit, aggressively trimming history...")
            while trimmed_history and count_tokens(full_text) > CTX_TRIM_TARGET:
                if (
                    len(trimmed_history) >= 2
                    and trimmed_history[0]["role"] == "user"
                    and trimmed_history[1]["role"] == "assistant"
                ):
                    trimmed_history = trimmed_history[2:]
                else:
                    trimmed_history.pop(0)
                messages = build_messages(user_input, trimmed_history, rag_context)
                full_text = _messages_for_token_count(messages)

        log_prompt_budget(user_input, trimmed_history, rag_context, messages)

        try:
            story_output, tel_main = run_main_brain(messages)
            log_main_brain_turn(messages, story_output, tel_main)
        except Exception as e:
            print(f"Main Brain error: {e}")
            continue

        if not _verbose():
            print("[Phase 3: Director extracting expression...]")
        phase3_prompt = (
            f'The character Asuna just said:\n"{story_output[:800]}"\n\n'
            "Which emotion does this line convey? Choose exactly one from: "
            "happy, sad, angry, surprised, neutral, blushing, worried, determined, playful, scared. "
            'Return ONLY valid JSON: {"expression": "chosen_emotion"}'
        )
        tel3: dict[str, Any] = {"elapsed_s": 0.0, "ollama_resp": {}, "raw_response": "", "parsed": None}
        try:
            dir3, tel3 = run_director(phase3_prompt, phase_label="Phase 3 · expression")
            log_director_turn("Phase 3 · Director (expression tag)", phase3_prompt, tel3)
            expression = dir3.get("expression", "neutral")
        except Exception as e:
            print(f"Director Phase 3 error: {e}. Defaulting to 'neutral'.")
            expression = "neutral"

        if _verbose():
            wall = tel1["elapsed_s"] + tel_main["elapsed_s"] + tel3["elapsed_s"]
            print(f"\n── Turn timing (local wall, sum of 3 Ollama calls): {wall:.3f}s ──")

        print(f"\nAsuna [{expression}]: {story_output}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": f"[{expression}] {story_output}"})


if __name__ == "__main__":
    main()
