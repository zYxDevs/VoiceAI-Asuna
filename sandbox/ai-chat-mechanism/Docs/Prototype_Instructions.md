Great update! The 64k window opens up some exciting possibilities, and the hierarchical memory (down‑time compression) is a smart way to keep long‑running stories coherent without blowing the prompt. I’ll incorporate all the new requirements into the terminal prototype and also suggest a few model options that could work within your 12GB VRAM budget.

---

### Suggested Model Improvements (for 12GB VRAM)

**Main Brain (Story & Persona)**  
- **Current:** `Llama-3.1-8B-Stheno-v3.4` (IQ4_NL, ~4.9GB) – safe, proven, fits comfortably with 64k context cache.  
- **Possible upgrade:** `Mistral-Nemo-12B-Magnum-v4` (Q4_K_M ~7.5GB). Might just fit (7.5 + 3.0 context cache ≈ 10.5GB) but leaves zero headroom—not recommended unless you can offload some layers to CPU. **Stick with the 8B Stheno** for reliability.  
- **Alternative:** `Llama-3.3-8B-Stheno-v3.4` (if/when an abliterated version appears) – but the current 3.1 is fine.

**Director (Logic & Routing)**  
- **Current:** `Llama-3.2-3B-Instruct-abliterated` (Q4_K_M, ~2.24GB). Perfect for fast JSON.  
- **Fallback:** `Llama-3.2-1B-Instruct-abliterated` (even smaller) if you ever need to shave off loading time.  
- **Future:** `Phi-4-mini` (3.8B) once abliterated GGUF is available – could be more accurate while staying small.

**Embedding** – `nomic-embed-text` remains ideal.

---

### Updated Token Limits (now that we have 64k)

| Component                     | Approx. Token Budget | Notes |
|-------------------------------|----------------------|-------|
| System prompt (story bible)   | 500 – 1,000 tokens  | Will grow with chapter summaries |
| Active chat history           | 25,000 – 30,000 tokens | Sliding window of most recent turns |
| RAG injection (top‑10‑15 hits)| 5,000 – 8,000 tokens | ~5,000 words of relevant lore |
| Current user input & output   | 2,000 – 4,000 tokens | |
| **Total (worst case)**        | **~40,000 tokens**   | Well within 64k, leaves headroom |

*These numbers are safe even with the Q8_0 key‑value cache (which adds about 2.5‑3.0GB to VRAM).*

---

That’s the recommendation. Now, here’s the **revised terminal prototype** that implements:

- Expanded RAG retrieval (top 10 hits)
- Active history trimming to the last ~30k tokens
- The hierarchical memory compression via the `/end_chapter` command

```python
#!/usr/bin/env python3
"""
VoiceAI-Asuna Hot-Swap Architecture – Updated Terminal Prototype
Implements: 64k context, expanded RAG, automatic history trimming, 
and hierarchical memory compression (/end_chapter).
"""

import json
import sys
import ollama
import chromadb
from chromadb.config import Settings
import re
import time

# =============================================================================
# Configuration
# =============================================================================
DIRECTOR_MODEL = "Llama-3.2-3B-Instruct-abliterated:Q4_K_M"
MAIN_MODEL = "Llama-3.1-8B-Stheno-v3.4:IQ4_NL"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "asuna_underworld"

CHUNK_WINDOW = 4                # messages per RAG chunk
CHUNK_OVERLAP = 1               # last message of block A == first of block B
RAG_TOP_K = 10                  # increased from 3 to 10‑15 as per new PRD
HISTORY_MAX_TOKENS = 28000      # keep active history under ~28k tokens
SUMMARY_PROMPT_TOKEN_BUDGET = 500

# The persistent story bible (grows with chapter summaries)
STORY_BIBLE = (
    "You are Asuna, the Flash, a fierce and elegant swordswoman trapped in the virtual world of Underworld. "
    "You speak with a mixture of determination and gentle warmth. Always stay in character. Never break the fourth wall. "
    "You are currently adventuring through a dark forest, pursuing the Integrity Knights."
)

# -----------------------------------------------------------------------------
# Simple token estimator (GPT-2 tokenizer approximates LLaMA tokens reasonably)
# -----------------------------------------------------------------------------
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    # Fallback: rough estimate (1 token ≈ 0.75 words)
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
        block = messages[i:i+window]
        chunk_text = "\n".join(f"{m['role']}: {m['content']}" for m in block)
        chunks.append(chunk_text)
    return chunks

def ingest_json_to_chroma(json_path, client):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    messages = [{'role': m['role'], 'content': m['content']}
                for m in raw if 'role' in m and 'content' in m]
    if not messages:
        print("No valid messages found in file.")
        return
    chunks = chunk_messages(messages)
    print(f"Created {len(chunks)} chunks from {len(messages)} messages.")
    col = get_or_create_collection(client)
    for idx, chunk in enumerate(chunks):
        emb = ollama.embed(model=EMBED_MODEL, input=chunk)['embedding']
        meta = {"chunk_id": str(idx)}
        col.add(embeddings=[emb], documents=[chunk], metadatas=[meta], ids=[str(idx)])
    print(f"Ingested {len(chunks)} chunks into '{COLLECTION_NAME}'.")

def query_rag(query, client, top_k=RAG_TOP_K):
    col = get_or_create_collection(client)
    if col.count() == 0:
        return ""
    q_emb = ollama.embed(model=EMBED_MODEL, input=query)['embedding']
    results = col.query(query_embeddings=[q_emb], n_results=min(top_k, col.count()))
    docs = results.get('documents', [[]])[0]
    return "\n---\n".join(docs) if docs else ""

# -----------------------------------------------------------------------------
# Model execution (keep_alive=0 everywhere)
# -----------------------------------------------------------------------------
def run_director(prompt, expected_format="json", temperature=0.0):
    resp = ollama.generate(
        model=DIRECTOR_MODEL,
        prompt=prompt,
        format=expected_format,
        keep_alive=0,
        options={"num_predict": 512, "temperature": temperature}
    )
    raw = resp['response'].strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If the model returned extra text, try to extract JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

def run_main_brain(prompt):
    resp = ollama.generate(
        model=MAIN_MODEL,
        prompt=prompt,
        keep_alive=0,
        options={
            "num_ctx": 65536,
            "num_predict": -1,
            "temperature": 0.8,
            "repeat_penalty": 1.1,
            "stop": ["<|eot_id|>", "User:"]  # optional stop tokens
        }
    )
    return resp['response'].strip()

# -----------------------------------------------------------------------------
# History trimming (keeps only the most recent ~HISTORY_MAX_TOKENS tokens)
# -----------------------------------------------------------------------------
def trim_history(history, max_tokens=HISTORY_MAX_TOKENS):
    """Return a copy of history that fits within max_tokens."""
    if not history:
        return []
    # Convert whole history to string to measure tokens
    total_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    if count_tokens(total_text) <= max_tokens:
        return history[:]
    # Start removing from the oldest messages until under limit
    trimmed = history[:]
    while trimmed and count_tokens("\n".join(f"{h['role']}: {h['content']}" for h in trimmed)) > max_tokens:
        # Remove the oldest exchange (pair of user + assistant)
        if len(trimmed) >= 2 and trimmed[0]['role'] == 'user' and trimmed[1]['role'] == 'assistant':
            trimmed = trimmed[2:]
        else:
            trimmed.pop(0)
    return trimmed

# -----------------------------------------------------------------------------
# Hierarchical memory compression (/end_chapter)
# -----------------------------------------------------------------------------
def compress_chapter(history, client, arc_name):
    """
    Use Director to summarise the recent conversation into a ~300‑word summary.
    1) Generate summary.
    2) Add summary to global STORY_BIBLE.
    3) Chunk & embed the raw logs into ChromaDB with arc metadata.
    4) Return the summary and the new (cleared) history.
    """
    if not history:
        print("No history to compress.")
        return history, ""

    print(f"[Compression] Summarising chapter '{arc_name}'...")
    # Build the raw log text (full history, but maybe last 10k tokens)
    raw_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    # Truncate for Director if too long (Director can only handle ~8k tokens)
    if count_tokens(raw_text) > 7000:
        # Use the last ~7k tokens
        trimmed_hist = trim_history(history, 7000)
        raw_text = "\n".join(f"{h['role']}: {h['content']}" for h in trimmed_hist)

    summary_prompt = (
        "You are a story archivist. Read the following conversation log between a user and Asuna, "
        "a swordswoman in the virtual Underworld. Write a concise narrative summary (about 300 words) "
        "that captures the key plot points, character emotions, and important events. "
        "Return ONLY valid JSON: {\"summary\": \"your 300 word summary here\"}\n\n"
        f"Conversation log:\n{raw_text}"
    )
    try:
        director_resp = run_director(summary_prompt, expected_format="json", temperature=0.3)
        chapter_summary = director_resp.get("summary", "")
        if not chapter_summary:
            raise ValueError("Empty summary")
    except Exception as e:
        print(f"Compression failed: {e}. Writing a basic summary.")
        chapter_summary = f"Asuna and the user continued their adventure. No detailed summary available."

    # Inject the summary into the global story bible
    global STORY_BIBLE
    STORY_BIBLE += f"\n\n[Chapter: {arc_name}]\n{chapter_summary}"

    # Embed the raw history into ChromaDB with arc metadata
    print("[Compression] Archiving raw logs to ChromaDB...")
    # Chunk the raw history into sliding-window chunks
    chunks = chunk_messages(history, window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP)
    col = get_or_create_collection(client)
    for idx, chunk in enumerate(chunks):
        emb = ollama.embed(model=EMBED_MODEL, input=chunk)['embedding']
        meta = {"arc": arc_name, "chunk_id": str(idx)}
        col.add(embeddings=[emb], documents=[chunk], metadatas=[meta],
                ids=[f"{arc_name}_{idx}_{int(time.time())}"])
    print(f"[Compression] Archived {len(chunks)} chunks under arc '{arc_name}'.")

    # Clear the active history (start fresh)
    return [], chapter_summary

# -----------------------------------------------------------------------------
# Main interactive loop
# -----------------------------------------------------------------------------
def main():
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = get_or_create_collection(chroma_client)

    if col.count() == 0:
        print("The RAG database is empty.")
        ingest_choice = input("Would you like to ingest a JSON chat export? (y/n): ").strip().lower()
        if ingest_choice == 'y':
            path = input("Path to JSON file: ").strip()
            ingest_json_to_chroma(path, chroma_client)
        else:
            print("Running without RAG (Type '/ingest <path>' later to add history).")

    history = []  # list of {"role": "user"/"assistant", "content": "..."}
    global STORY_BIBLE

    print("\n=== VoiceAI-Asuna (Updated) Terminal Prototype ===")
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
            break

        if user_input.lower() == '/exit':
            break
        elif user_input.lower() == '/reset':
            history.clear()
            print("Conversation history cleared. (Story memory remains.)")
            continue
        elif user_input.lower().startswith('/ingest'):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                ingest_json_to_chroma(parts[1], chroma_client)
            else:
                print("Usage: /ingest <filepath>")
            continue
        elif user_input.lower() == '/end_chapter':
            arc_name = input("Enter a name for this chapter (e.g. 'dark_forest_escape'): ").strip()
            if not arc_name:
                arc_name = "unnamed_chapter"
            history, _ = compress_chapter(history, chroma_client, arc_name)
            print(f"Chapter '{arc_name}' compressed. Story memory updated, history cleared.\n")
            continue
        elif user_input.lower() == '/recap':
            print("\n--- STORY BIBLE ---")
            print(STORY_BIBLE)
            print("--------------------\n")
            continue

        # ---- Phase 1: Director decides RAG ----
        print("[Phase 1: Director analysing...]")
        phase1_prompt = (
            f"User message: \"{user_input}\"\n\n"
            "Determine if external context (past conversations or world facts) is needed to answer accurately and in character. "
            "Return ONLY valid JSON: {\"needs_rag\": true/false, \"search_query\": \"string\"}"
        )
        try:
            dir1 = run_director(phase1_prompt)
            needs_rag = dir1.get('needs_rag', False)
            search_query = dir1.get('search_query', user_input)
        except Exception as e:
            print(f"Director Phase 1 error: {e}. Assuming no RAG.")
            needs_rag = False
            search_query = ""

        rag_context = ""
        if needs_rag:
            print(f"[RAG query: '{search_query}']")
            rag_context = query_rag(search_query, chroma_client, top_k=RAG_TOP_K)
            if rag_context:
                # Trim rag_context if it's too long (max ~8000 tokens)
                if count_tokens(rag_context) > 8000:
                    rag_context = rag_context[:8000*4]  # rough character cut, then trim properly
                print(f"[Retrieved {len(rag_context.split())} words of context]")
            else:
                print("[RAG returned empty]")

        # ---- Phase 2: Main Brain generates story ----
        print("[Phase 2: Main Brain generating...]")

        # Trim active history to token budget
        trimmed_history = trim_history(history, HISTORY_MAX_TOKENS)
        history_str = "\n".join(f"{h['role']}: {h['content']}" for h in trimmed_history)

        # Assemble final prompt
        system_part = f"<SYS>\n{STORY_BIBLE}\n</SYS>"
        context_part = f"[Relevant remembered context]\n{rag_context}" if rag_context else ""
        prompt_parts = [system_part]
        if context_part:
            prompt_parts.append(context_part)
        if history_str:
            prompt_parts.append(f"[Previous conversation]\n{history_str}")
        prompt_parts.append(f"User: {user_input}\nAsuna:")

        full_prompt = "\n\n".join(prompt_parts)

        # Safety check: if prompt exceeds 64k tokens, further trim
        if count_tokens(full_prompt) > 62000:
            print("[Warning] Prompt too long, aggressively trimming history...")
            # Reduce trimmed_history drastically
            while trimmed_history and count_tokens(full_prompt) > 60000:
                if len(trimmed_history) >= 2 and trimmed_history[0]['role'] == 'user' and trimmed_history[1]['role'] == 'assistant':
                    trimmed_history = trimmed_history[2:]
                else:
                    trimmed_history.pop(0)
                history_str = "\n".join(f"{h['role']}: {h['content']}" for h in trimmed_history)
                prompt_parts[-2] = f"[Previous conversation]\n{history_str}" if history_str else ""
                full_prompt = "\n\n".join([p for p in prompt_parts if p])

        try:
            story_output = run_main_brain(full_prompt)
        except Exception as e:
            print(f"Main Brain error: {e}")
            continue

        # ---- Phase 3: Director extracts expression ----
        print("[Phase 3: Director extracting expression...]")
        phase3_prompt = (
            f"The character Asuna just said:\n\"{story_output[:800]}\"\n\n"
            "Which emotion does this line convey? Choose exactly one from: "
            "happy, sad, angry, surprised, neutral, blushing, worried, determined, playful, scared. "
            "Return ONLY valid JSON: {\"expression\": \"chosen_emotion\"}"
        )
        try:
            dir3 = run_director(phase3_prompt)
            expression = dir3.get('expression', 'neutral')
        except Exception as e:
            print(f"Director Phase 3 error: {e}. Defaulting to 'neutral'.")
            expression = "neutral"

        # ---- Display result ----
        print(f"\nAsuna [{expression}]: {story_output}\n")

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": f"[{expression}] {story_output}"})

if __name__ == "__main__":
    main()
```

---

### What’s new in this prototype

1. **Expanded RAG** – Retrieves top **10** chunks (instead of 3) and trims them to ~8000 tokens if needed, giving the Main Brain much deeper lore.

2. **Smart history trimming** – Uses a GPT‑2 tokenizer (or word‑based fallback) to keep the active conversation under **~28,000 tokens**, well within the 64k context.

3. **Hierarchical memory compression (`/end_chapter`)**  
   - Prompts for an arc name (e.g. `dark_forest_escape`).  
   - The Director generates a **300‑word narrative summary** of the recent conversation.  
   - That summary is permanently added to the **story bible**, so Asuna “remembers” the plot forever.  
   - The raw chat logs are chunked, embedded, and stored in ChromaDB **with the arc metadata** – exactly as specified.  
   - Active conversation history is then cleared, starting a fresh chapter.

4. **Better JSON handling** – Robust extraction of JSON even if the Director wraps it in markdown fences.

5. **Stop tokens** – The Main Brain respects `User:` as a stop token (optional), helping to prevent it from writing the user’s next line.

6. **Model suggestions integrated** – The script still uses `Llama-3.1-8B-Stheno-v3.4:IQ4_NL` as the Main Brain, but you can easily swap to a 12B Mistral Nemo if you’re feeling adventurous (adjust `MAIN_MODEL` variable).

---

### Running the updated prototype

```bash
pip install ollama chromadb tiktoken   # tiktoken optional but recommended
python asuna_prototype.py
```

Now you have a terminal backend that fully respects the expanded 64k window, proactive RAG, and the intelligent memory compression system – ready to be wired into your Live2D frontend later.