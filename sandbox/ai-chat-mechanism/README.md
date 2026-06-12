# AI chat mechanism (sandbox)

Isolated terminal prototype for the hot-swap pipeline (Director → Main Brain → Director). It does **not** import the main VoiceAI-Asuna app; use it for experiments only.

**Docs:** [`Docs/PRD.md`](Docs/PRD.md), [`Docs/Prototype_Instructions.md`](Docs/Prototype_Instructions.md).

## Prerequisites

- Python 3.10+ recommended  
- [Ollama](https://ollama.com/) installed and running  

## Setup

```bash
cd sandbox/ai-chat-mechanism
pip install -r requirements.txt
```

Pull the default models (first run downloads several GB):

```bash
ollama pull nomic-embed-text
ollama pull huihui_ai/llama3.2-abliterate:3b-instruct-q4_K_M
ollama pull hf.co/mradermacher/L3-8B-Stheno-v3.2-abliterated-GGUF:Q4_K_M
```

Optional (matches PRD notes on VRAM / KV cache):

```bash
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
```

(On PowerShell, use `$env:VAR="value"`.)

## Run

```bash
python asuna_prototype.py
```

Interactive commands: `/exit`, `/reset`, `/ingest <path>`, `/end_chapter`, `/recap`.

## Environment overrides

| Variable | Purpose |
|----------|---------|
| `ASUNA_DIRECTOR_MODEL` | Micro-director (JSON routing); default is `huihui_ai/llama3.2-abliterate:3b-instruct-q4_K_M`. |
| `ASUNA_MAIN_MODEL` | Story model; default is `hf.co/mradermacher/L3-8B-Stheno-v3.2-abliterated-GGUF:Q4_K_M`. |
| `ASUNA_EMBED_MODEL` | Embeddings for ChromaDB / RAG; default `nomic-embed-text`. |
| `ASUNA_MAIN_NUM_PREDICT` | Max new tokens from the main model (`-1` = unlimited per PRD). Use e.g. `256` for short smoke tests. |
| `ASUNA_QUIET` | Set to `1` to hide verbose trace (full prompts, director raw output, token breakdown, RAG dump). Default is **verbose** for this prototype. |

## Smoke test (piped)

```powershell
$env:ASUNA_MAIN_NUM_PREDICT="256"
"n`nHi.`n/exit`n" | python asuna_prototype.py
```
