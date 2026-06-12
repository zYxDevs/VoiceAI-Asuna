# Product Requirements Document (PRD): VoiceAI-Asuna Hot-Swap Architecture

## 1. Project Overview
**Name:** VoiceAI-Asuna (Local Backend)
**Description:** A fully local, uncensored, retrieval-augmented generation (RAG) backend designed to power a Live2D avatar (Asuna). The system uses a dual-agent architecture (Director + Main Brain) that hot-swaps models in and out of VRAM to maintain a massive 64k context window while strictly adhering to a 12GB hardware limit.
**Primary Use Case:** Continuous, long-form roleplay and novel writing (e.g., Asuna in the Underworld) with real-time semantic memory and dynamic Live2D expression triggering.

## 2. Hardware Constraints & Environment
**Target Hardware:**
* **GPU:** Nvidia RTX 3060 (Strict 12GB VRAM limit).
* **Storage:** Gen4 NVMe SSD (Critical for fast model hot-swapping).
* **OS Overhead:** Windows Desktop Window Manager (DWM) reserves ~2.0GB VRAM.
* **Usable VRAM Target:** ~10.0GB maximum sustained load.

**Required Environment Variables (Ollama):**
* `OLLAMA_FLASH_ATTENTION=1` (Reduces memory footprint and speeds up generation).
* `OLLAMA_KV_CACHE_TYPE=q8_0` (Quantizes context memory, allowing 64k context to fit within ~3.0GB VRAM).

## 3. Model Architecture
The system relies on two specialized models running sequentially. They **must never** be loaded into VRAM simultaneously.

### Agent 1: The "Main Brain" (Story & Persona)
* **Model:** `Llama-3.1-8B-Stheno-v3.4` (Abliterated/Uncensored).
* **Source URL:** https://huggingface.co/Lewdiculous/Llama-3.1-8B-Stheno-v3.4-GGUF-IQ-Imatrix
* **Quantization:** `IQ4_NL` or `Q4_K_M` (~4.9GB file size).
* **Context Window (`num_ctx`):** 65,536 (64k tokens).
* **Output Limits (`num_predict`):** `-1` (Infinite, until context fills or natural stop).
* **Role:** Maintains the persona, generates high-quality narrative prose, and acts as the conversational partner.

### Agent 2: The "Micro-Director" (Logic & Routing)
* **Model:** `Llama-3.2-3B-Instruct-abliterated` (Primary) or `Llama-3.2-1B-Instruct-abliterated` (Fallback).
* **Source URLs:** 
    * 3B Version: https://huggingface.co/mradermacher/Llama-3.2-3B-Instruct-abliterated-GGUF
    * 1B Version: https://huggingface.co/mradermacher/Llama-3.2-1B-Instruct-abliterated-GGUF
* **Quantization:** `Q4_K_M` (~2.24GB) or `Q8_0` (~3.84GB) for the 3B. 
* **Role:** Analyzes user input for RAG triggers, and parses Main Brain output for Live2D expressions. Fast, strict JSON adherence.

## 4. The 3-Phase Hot-Swap Execution Pipeline
To prevent VRAM spillover (crashing into system DDR4), the Python orchestrator uses Ollama's API with `keep_alive: 0` for **every** request. This forces the Gen4 SSD to swap models in ~2 seconds per transition. This load time is masked on the frontend by a Live2D "thinking" animation.

* **Phase 1: Pre-Run (RAG Trigger)**
    * **Action:** User submits input. Python loads the Director (3B).
    * **Prompt:** Director evaluates if external knowledge is needed.
    * **Output:** JSON `{"needs_rag": boolean, "search_query": "string"}`.
    * **Cleanup:** `keep_alive: 0` forces Director out of VRAM.
    * **Post-Action:** If true, Python queries ChromaDB and prepares context.

* **Phase 2: Generation (Main Story)**
    * **Action:** Python loads the Main Brain (8B Stheno).
    * **Prompt:** RAG context (if any) + System Prompt (Story Bible) + User Input.
    * **Output:** Uncensored narrative text.
    * **Cleanup:** `keep_alive: 0` forces Main Brain out of VRAM.

* **Phase 3: Post-Run (Live2D Trigger)**
    * **Action:** Python reloads the Director (3B).
    * **Prompt:** Director reads the Phase 2 story output.
    * **Output:** JSON `{"expression": "happy/angry/sad/etc"}`.
    * **Cleanup:** `keep_alive: 0` forces Director out of VRAM.
    * **Post-Action:** Python passes text to UI and triggers Live2D CSS/JS state.

## 5. Memory & Context Strategy (Scaled for 64k)
With the VRAM pipeline secured, the 64,000 token space is utilized to maximize intelligence without hitting memory limits.

* **Active Chat History:** Retains the last 20,000 to 30,000 words natively in the prompt context.
* **Expanded RAG Injection:** The Phase 1 Director is configured to pull the top 10-15 semantic database hits (~5,000+ words of deep lore/world-building) directly into the prompt on relevant queries.
* **Semantic Chunking:** Data in ChromaDB is stored as sliding-window conversational pairs (3-5 messages per block) with overlapping edges to prevent context loss. Metadata tags (e.g., `{"arc": "underworld_arrival"}`) are attached for precise Director filtering.

## 6. Future Implementation: Hierarchical Memory (Down-Time Compression)
To prevent infinite context growth and maintain long-term story coherence, the system will implement tiered memory management during system idle periods.

* **The Trigger:** Initiated either manually via a UI "End of Chapter" button or automatically after a set period of conversational idle time.
* **The Compression Run:** The backend wakes the 3B Director to analyze the raw logs of the completed session.
* **Core Memory Update (Summary):** The Director generates a dense, 300-word narrative summary of the events. This summary is permanently injected into the Main Brain's persistent System Prompt, giving Asuna continuous "general awareness" of the plot.
* **Specific Recall Archive (ChromaDB):** The raw, full-text conversation logs are simultaneously embedded and archived into the ChromaDB vector store, allowing the Phase 1 Director to fetch exact quotes later.

## 7. Frontend / UI Integration
* The frontend remains a custom CSS/JS UI prioritizing exact Live2D absolute positioning.
* The Python backend serves as middleware via WebSocket or local HTTP server to the UI.
* Frontend displays a "Thinking..." state (2-5 seconds) while Phase 1 and 2 execute, masking the larger context ingestion and SSD read times gracefully.