# AI Log

I used Codex (GPT-5 family) to scaffold and review the Python service, reason about the SQLite reservation boundary, and draft the documentation; I checked official Groq and Gemini API references for endpoints, authentication, model IDs, and token fields.

Codex first blamed a container startup failure on `uvicorn` missing from `PATH`. I inspected `requirements.txt`, found it empty, restored the dependencies, and rebuilt without cache.

I overrode Codex's original `gemini-2.5-flash-lite` default after a direct provider call returned `404`. I replaced it with the current `gemini-3.1-flash-lite` model and reran the forced-fallback test successfully.