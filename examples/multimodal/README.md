# Examples: multimodal

The Phase 12 demo: text, image, and audio travel as one ordered message
through Gemma 4's native modalities, and multimodal reasoning can drive
tool calls inside the agent loop.

```sh
cargo run -p multimodal-example
```

Shows, on the real engine:

1. **Mixed text/image** — a question plus `red.png` (the E2B checkpoint
   answers directly).
2. **Mixed text/audio** — a question plus `tone.wav` (accepted and declined
   gracefully: the checkpoint has no audio channel; a future audio-capable
   checkpoint answers directly with no code changes).
3. **Image turn → tool workflow** — image reasoning leads to a filesearch
   call inside one `session.send` (best-effort: the small checkpoint is not
   reliably agentic, so the loop answers directly when it does not name the
   tool).

Needs `models/gemma-4-E2B-it.litertlm` (or `NOEMA_GEMMA_MODEL`) and the
Needle engine (`prebuilt/needle/`).
