# Building, publishing, and using Noema crates

Noema is a Cargo workspace of independent crates. This guide covers three
things: building the workspace, publishing the crates to crates.io, and
consuming them from your own projects.

---

## 1. Building

### Requirements

- **Rust** (stable, MSRV 1.75). `rustup` works on Windows / macOS / Linux.
- **Windows or macOS.** The two local engines ship prebuilt binaries:
  - **LiteRT-LM** (Gemma 4): Windows DLLs in `prebuilt/` (staged next to
    executables by `crates/noema-native`); macOS dylibs in `prebuilt/macos/`
    (rpath-embedded, no staging needed). The macOS C API library
    (`litert-lm.dylib`) comes from Google's LiteRT-LM v0.16.0+ release.
  - **Needle 2**: `noema-needle` loads the engine at runtime (`DylibEngine`);
    `noema-needle-static` links `libneedle.a` at build time (`StaticEngine`).
    macOS arm64 uses the static path — place `libneedle.a` + `needle.h` from
    Cactus Compute's HuggingFace repo in `prebuilt/needle/macos-arm64/`.
- The Gemma model file lives in `models/gemma-4-E2B-it.litertlm`
  (overridable with `NOEMA_GEMMA_MODEL`).

### Build everything

```sh
cargo build --workspace          # all crates + examples
cargo build -p noema-api         # just the frontend-facing API crate
cargo test --workspace           # all unit tests (fast)
cargo test --workspace -- --ignored   # real-inference tests (needs the engines + model)
```

### The crate map

| Crate | What it is | Publish? |
| --- | --- | --- |
| `noema-events` | Event vocabulary + streaming bus | ✅ pure Rust |
| `noema-tools` | Tool traits, schemas, risk, registry | ✅ pure Rust |
| `noema-approval` | Human-approval lifecycle | ✅ pure Rust |
| `noema-core` | The runtime, sessions, agent loop, config | ✅ pure Rust |
| `noema-api` | The single frontend-facing crate (re-exports) | ✅ pure Rust |
| `noema-provider-http` | OpenAI-compatible cloud provider | ✅ pure Rust |
| `noema-filesearch` | Reference tool crate | ✅ pure Rust |
| `noema-rig` | Rig adapters | ✅ pure Rust (depends on `rig-core` from crates.io) |
| `noema-gemma` | Gemma 4 adapter | ⚠️ needs `litert-lm-rust` + model artifacts at runtime |
| `noema-needle` | Needle 2 FFI binding | ⚠️ needs the prebuilt engine at runtime |
| `noema-needle-static` | Needle 2 static link | ⚠️ needs `libneedle.a` at build time |
| `noema-router` | Needle router + tool formatters | ⚠️ depends on `noema-needle` |
| `noema-native` | Stages LiteRT DLLs next to executables | ⚠️ Windows-specific |
| `litert-lm-rust` | Vendored LiteRT-LM binding | ⚠️ build script downloads binaries |

---

## 2. Publishing to crates.io

All workspace dependencies already carry `version = "0.1.0"` alongside
their `path`, so `cargo publish` can resolve them once the dependencies are
on crates.io.

### Steps

1. **Publish bottom-up.** crates.io requires a crate's dependencies to be
   published first:

   ```sh
   cargo publish -p noema-events
   cargo publish -p noema-tools
   cargo publish -p noema-approval
   cargo publish -p noema-context
   cargo publish -p noema-core
   cargo publish -p noema-provider-http
   cargo publish -p noema-api
   cargo publish -p noema-filesearch
   ```

   `noema-rig` can follow (it also depends on the crates.io `rig-core`).
   `noema-gemma` / `noema-needle` / `noema-needle-static` /
   `noema-router` / `noema-native` / `litert-lm-rust` are publishable but
   carry native-runtime caveats (see below) — publish them after
   `noema-core` if you want them available.

2. **Authenticate once:** `cargo login <API-token>` (token from
   <https://crates.io/settings/tokens>).

3. **Before each publish**, make sure:
   - The version in `Cargo.toml` (`version.workspace = true` → `0.1.0`)
     is not already taken on crates.io. Bump it (and the workspace
     `version`) for subsequent releases.
   - `cargo publish --dry-run -p <crate>` passes (it packages without
     uploading).

### Caveats

- **Native artifacts are not uploaded.** crates.io only accepts source.
  `noema-gemma` and `noema-needle` need their binaries at *runtime*; consumers
  must supply `prebuilt/` (LiteRT DLLs, Needle engine) and the model files.
  `noema-needle-static` needs `libneedle.a` at *build time* — the symbols
  are baked into the final binary.
  Document this in the READMEs, and fail with a clear error at runtime (the
  adapters already do: `NOEMA_GEMMA_MODEL` / `NEEDLE_LIB_PATH`).
- **`litert-lm-rust`**'s `build.rs` tries to download native libraries when
  they are missing. When publishing, prefer keeping the DLLs in-repo or
  shipping a `noema-native`-style staging crate.
- Never publish secrets: check for API keys or model paths in `Cargo.toml`
  and `build.rs` before running `cargo publish`.

---

## 3. Using the crates in your own project

### 3.1 The quick path: `noema-api`

For a frontend or application that just wants to talk to Noema, depend on
the single public crate:

```toml
[dependencies]
noema-api = "0.1"
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
```

```rust
use noema_api::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Register a model (any implementation of `Model`).
    let noema = Noema::builder()
        .with_model(MyModel)          // GemmaModel, a cloud adapter, a mock…
        .with_tool(MyTool)            // any NoemaTool implementation
        .build()
        .await?;

    // 2. Create a session and send messages.
    let session = noema.create_session().await?;
    let mut events = session.events();
    let outcome = session.send(Message::text(Role::User, "hello")).await?;

    // 3. Drive approvals and tools through the same API.
    for pending in session.pending_approvals() {
        session.approve_tool(pending.id.clone())?;
    }
    Ok(())
}
```

`noema-api` re-exports everything you need: `Noema`, `NoemaBuilder`,
`Session`, `Message` / `ContentPart` (text/image/audio), `Event`,
`NoemaTool` / `ToolRegistry`, `ApprovalPolicy`, `ModelProvider`,
`OpenAICompatibleProvider`, `MetricsSnapshot`, and the config types.

### 3.2 Local models (Gemma / Needle)

Add the model crates and provide the artifacts:

```toml
[dependencies]
noema-gemma = "0.1"     # needs the LiteRT-LM DLLs + model file at runtime
noema-router = "0.1"    # need the prebuilt Needle engine at runtime
noema-filesearch = "0.1"
```

```rust
use noema_api::prelude::*;
use noema_gemma::GemmaModel;
use noema_router::{NeedleRouter, NeedleToolFormatter};

let mut registry = ToolRegistry::new();
registry.register(noema_filesearch::Filesearch::default())?;
let schema = registry.get(noema_filesearch::TOOL_NAME).unwrap().schema();

let noema = Noema::builder()
    .with_model(GemmaModel::from_default()?)          // NOEMA_GEMMA_MODEL or models/
    .with_router(NeedleRouter::from_default()?)       // NEEDLE_LIB_PATH or prebuilt/needle/
    .with_tools(registry.clone())
    .with_tool_formatter_for(
        noema_filesearch::TOOL_NAME,
        NeedleToolFormatter::from_tool(&schema, None)?,
    )
    .build()
    .await?;
```

### 3.3 Cloud escalation

```rust
use noema_api::prelude::*;

let noema = Noema::builder()
    .with_model(LocalModel)
    .with_provider(OpenAICompatibleProvider::new(
        "openai",
        "gemini-2.5-pro",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        std::env::var("GEMINI_API_KEY").ok(),
    ))
    .with_escalation_policy(EscalationPolicy {
        allow_local: false,
        allow_cloud: true,
        preferred_provider: Some("openai".into()),
        ..EscalationPolicy::default()
    })
    .build()
    .await?;
```

### 3.4 Writing your own tool

See `docs/usage.md` §6 ("Implementing your own tool"): create a crate
depending on `noema-tools`, implement `NoemaTool` (metadata, schema, risk,
`execute`), and register it. No core changes needed.

### 3.5 Using the crates by path (no publish)

While developing, you can depend on the crates directly from this
repository:

```toml
[dependencies]
noema-api = { path = "../noema/crates/noema-api" }
```

Cargo resolves the other `noema-*` workspace crates automatically.
