# litert-lm-rust

Safe, idiomatic Rust bindings for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) — Google's on-device LLM runtime.

This crate wraps the stable **C API** (`c/engine.h`), which is the supported FFI surface for the C++ Conversation / Engine stack described in [`litert.md`](./litert.md).

## Features

| Feature | Support |
| --- | --- |
| Conversation API (`send_message` / streaming) | ✅ |
| Multimodal (vision + audio backends, image/audio parts) | ✅ |
| **MTP** (Multi-Token Prediction / speculative decoding) | ✅ |
| Session API (prefill / decode / generate_content) | ✅ |
| Tools, thinking, constrained decoding, LoRA, benchmarks | ✅ |

## Install

```toml
[dependencies]
litert-lm-rust = "0.2.2"
```

The crate will automatically download the required native libraries from GitHub releases on first build (Windows only). No manual setup is required for most use cases.

## Quick start

```rust
use litert_lm_rust::{Backend, Engine, Message};

fn main() -> litert_lm_rust::Result<()> {
    let engine = Engine::builder("model.litertlm")
        .backend(Backend::Cpu)
        .build()?;
    let mut conversation = engine.create_conversation(Default::default())?;
    let reply = conversation.send_message(Message::user("Hello!"))?;
    println!("{reply}");
    Ok(())
}
```

### Multimodal

```rust
use litert_lm_rust::{Backend, ContentPart, Engine, Message};

let engine = Engine::builder("gemma.litertlm")
    .backend(Backend::Cpu)
    .vision_backend(Backend::Gpu)
    .audio_backend(Backend::Cpu)
    .build()?;

let mut conversation = engine.create_conversation(Default::default())?;
let reply = conversation.send_message(Message::user_parts([
    ContentPart::text("Describe this image:"),
    ContentPart::image_path("photo.jpg"),
])?)?;
```

### MTP (Multi-Token Prediction)

```rust
let engine = Engine::builder("model.litertlm")
    .backend(Backend::Gpu)
    .multi_token_prediction(true) // enable_speculative_decoding
    .build()?;
```

## Native library

### Automatic Download (Windows - Recommended)

The crate includes a `download-native` feature (enabled by default) that automatically downloads the required native libraries from GitHub releases on first build. The native libraries are built from the LiteRT-LM source using a GitHub Actions workflow and published as release assets.

**Downloaded files:**
- `litert-lm.dll` - Main LiteRT-LM library
- `litert-lm.if.lib` - Import library for linking
- `libLiteRt.dll` - LiteRT runtime library
- `libGemmaModelConstraintProvider.dll` - Gemma constraint provider
- `libLiteRtTopKWebGpuSampler.dll` - WebGPU sampler
- `libLiteRtWebGpuAccelerator.dll` - WebGPU accelerator
- `libwebgpu_dawn.dll` - Dawn WebGPU implementation

These libraries are automatically downloaded to the crate's `prebuilt/` directory during the first build. No manual setup is required.

### Using in another project

If you use this crate as a **path dependency** in another workspace, copy the DLLs and `.if.lib` from the `prebuilt/` folder into your local copy of the crate. The `build.rs` looks for them in these directories (in order):

1. The `LITERT_LM_LIB_DIR` environment variable
2. `<build_out>/prebuilt/` (auto-downloaded location)
3. `<crate_dir>/prebuilt/` — **easiest: just copy files here**
4. `<crate_dir>/native/`
5. `<crate_dir>/c/`
6. `<crate_dir>/c/build/`

At runtime the DLLs must also be on `PATH` or next to the executable.

### Manual Setup

If you prefer to build the LiteRT-LM C API from source:

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM
cd LiteRT-LM
git checkout <release-tag>   # e.g. v0.14.0
bazel build //c:engine
```

Then point this crate at the output:

```bash
# Linux
export LITERT_LM_LIB_DIR=$PWD/bazel-bin/c
export LITERT_LM_LIB_NAME=engine

# Windows (PowerShell)
$env:LITERT_LM_LIB_DIR = "C:\path\to\bazel-bin\c"
$env:LITERT_LM_LIB_NAME = "engine"
```

Alternatively, place the shared library in this crate's `c/`, `native/`, or `prebuilt/` folder.

| Variable | Meaning |
| --- | --- |
| `LITERT_LM_LIB_DIR` | Directory containing the shared library |
| `LITERT_LM_LIB_NAME` | Library name without `lib` / extension (default tries `LiteRtLmC` / `engine`) |
| `LITERT_LM_STATIC` | If set, link statically |
| `LITERT_LM_INCLUDE_DIR` | Override header path (defaults to vendored `c/`) |

**Note:** To disable automatic download and use your own libraries, add `default-features = false` to your Cargo.toml:
```toml
[dependencies]
litert-lm-rust = { version = "0.13", default-features = false }
```

### Legacy pre‑built Windows binaries

If you have the pre‑built LiteRT‑LM binaries (from the `LiteRT-LM/prebuilt/windows_x86_64` folder of the `litert‑lm‑rust` repository) you can avoid building from source.

1. **Place the folder** somewhere accessible, e.g. as a sibling of your project:
   ```
   <workspace_root>/litert-lm-rust/LiteRT-LM/prebuilt/windows_x86_64
   ```
   This directory should contain `LiteRt.dll` (or `libLiteRt.dll`) and the generated import library `LiteRt.lib` (or `libLiteRt.lib`). If you only have the DLL, generate the import library with:
   ```sh
   llvm-objdump -p libLiteRt.dll > symbols.txt
   llvm-dlltool --def=symbols.txt --output-lib=LiteRt.lib
   ```
2. **Linking**: The crate’s `build.rs` automatically locates this directory on Windows (it walks up from `CARGO_MANIFEST_DIR` to find the sibling `litert‑lm‑rust` folder). No additional environment variables are required unless you want to override the location. To override, set:
   ```powershell
   $env:LITERT_LM_LIB_DIR = "C:\\path\\to\\windows_x86_64"
   $env:LITERT_LM_LIB_NAME = "LiteRt"   # or "libLiteRt"
   ```
3. **What the build script does**:
   - Adds the directory to the linker search path.
   - Links `LiteRt` as a dynamic library.
   - Copies the runtime DLL(s) next to the final executable so they are found at runtime.

After these steps, `cargo build` (or `cargo tauri build`) should succeed without the `LINK : fatal error LNK1181: cannot open input file 'LiteRt.lib'` error.

The `c/` directory also vendors the LiteRT C/C++ SDK headers used when building LiteRT-LM itself.

## Examples

```bash
cargo run --example text_chat -- model.litertlm "Hello"
cargo run --example multimodal -- model.litertlm photo.jpg
cargo run --example mtp -- model.litertlm
```

## Regenerating bindings

```bash
cargo build --features bindgen,docs-only
```

This updates `src/bindings.rs` from `c/wrapper.h` / `c/engine.h`.

## Publish

```bash
cargo publish --dry-run
# requires network + crates.io token for a real publish:
# cargo publish
```

Docs builds use `--features docs-only` so they do not need a local native library.

## License

Apache-2.0. LiteRT-LM headers and runtime are © The ODML Authors / Google LLC under Apache-2.0.
