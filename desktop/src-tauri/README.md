# Bani-Mic desktop shell

Tauri 2.x wrapper around the Python sidecar (`../sidecar/dist/bani-mic`).

## How it works

```
User double-clicks Bani Mic.app
   │
   ▼
Tauri Rust shell (this crate) starts
   │  ├─→ Opens window, loads frontend-stub/stub.html (splash)
   │  └─→ Spawns binaries/bani-mic-<triple> (the PyInstaller bundle)
   │       │
   │       ▼
   │     Sidecar warms ASR + corpus, prints `BANI_READY port=<n>` on stdout
   │
   ▼
Shell parses port=<n>, navigates the WebView to http://127.0.0.1:<n>/mic/
   │
   ▼
Same UX as the hosted version but native ONNX Runtime under the hood
```

## Dev build

```sh
# 1. build the sidecar bundle (once, or after touching scripts/sidecar/)
cd ../sidecar && pyinstaller --clean --noconfirm sidecar.spec

# 2. ensure the symlink for Tauri's externalBin is current
cd ../src-tauri/binaries
ln -sf ../../sidecar/dist/bani-mic bani-mic-$(rustc -vV | sed -n 's/host: //p')

# 3. run the shell
cd ..
cargo tauri dev
```

The dev cycle does NOT rebuild the sidecar. If you change anything in
`../sidecar/server.py` or `../../scripts/`, re-run pyinstaller in step 1.

## Production build

```sh
cargo tauri build
```

Produces `src-tauri/target/release/bundle/macos/Bani Mic.app/` (and a
`.dmg` if you have `cargo-bundle`'s deps available). The .app embeds
the `bani-mic` sidecar so it's a single drag-and-drop install for the
end user.

## File map

```
src-tauri/
  Cargo.toml             # Rust deps: tauri 2, tauri-plugin-shell
  build.rs               # tauri_build::build() — generates schemas
  tauri.conf.json        # window size, identifier, externalBin
  capabilities/default.json   # permission set for the main window
  src/main.rs            # spawn sidecar, parse port, navigate WebView
  binaries/
    bani-mic-<triple>    # symlink → ../../sidecar/dist/bani-mic
  icons/                 # generated; see `cargo tauri icon`

frontend-stub/
  stub.html              # splash page shown for ~1-2s while sidecar warms
```

## Why a stub frontend?

Tauri requires *some* frontendDist or devUrl to exist. Our "frontend"
lives in the Python sidecar at `http://127.0.0.1:<port>/mic/`, but the
WebView can't show that until the sidecar is ready. The stub is a
zero-dependency HTML/CSS-only splash that fills the window instantly
on launch; the Rust shell swaps it for the live URL as soon as the
sidecar prints `BANI_READY`.

## Updating the sidecar binary

```sh
cd ../sidecar && pyinstaller --clean --noconfirm sidecar.spec
# symlink already points at dist/, so cargo tauri picks up the new build
cd ../src-tauri && cargo tauri dev
```
