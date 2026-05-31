# Desktop app

Tauri shell that wraps the Python engine as a sidecar and loads the same `/mic/` UI as the hosted web build.

```
Tauri shell (Rust)
  └─ spawns gurbani-captioning sidecar (PyInstaller bundle of engine/)
       └─ aiohttp server on localhost:<random port>
            └─ WebView loads http://localhost:<port>/mic/
                 └─ WebSocket to /ws for real-time events
```

Pre-built downloads: <https://github.com/karanbirsingh/kirtan-captioning/releases>

## Build from source (macOS / Linux)

Requires Rust + Python 3.11+.

```bash
# 1. Build the Python sidecar (one-file, ~200 MB)
cd desktop/sidecar
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --clean --noconfirm sidecar.spec

# 2. Stage the sidecar binary for Tauri (target-triple suffix is required)
cp dist/gurbani-captioning \
   ../src-tauri/binaries/gurbani-captioning-$(rustc -vV | sed -n 's/host: //p')

# 3. Build the Tauri app
cd ../src-tauri
cargo tauri build
```

Output: `desktop/src-tauri/target/release/bundle/`:
- macOS: `dmg/Gurbani Captioning_<version>_aarch64.dmg`
- Linux: `appimage/...`, `deb/...`

## Windows

Same flow as macOS/Linux, with three differences:
- Use `onnxruntime-directml` instead of `onnxruntime` in the sidecar (already gated in `desktop/sidecar/requirements.txt` by `sys_platform`). Lights up GPU acceleration on any DX12 GPU including iGPUs.
- The Tauri bundler emits an NSIS installer (`.exe`) and an MSI. Both are produced by `cargo tauri build`.
- The sidecar binary suffix in `binaries/` is `gurbani-captioning-x86_64-pc-windows-msvc.exe` — copy from `dist/gurbani-captioning.exe` produced by PyInstaller. Run `rustc -vV | findstr host` in PowerShell to confirm the triple.

Prerequisites (once): Rust with the **MSVC** toolchain, **Python 3.11+**, **Visual Studio Build Tools** (for `link.exe`), the **WebView2** runtime (preinstalled on Win11), and the Tauri CLI (`cargo install tauri-cli`).

```powershell
# 1. Build the Python sidecar (pulls onnxruntime-directml on Windows)
cd desktop\sidecar
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --clean --noconfirm sidecar.spec

# 2. Stage the sidecar with the MSVC target-triple suffix
#    (confirm the triple with: rustc -vV | findstr host)
copy dist\gurbani-captioning.exe ..\src-tauri\binaries\gurbani-captioning-x86_64-pc-windows-msvc.exe

# 3. Build the Tauri app
cd ..\src-tauri
cargo tauri build
# -> target\release\bundle\nsis\Gurbani Captioning_<version>_x64-setup.exe
```

No code signing yet on Windows either — Defender SmartScreen will show an "unrecognized app" warning on first install. Fixed by an EV code-signing cert (~$200/yr), deferred.

## Architecture notes

- **Sidecar lifecycle**: Rust shell spawns the sidecar, watches stdout for `BANI_READY port=<n>`, then navigates the WebView. On window close (or app quit, or shell crash), the sidecar self-exits within 2s via the `BANI_SHELL_PID` watchdog. See `src-tauri/src/main.rs`.
- **Code signing (macOS)**: the released `.dmg` is signed with Developer ID Application (KARANBIR SINGH, KF66U685PK), notarized, and stapled — Apple notary `Accepted` and `spctl` reports `source=Notarized Developer ID`, so it launches with no "damaged"/quarantine warning (even offline). Windows builds remain unsigned (see above).
- **Logs**: `~/Library/Logs/dev.gurbani.captioning/Gurbani Captioning.log` (macOS) or `%LOCALAPPDATA%\dev.gurbani.captioning\logs\` (Windows). Captures both Rust shell + sidecar stdout/stderr.
