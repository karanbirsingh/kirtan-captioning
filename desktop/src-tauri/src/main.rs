// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Gurbani Captioning desktop shell entry point.
//!
//! Boots the Python sidecar (PyInstaller-bundled `gurbani-captioning`), waits for it
//! to print `BANI_READY port=<n>` on stdout, and navigates the main WebView
//! at `http://127.0.0.1:<n>/mic/`.
//!
//! Why this shape rather than `tauri::api::http::*` or an inline web server:
//!   - The sidecar already serves the same HTML/JS the hosted /mic/
//!     UX uses. We don't want a parallel Rust HTTP server.
//!   - Spawning a child process gives us a clean kill on app close (we send
//!     SIGTERM in `on_window_event`) and crash recovery (the watchdog
//!     thread restarts the sidecar if it dies — TODO for v0.2).
//!
//! Lifecycle:
//!   1. main()  → builder.setup() spawns sidecar via tauri-plugin-shell
//!   2. Read child stdout line-by-line on a background thread
//!   3. On the line `BANI_READY port=<n>`, navigate the main window
//!   4. On window-close, kill the child and exit
//!
//! All ports live on 127.0.0.1 only — the sidecar binds to loopback by
//! default and never accepts off-machine connections.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::process::{CommandEvent, CommandChild};
use tauri_plugin_shell::ShellExt;

/// How long we wait for the sidecar to print `BANI_READY port=N` before
/// concluding it's wedged. ONNX init + corpus index typically take 2-3s on
/// cold start; we give a generous 30s for cold APFS / slow disks.
const SIDECAR_READY_TIMEOUT: Duration = Duration::from_secs(30);
/// Last N lines of sidecar stderr to carry over to the error page if the
/// sidecar dies before ready. Enough context to diagnose import failures
/// and ONNX model errors without flooding the URL.
const SIDECAR_ERROR_TAIL_LINES: usize = 20;

/// Helper: navigate the main window to the bundled error.html page with
/// `reason` + `log` query params. The error page reads them via JS and
/// renders a Retry button. Idempotent — calling twice just re-navigates.
fn navigate_to_error(app: &tauri::AppHandle, reason: &str, log_tail: &str) {
    if let Some(win) = app.get_webview_window("main") {
        // URL-encode the log tail so newlines + special chars survive. Cap
        // length so we don't hit any browser URL-length limit (most accept
        // ~2MB but be polite).
        let truncated = if log_tail.len() > 4000 {
            &log_tail[log_tail.len() - 4000..]
        } else {
            log_tail
        };
        let url = format!(
            "tauri://localhost/error.html?reason={}&log={}",
            urlencoding::encode(reason),
            urlencoding::encode(truncated),
        );
        if let Ok(parsed) = tauri::Url::parse(&url) {
            let _ = win.navigate(parsed);
        }
    }
}

/// Holds the sidecar child process so we can kill it on window close.
/// Wrapped in Mutex<Option<...>> so we can drop the handle (and thus
/// terminate the child) on shutdown without a `&mut self` to the state.
struct SidecarState {
    child: Mutex<Option<CommandChild>>,
}

/// Flip the private WKPreferences keys that gate `navigator.mediaDevices`
/// on macOS. WKWebView (unlike Safari) hides `getUserMedia` by default
/// for two reasons:
///
///   1. `mediaDevicesEnabled` defaults to `false` — must be explicitly
///      enabled via the private KVC key on `WKPreferences`.
///   2. `mediaCaptureRequiresSecureConnection` defaults to `true`, which
///      blocks media capture on `http://` origins. Our sidecar serves
///      from `http://127.0.0.1:<port>` so we must turn this off.
///
/// These are the same private API calls that wry already makes for
/// `fullScreenEnabled` and `developerExtrasEnabled`, so we're not
/// inventing new private API usage — see
/// `tauri-apps/wry::wkwebview::mod.rs` for the same pattern.
///
/// Why not put this in `tauri.conf.json`? Tauri's public config has no
/// knob for these private WKPreferences keys (see
/// docs.rs/tauri/2.11.0 — only `data_store_identifier`,
/// `allow_link_preview`, etc. are exposed). The only escape hatch is
/// `WebviewWindow::with_webview`, which is what we use here.
///
/// NOTE: For the user-visible permission prompt to actually appear,
/// `Info.plist` must also contain `NSMicrophoneUsageDescription` —
/// that's handled by the bundle config + `Info.plist` next to
/// `tauri.conf.json`. Without the plist key macOS will outright reject
/// the capture request *after* this code exposes the API, with a
/// generic "permission denied" error.
#[cfg(target_os = "macos")]
fn enable_macos_media_devices(window: &tauri::WebviewWindow) {
    // `setValue_forKey` is the Obj-C `-setValue:forKey:` KVC method.
    // It comes in via the `NSObjectNSKeyValueCoding` extension trait
    // in objc2-foundation 0.3 — `NSObject` (and any subclass like
    // `WKPreferences`) gets the method only when this trait is in
    // scope. The name is awkward but stable; same trait wry uses
    // internally for `fullScreenEnabled` etc.
    use objc2_foundation::{ns_string, NSNumber, NSObjectNSKeyValueCoding};
    use objc2_web_kit::WKWebView;

    let result = window.with_webview(|webview| {
        // SAFETY: `webview.inner()` returns the raw `*mut c_void`
        // pointing at the underlying WKWebView Obj-C object. Tauri
        // dispatches this closure on the main thread, which is also
        // the only thread allowed to touch WKWebView. The cast is
        // valid for the lifetime of the closure because Tauri keeps
        // the webview retained while it runs us.
        unsafe {
            let view: &WKWebView = &*webview.inner().cast();
            let configuration = view.configuration();
            let preferences = configuration.preferences();
            let yes = NSNumber::numberWithBool(true);
            let no = NSNumber::numberWithBool(false);

            // Expose `navigator.mediaDevices` (default: false on WKWebView).
            preferences.setValue_forKey(Some(&yes), ns_string!("mediaDevicesEnabled"));
            // Allow capture from http:// origins like 127.0.0.1.
            preferences.setValue_forKey(
                Some(&no),
                ns_string!("mediaCaptureRequiresSecureConnection"),
            );
            // WebRTC peer connections — required by some pipelines that
            // wrap getUserMedia. Cheap to set; no-op if already true.
            preferences.setValue_forKey(Some(&yes), ns_string!("peerConnectionEnabled"));
        }
        eprintln!("[shell] enabled WKPreferences mediaDevicesEnabled / mediaCaptureRequiresSecureConnection=false");
    });

    if let Err(e) = result {
        eprintln!("[shell] WARN: failed to flip WKPreferences for media: {:?}", e);
    }
}

#[cfg(not(target_os = "macos"))]
fn enable_macos_media_devices(_window: &tauri::WebviewWindow) {
    // No-op on Windows/Linux — WebView2 / WebKit2GTK expose
    // mediaDevices by default. Permission prompts come from the
    // OS-level mic picker on those platforms.
}

fn main() {
    tauri::Builder::default()
        // single-instance MUST be the first plugin per its docs — it intercepts
        // the second launch before the rest of the builder runs. Callback fires
        // in the *already-running* instance; we just focus the main window.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        // File logger: stdout (visible when launched from Terminal) + a rotating
        // file under the OS log dir (~/Library/Logs/Gurbani Captioning/ on macOS,
        // %LOCALAPPDATA%/Gurbani Captioning/logs/ on Windows). Default level Info
        // for our shell, Warn for noisy deps. Picks up Rust's `log` macros and
        // any eprintln!/println! routed through the `log` crate (we kept the
        // existing eprintln! sidecar pipes — they still hit stdout/stderr).
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Debug)  // accept debug-level (2) from JS telemetry (AGC, events)
                // Keep multiple GB of history for long sessions — default
                // rotation truncates after ~10 MB which loses the engine
                // timing lines we need for post-mortems. 50 MB * 5 files =
                // ~250 MB ceiling, plenty for a multi-hour kirtan session.
                .max_file_size(50_000_000)
                .rotation_strategy(tauri_plugin_log::RotationStrategy::KeepAll)
                .target(tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout))
                .target(tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir { file_name: None }))
                .build(),
        )
        // Updater: deferred. Plugin requires a signing-key + endpoints config
        // in tauri.conf.json which we haven't set up yet. The in-page banner in
        // mic.html (which checks GitHub Releases tag_name) already covers the
        // user-facing update notification, so this isn't blocking. Re-enable
        // once we have a Developer ID + minisign keys generated:
        //   tauri signer generate -w ~/.tauri/gurbani-captioning.key
        // .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState { child: Mutex::new(None) })
        .setup(|app| {
            // Spawn the bundled `gurbani-captioning` sidecar. tauri-plugin-shell looks
            // it up in resources via the externalBin entry in
            // tauri.conf.json. In dev (`cargo tauri dev`) Tauri appends the
            // target triple suffix (e.g. `gurbani-captioning-aarch64-apple-darwin`),
            // so we keep the symlink convention from sidecar.spec output:
            // `binaries/gurbani-captioning` is a copy/symlink of the PyInstaller
            // dist binary.
            let sidecar = app
                .shell()
                .sidecar("gurbani-captioning")
                .expect("gurbani-captioning sidecar binary must be bundled — see externalBin in tauri.conf.json")
                // Parent-death watchdog: pass our PID so the sidecar can poll
                // and self-exit if we crash, get force-quit, or get reparented
                // to launchd. Polling getppid() alone doesn't work because the
                // PyInstaller bootloader sits between us and the Python child.
                .env("BANI_SHELL_PID", std::process::id().to_string());

            let (mut rx, child) = sidecar
                .spawn()
                .expect("failed to spawn gurbani-captioning sidecar");

            // Stash the child so on_window_event can kill it on shutdown.
            // SAFETY: tauri::State is Send+Sync; the Mutex is just so we
            // can swap None in / Some out from the cleanup handler.
            *app.state::<SidecarState>().child.lock().unwrap() = Some(child);

            // Window will be (re)created once we have the sidecar port.
            // Until then the user sees... nothing — no main window yet,
            // just a Dock icon. We open a *splash* window first so they
            // see immediate feedback that the app launched. Later we'll
            // replace its URL with the real /mic/ page.
            let splash = WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::App("stub.html".into()),
            )
            .title("Gurbani Captioning")
            .inner_size(1200.0, 800.0)
            .min_inner_size(800.0, 600.0)
            .resizable(true)
            // Browser-style ⌘+/⌘-/⌘0 zoom. Tauri injects a polyfill that calls
            // setPageZoom under the hood — works correctly with WebKit unlike a
            // CSS-only approach. Requires `core:webview:allow-set-webview-zoom`
            // capability (granted in capabilities/default.json).
            .zoom_hotkeys_enabled(true)
            .build()?;

            // Flip the private WKPreferences keys that expose
            // `navigator.mediaDevices` in WKWebView. Must happen before
            // any page tries to use getUserMedia — splash.html doesn't,
            // but the /mic/ page we'll navigate to once the sidecar
            // signals ready certainly does. See doc-comment on the
            // function for the full rationale.
            enable_macos_media_devices(&splash);

            // Background thread: drain sidecar stdout, watch for the
            // ready line. Once seen, navigate the splash to /mic/.
            // Also runs a 30s deadline: if no BANI_READY arrives by then,
            // OR if the sidecar exits before ready, we navigate to error.html
            // with the tail of stderr so the user gets actionable feedback
            // instead of an indefinitely-hung splash screen.
            let app_handle = app.handle().clone();
            // Shared "are we ready yet?" + last-N stderr lines, both used
            // by the deadline thread to decide whether to show the error page.
            let port: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));
            let stderr_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::with_capacity(SIDECAR_ERROR_TAIL_LINES)));
            // Deadline watchdog: if `port` is still None after the timeout,
            // assume the sidecar is wedged and surface an error page. This
            // catches "ONNX init hung," "import-time exception with no print,"
            // and "port bind failed because another instance is already running."
            {
                let port = Arc::clone(&port);
                let stderr_tail = Arc::clone(&stderr_tail);
                let app_handle = app_handle.clone();
                thread::spawn(move || {
                    thread::sleep(SIDECAR_READY_TIMEOUT);
                    if port.lock().unwrap().is_none() {
                        let tail = stderr_tail.lock().unwrap().join("\n");
                        log::error!("[shell] sidecar didn't print BANI_READY within {}s; showing error page", SIDECAR_READY_TIMEOUT.as_secs());
                        navigate_to_error(
                            &app_handle,
                            "Engine didn't start in time (30s). Another copy of the app may be running, or the bundled model is missing/corrupt.",
                            &tail,
                        );
                    }
                });
            }
            thread::spawn(move || {
                while let Some(event) = rx.blocking_recv() {
                    match event {
                        CommandEvent::Stdout(line_bytes) => {
                            let line = String::from_utf8_lossy(&line_bytes);
                            let trimmed = line.trim_end();
                            eprintln!("[sidecar.out] {}", trimmed);
                            log::info!(target: "sidecar.out", "{}", trimmed);

                            // Look for the magic ready line. Format is
                            // exactly `BANI_READY port=<n>` (locked in by
                            // server.py — must update both sides if it
                            // ever changes).
                            if let Some(p) = parse_ready_port(&line) {
                                let already_set = {
                                    let mut guard = port.lock().unwrap();
                                    let was = guard.is_some();
                                    *guard = Some(p);
                                    was
                                };
                                if !already_set {
                                    let url = format!("http://127.0.0.1:{}/mic/", p);
                                    eprintln!("[shell] navigating webview to {}", url);
                                    if let Some(win) = app_handle.get_webview_window("main") {
                                        if let Ok(parsed) = tauri::Url::parse(&url) {
                                            let _ = win.navigate(parsed);
                                        }
                                    }
                                    let _ = app_handle.emit("sidecar-ready", url);
                                }
                            }
                        }
                        CommandEvent::Stderr(line_bytes) => {
                            let line = String::from_utf8_lossy(&line_bytes);
                            let trimmed = line.trim_end().to_string();
                            eprintln!("[sidecar.err] {}", trimmed);
                            log::info!(target: "sidecar.err", "{}", trimmed);
                            // Keep the last N lines in a ring buffer for the
                            // error page. We only care about these if the
                            // sidecar dies before BANI_READY — once ready,
                            // they're just informational and we keep
                            // appending but only the most recent are used.
                            let mut tail = stderr_tail.lock().unwrap();
                            if tail.len() >= SIDECAR_ERROR_TAIL_LINES {
                                tail.remove(0);
                            }
                            tail.push(trimmed);
                        }
                        CommandEvent::Terminated(payload) => {
                            eprintln!(
                                "[sidecar] exited code={:?} signal={:?}",
                                payload.code, payload.signal
                            );
                            log::warn!(target: "sidecar", "exited code={:?} signal={:?}", payload.code, payload.signal);
                            // If we never got to BANI_READY, the sidecar
                            // died at startup — surface what we know to the
                            // user. If we DID get to ready, this is a
                            // mid-session crash (rarer); the user can see
                            // the stale UI but no new events will arrive.
                            if port.lock().unwrap().is_none() {
                                let tail = stderr_tail.lock().unwrap().join("\n");
                                let reason = format!(
                                    "Engine exited at startup (code {:?}, signal {:?}).",
                                    payload.code, payload.signal,
                                );
                                navigate_to_error(&app_handle, &reason, &tail);
                            }
                            break;
                        }
                        _ => {}
                    }
                }
            });

            // Make the splash visible. (build() above already does this in
            // current Tauri but explicit is fine.)
            let _ = splash.show();
            Ok(())
        })
        .on_window_event(|window, event| {
            // When the user closes the main window, kill the sidecar so we
            // don't leave a zombie Python process listening on a random
            // localhost port.
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                kill_sidecar(&window.app_handle());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Gurbani Captioning shell")
        .run(|app_handle, event| {
            // Also handle app-level exit (⌘Q, Quit from menu, force-quit signal):
            // window CloseRequested doesn't always fire first, so we'd otherwise
            // orphan the sidecar. The parent-death watchdog in the sidecar
            // catches anything we miss here.
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                kill_sidecar(app_handle);
            }
        });
}

/// Helper: kill the stashed sidecar child if we still own one. Idempotent.
fn kill_sidecar(app: &tauri::AppHandle) {
    if let Some(child) = app.state::<SidecarState>().child.lock().unwrap().take() {
        eprintln!("[shell] terminating sidecar");
        let _ = child.kill();
    }
}

/// Parse `BANI_READY port=12345` (with optional trailing whitespace) and
/// return the port. Returns None for any other line, including "almost"
/// matches — we'd rather silently keep waiting than navigate to a bogus
/// port and get a connection-refused error in the WebView.
fn parse_ready_port(line: &str) -> Option<u16> {
    let line = line.trim();
    let suffix = line.strip_prefix("BANI_READY port=")?;
    suffix.trim().parse::<u16>().ok()
}

#[cfg(test)]
mod tests {
    use super::parse_ready_port;

    #[test]
    fn parses_canonical_line() {
        assert_eq!(parse_ready_port("BANI_READY port=54321"), Some(54321));
    }

    #[test]
    fn parses_with_trailing_newline() {
        assert_eq!(parse_ready_port("BANI_READY port=54321\n"), Some(54321));
    }

    #[test]
    fn rejects_other_lines() {
        assert_eq!(parse_ready_port("[sidecar] ASR ready"), None);
        assert_eq!(parse_ready_port("BANI_READY"), None);
        assert_eq!(parse_ready_port("BANI_READY port="), None);
        assert_eq!(parse_ready_port("BANI_READY port=abc"), None);
    }
}
