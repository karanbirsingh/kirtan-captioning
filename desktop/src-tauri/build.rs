use std::fs;
use std::path::PathBuf;

fn main() {
    // Sync APP_VERSION in mic.html to match Cargo.toml version so the
    // in-app "New version available" banner doesn't show for the current release.
    let version = env!("CARGO_PKG_VERSION");
    let html_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../frontend-stub/mic.html");
    if let Ok(content) = fs::read_to_string(&html_path) {
        if let Some(start) = content.find("const APP_VERSION = \"") {
            let after = start + "const APP_VERSION = \"".len();
            if let Some(end_rel) = content[after..].find('"') {
                let end = after + end_rel;
                let current = &content[after..end];
                if current != version {
                    let mut new_content = String::with_capacity(content.len());
                    new_content.push_str(&content[..after]);
                    new_content.push_str(version);
                    new_content.push_str(&content[end..]);
                    fs::write(&html_path, new_content).expect("failed to write mic.html");
                    println!(
                        "cargo:warning=Synced APP_VERSION in mic.html: {} -> {}",
                        current, version
                    );
                }
            }
        }
    }
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../frontend-stub/mic.html");

    tauri_build::build()
}
