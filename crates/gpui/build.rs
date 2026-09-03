#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gles)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "windows" {
        #[cfg(feature = "windows-manifest")]
        embed_resource();
    }
}

#[cfg(feature = "windows-manifest")]
fn embed_resource() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("windows")
        .join("gpui.manifest.xml");
    println!("cargo:rerun-if-changed={}", manifest.display());

    // The resource script is written here, naming the manifest by absolute
    // path. MSVC's rc.exe resolves a resource file against the crate root it
    // runs from; a cross build's llvm-rc runs against the preprocessed copy
    // of the script inside OUT_DIR, where a crate-relative name resolves to
    // nothing, and it ignores include directories for resource files. An
    // absolute path is the one spelling both accept.
    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo sets OUT_DIR for build scripts"),
    );
    let rc_file = out_dir.join("gpui.rc");
    let manifest_literal = manifest.to_string_lossy().replace('\\', "\\\\");
    std::fs::write(
        &rc_file,
        format!("#define RT_MANIFEST 24\n1 RT_MANIFEST \"{manifest_literal}\"\n"),
    )
    .expect("writing the resource script into OUT_DIR");
    embed_resource::compile(&rc_file, embed_resource::NONE)
        .manifest_required()
        .unwrap();
}
