use std::{env, path::PathBuf, process::Command};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=native_probe.m");
    if env::var_os("CARGO_FEATURE_NATIVE_PROBES").is_none() { return Ok(()); }
    if env::var("CARGO_CFG_TARGET_OS")? != "macos" { return Err("native probes require macOS".into()); }
    let architecture = match env::var("CARGO_CFG_TARGET_ARCH")?.as_str() {
        "aarch64" => "arm64", "x86_64" => "x86_64", _ => return Err("unsupported native probe architecture".into()),
    };
    let output = PathBuf::from(env::var("OUT_DIR")?);
    let object = output.join("native_probe.o");
    let archive = output.join("libnative_probe.a");
    let status = Command::new("/usr/bin/clang").args(["-arch", architecture, "-x", "objective-c", "-O0", "-g", "-fno-omit-frame-pointer", "-fno-objc-arc", "-c", "native_probe.m", "-o"]).arg(&object).status()?;
    if !status.success() { return Err("native probe clang failed".into()); }
    let status = Command::new("/usr/bin/ar").arg("rcs").arg(&archive).arg(&object).status()?;
    if !status.success() { return Err("native probe archive failed".into()); }
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=native_probe");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreText");
    Ok(())
}
