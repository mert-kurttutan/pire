//! This build script emits the openblas linking directive if requested
#![allow(unused_variables)]
#[derive(PartialEq, Eq)]
enum Library {
    Static,
    Dynamic,
}

macro_rules! env_or {
    ($name:expr) => {
        if let Some(value) = std::option_env!($name) {
            value
        } else {
            "env"
        }
    };
 }

pub const SEQUENTIAL: bool = false;
pub const THREADED: bool = !SEQUENTIAL;

pub const LD_DIR: &str = if cfg!(windows) {
    "PATH"
} else if cfg!(target_os = "linux") {
    "LD_LIBRARY_PATH"
} else if cfg!(target_os = "macos") {
    "DYLD_LIBRARY_PATH"
} else {
    ""
};

pub const MKL_CORE: &str = "mkl_core";
pub const MKL_THREAD: &str = if SEQUENTIAL {
    "mkl_sequential"
} else {
    "mkl_intel_thread"
};
pub const THREADING_LIB: &str = if cfg!(windows) { "libiomp5md" } else { "iomp5" };
pub const MKL_INTERFACE: &str = if cfg!(target_pointer_width = "32") {
    "mkl_intel_ilp32"
 } else if cfg!(windows) {
     "mkl_rt"   // windows uses different lp interface library
 } else {
     "mkl_intel_lp64"
 };

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub const UNSUPPORTED_OS_ERROR: _ = "Target OS is not supported. Please contact me";

pub const LINK_DIRS: &[&str] = &[
    env_or!("GLARE_MKL_PATH"),
];


pub const LIB_DIRS: &[&str] = LINK_DIRS;

#[derive(Debug)]
pub enum BuildError {
    OneAPINotFound(std::path::PathBuf),
    OneAPINotADir(std::path::PathBuf),
    PathNotFound(std::env::VarError),
    AddSharedLibDirToPath(String),
}

fn main() -> Result<(), BuildError> {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-env-changed=STATIC");
    let library = if std::env::var("STATIC").unwrap_or_else(|_| "0".to_string()) == "1" {
        Library::Static
    } else {
        Library::Dynamic
    };

    let link_type: &str = if Library::Static == library {
        "static"
    } else {
        "dylib"
    };
    let lib_postfix: &str = if cfg!(windows) {
        ""
    } else {
        ""
    };

    for rel_lib_dir in LINK_DIRS {
        let lib_dir: std::path::PathBuf = rel_lib_dir.into();
        println!("cargo:rustc-link-search={}", lib_dir.display());
    }


    #[cfg(feature = "mkl")]
    {

        println!("cargo:rustc-link-lib={link_type}={MKL_INTERFACE}{lib_postfix}");
        println!("cargo:rustc-link-lib={link_type}={MKL_THREAD}{lib_postfix}");
        println!("cargo:rustc-link-lib={link_type}={MKL_CORE}{lib_postfix}");
        if THREADED {
            println!("cargo:rustc-link-lib=dylib={THREADING_LIB}");
        }

        if !cfg!(windows) {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
        }

        #[cfg(target_os = "macos")]
        {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}/{MACOS_COMPILER_PATH}",
                root.display(),
            );
        }
    }

    Ok(())
}