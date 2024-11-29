use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86_64")]
#[derive(Debug, Serialize, Deserialize)]
pub struct HWConfig {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512f16: bool,
    // pub avx512bf16: bool,
    pub avx512bw: bool,
    pub avx512_vnni: bool,
    pub fma: bool,
    pub fma4: bool,
    pub f16c: bool,
    pub family_id: u8,
    pub model_id: u8,
}
#[cfg(target_arch = "x86")]
#[derive(Debug, Serialize, Deserialize)]
pub struct HWConfig {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub family_id: u8,
    pub model_id: u8,
}
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Serialize, Deserialize)]
pub struct HWConfig {
    pub sve: bool,
    pub neon: bool,
    pub fp16: bool,
    pub f32mm: bool,
    pub fcma: bool,
    pub i8mm: bool,
    pub model_name: String,
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
#[derive(Debug, Serialize, Deserialize)]
pub struct HWConfig {}

pub fn detect_hw_config() -> HWConfig {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let extended_feature_info = cpuid.get_extended_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let avx = feature_info.has_avx();
        let fma = feature_info.has_fma();
        let avx2 = extended_feature_info.has_avx2();
        let avx512f16 = extended_feature_info.has_avx512_fp16();
        // let avx512bf16 = extended_feature_info.has_avx512_bf16();
        let avx512f = extended_feature_info.has_avx512f();
        let avx512bw = extended_feature_info.has_avx512bw();
        let avx512_vnni = extended_feature_info.has_avx512vnni();
        let f16c = feature_info.has_f16c();
        let extended_processor_info = cpuid.get_extended_processor_and_feature_identifiers().unwrap();
        let fma4 = extended_processor_info.has_fma4();
        let family_id = feature_info.family_id();
        let model_id = feature_info.model_id();
        return HWConfig {
            sse,
            sse2,
            sse3,
            ssse3,
            avx,
            avx2,
            avx512f,
            avx512f16,
            avx512bw,
            avx512_vnni,
            fma,
            fma4,
            f16c,
            family_id,
            model_id,
        };
    }

    #[cfg(target_arch = "x86")]
    {
        let cpuid = raw_cpuid::CpuId::new();
        let feature_info = cpuid.get_feature_info().unwrap();
        let sse = feature_info.has_sse();
        let sse2 = feature_info.has_sse2();
        let sse3 = feature_info.has_sse3();
        let ssse3 = feature_info.has_ssse3();
        let family_id = feature_info.family_id();
        let model_id = feature_info.model_id();
        return HWConfig { sse, sse2, sse3, ssse3, family_id, model_id };
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected;
        let neon = is_aarch64_feature_detected!("neon");
        let sve = is_aarch64_feature_detected!("sve");
        let fp16 = is_aarch64_feature_detected!("fp16");
        let f32mm = is_aarch64_feature_detected!("f32mm");
        let fcma = is_aarch64_feature_detected!("fcma");
        let i8mm = is_aarch64_feature_detected!("i8mm");
        // hack until sys-info gets updated
        #[cfg(target_os = "linux")]
        let model_name = std::fs::read_to_string("/proc/cpuinfo")
            .unwrap()
            .lines()
            .find(|line| line.starts_with("CPU part"))
            .unwrap()
            .split(":")
            .collect::<Vec<&str>>()[1]
            .trim()
            .to_string();
        #[cfg(target_os = "windows")]
        let model_name = "Unknown".to_string();
        return HWConfig { sve, neon, fp16, f32mm, fcma, i8mm };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        return HWConfig {};
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub hw_config: HWConfig,
    pub os: String,
    pub arch: String,
    pub num_threads: usize,
}

pub fn get_benchmark_config() -> BenchmarkConfig {
    let hw_config = detect_hw_config();
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let default_num_threads = std::thread::available_parallelism().unwrap().get();
    let num_threads = std::env::var("NUM_THREADS").map(|s| s.parse().unwrap()).unwrap_or(default_num_threads);
    return BenchmarkConfig { hw_config, os, arch, num_threads };
}
