pub struct TargetCpu {
    pub target_cpu: &'static str,
    pub target_arch: &'static str,
    pub target_features: &'static [&'static str],
}

pub const TARGET_CPUS: &[TargetCpu] = &[
    TargetCpu {
        target_cpu: "v4",
        target_arch: "x86_64",
        target_features: &[
            "avx512bw", "avx512cd", "avx512dq", "avx512vl", // simd
            "bmi1", "bmi2", "lzcnt", "movbe", "popcnt", // bit-operations
        ],
    },
    TargetCpu {
        target_cpu: "v3",
        target_arch: "x86_64",
        target_features: &[
            "avx2", "f16c", "fma", // simd
            "bmi1", "bmi2", "lzcnt", "movbe", "popcnt", // bit-operations
        ],
    },
    TargetCpu {
        target_cpu: "v2",
        target_arch: "x86_64",
        target_features: &[
            "sse4.2", // simd
            "popcnt", // bit-operations
        ],
    },
    TargetCpu {
        target_cpu: "a3.512",
        target_arch: "aarch64",
        target_features: &[
            "sve", // simd
        ],
    },
    TargetCpu {
        target_cpu: "a3.256",
        target_arch: "aarch64",
        target_features: &[
            "sve", // simd
        ],
    },
    TargetCpu {
        target_cpu: "a3.128",
        target_arch: "aarch64",
        target_features: &[
            "sve", // simd
        ],
    },
    TargetCpu {
        target_cpu: "a2",
        target_arch: "aarch64",
        target_features: &[
            "neon", // simd
        ],
    },
];
