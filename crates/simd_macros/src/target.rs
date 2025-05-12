// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

pub struct TargetCpu {
    pub target_cpu: &'static str,
    pub target_arch: &'static str,
    pub target_features: &'static [&'static str],
}

pub const TARGET_CPUS: &[TargetCpu] = &[
    // This is a temporary hack, for using AVX512FP16 without unstable features.
    TargetCpu {
        target_cpu: "v4fp16",
        target_arch: "x86_64",
        target_features: &[
            "avx512bw", "avx512cd", "avx512dq", "avx512vl", // simd
            "bmi1", "bmi2", "lzcnt", "movbe", "popcnt", // bit-operations
        ],
    },
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
