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
        target_features: &["sve"],
    },
    TargetCpu {
        target_cpu: "a3.256",
        target_arch: "aarch64",
        target_features: &["sve"],
    },
    TargetCpu {
        target_cpu: "a3.128",
        target_arch: "aarch64",
        target_features: &["sve"],
    },
    TargetCpu {
        target_cpu: "a2",
        target_arch: "aarch64",
        target_features: &["neon"],
    },
    TargetCpu {
        target_cpu: "z17",
        target_arch: "s390x",
        target_features: &[
            "vector",
            "vector-enhancements-1",
            "vector-enhancements-2",
            "vector-enhancements-3",
            "vector-packed-decimal",
            "vector-packed-decimal-enhancement",
            "vector-packed-decimal-enhancement-2",
            "vector-packed-decimal-enhancement-3",
            "nnp-assist",
            "miscellaneous-extensions-2",
            "miscellaneous-extensions-3",
            "miscellaneous-extensions-4",
        ],
    },
    TargetCpu {
        target_cpu: "z16",
        target_arch: "s390x",
        target_features: &[
            "vector",
            "vector-enhancements-1",
            "vector-enhancements-2",
            "vector-packed-decimal",
            "vector-packed-decimal-enhancement",
            "vector-packed-decimal-enhancement-2",
            "nnp-assist",
            "miscellaneous-extensions-2",
            "miscellaneous-extensions-3",
        ],
    },
    TargetCpu {
        target_cpu: "z15",
        target_arch: "s390x",
        target_features: &[
            "vector",
            "vector-enhancements-1",
            "vector-enhancements-2",
            "vector-packed-decimal",
            "vector-packed-decimal-enhancement",
            "miscellaneous-extensions-2",
            "miscellaneous-extensions-3",
        ],
    },
    TargetCpu {
        target_cpu: "z14",
        target_arch: "s390x",
        target_features: &[
            "vector",
            "vector-enhancements-1",
            "vector-packed-decimal",
            "miscellaneous-extensions-2",
        ],
    },
    TargetCpu {
        target_cpu: "z13",
        target_arch: "s390x",
        target_features: &["vector"],
    },
    TargetCpu {
        target_cpu: "p9",
        target_arch: "powerpc64",
        target_features: &[
            "altivec",
            "vsx",
            "power8-altivec",
            "power8-crypto",
            "power8-vector",
            "power9-altivec",
            "power9-vector",
        ],
    },
    TargetCpu {
        target_cpu: "p8",
        target_arch: "powerpc64",
        target_features: &[
            "altivec",
            "vsx",
            "power8-altivec",
            "power8-crypto",
            "power8-vector",
        ],
    },
    TargetCpu {
        target_cpu: "p7",
        target_arch: "powerpc64",
        target_features: &["altivec", "vsx"],
    },
    TargetCpu {
        target_cpu: "r1",
        target_arch: "riscv64",
        target_features: &["v"],
    },
];
