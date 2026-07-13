// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the terms of the ELv2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025-2026 TensorChord Inc.

#[derive(Clone, Copy, Debug)]
pub enum MaxsimCostBackend {
    CoarseOnly,
    CpuExact,
    Gpu,
    Auto,
}

#[derive(Clone, Copy, Debug)]
pub struct MaxsimCostInput {
    pub heap_rows: f64,
    pub index_tokens: f64,
    pub token_nodes_per_query: f64,
    pub base_index_pages: f64,
    pub dimension: u32,
    pub element_bits: u32,
    pub query_tokens: u32,
    pub limit_tuples: Option<f64>,
    pub filter_selectivity: f64,
    pub candidate_limit: Option<u32>,
    pub backend: MaxsimCostBackend,
}

#[derive(Clone, Copy, Debug)]
pub struct MaxsimCostEstimate {
    pub startup_cost: f64,
    pub total_cost: f64,
    pub selectivity: f64,
    pub index_pages: f64,
}

pub fn estimate_maxsim_cost(input: MaxsimCostInput) -> MaxsimCostEstimate {
    let heap_rows = input.heap_rows.max(1.0);
    let index_tokens = input.index_tokens.max(heap_rows);
    let query_tokens = f64::from(input.query_tokens.max(1));
    let average_document_tokens = (index_tokens / heap_rows).clamp(1.0, 65_536.0);
    let token_visits = input.token_nodes_per_query.max(1.0) * query_tokens;

    // We do not yet persist page-level candidate statistics. Until then, use a
    // conservative occupancy estimate: token visits are sampled from the token
    // index, and a page is a candidate when at least one of its average tokens
    // is visited. This is intentionally bounded by the heap row count.
    let token_visit_fraction = (token_visits / index_tokens).clamp(0.0, 1.0);
    let candidate_probability = 1.0 - (1.0 - token_visit_fraction).powf(average_document_tokens);
    let generated_pages = (heap_rows * candidate_probability).clamp(1.0, heap_rows);

    let filtered_limit = input
        .limit_tuples
        .map(|limit| limit.max(1.0) / input.filter_selectivity.clamp(1e-9, 1.0));
    let exact_candidate_count = input
        .candidate_limit
        .map_or(generated_pages, |limit| {
            f64::from(limit).min(generated_pages)
        })
        .clamp(1.0, heap_rows);
    let returned_pages = match input.backend {
        MaxsimCostBackend::CoarseOnly => filtered_limit
            .unwrap_or(generated_pages)
            .min(generated_pages),
        MaxsimCostBackend::CpuExact | MaxsimCostBackend::Gpu | MaxsimCostBackend::Auto => {
            exact_candidate_count
        }
    };

    // Candidate generation and aggregation are eager in the current scanner,
    // so their full work belongs to startup cost even when SQL has a small
    // LIMIT. The constants are deliberately conservative placeholders until
    // committed corpus benchmarks replace them with fitted values.
    let search_cost = 0.001 * token_visits;
    let aggregation_cost = 0.01 * token_visits + 0.05 * generated_pages;
    let exact_components = exact_candidate_count
        * average_document_tokens
        * query_tokens
        * f64::from(input.dimension.max(1));
    let tensor_bytes = exact_candidate_count
        * average_document_tokens
        * f64::from(input.dimension.max(1))
        * f64::from(input.element_bits.max(1))
        / 8.0;
    let cpu_exact_cost = exact_candidate_count + exact_components * 1e-6;
    let gpu_exact_cost = 5.0 + tensor_bytes * 1e-7 + exact_components * 5e-8;
    let backend_cost = match input.backend {
        MaxsimCostBackend::CoarseOnly => 0.0,
        MaxsimCostBackend::CpuExact => cpu_exact_cost,
        MaxsimCostBackend::Gpu => gpu_exact_cost,
        // Price a small but nonzero fallback risk. Runtime still performs a
        // complete CPU rerank on every GPU failure.
        MaxsimCostBackend::Auto => gpu_exact_cost + 0.05 * cpu_exact_cost,
    };
    let startup_cost = search_cost + aggregation_cost + backend_cost;
    let total_cost = startup_cost + returned_pages;
    let selectivity = (returned_pages / heap_rows).clamp(1e-9, 1.0);
    let index_pages =
        input.base_index_pages.max(1.0) * (1.0 + 0.25 * (query_tokens - 1.0).max(0.0));

    MaxsimCostEstimate {
        startup_cost,
        total_cost,
        selectivity,
        index_pages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(backend: MaxsimCostBackend) -> MaxsimCostInput {
        MaxsimCostInput {
            heap_rows: 34_054.0,
            index_tokens: 25_438_338.0,
            token_nodes_per_query: 10_000.0,
            base_index_pages: 20_000.0,
            dimension: 320,
            element_bits: 16,
            query_tokens: 32,
            limit_tuples: Some(20.0),
            filter_selectivity: 1.0,
            candidate_limit: Some(256),
            backend,
        }
    }

    #[test]
    fn maxsim_is_never_zero_cost() {
        for backend in [
            MaxsimCostBackend::CoarseOnly,
            MaxsimCostBackend::CpuExact,
            MaxsimCostBackend::Gpu,
            MaxsimCostBackend::Auto,
        ] {
            let estimate = estimate_maxsim_cost(input(backend));
            assert!(estimate.startup_cost > 0.0);
            assert!(estimate.total_cost >= estimate.startup_cost);
            assert!((1e-9..=1.0).contains(&estimate.selectivity));
            assert!(estimate.index_pages >= 1.0);
        }
    }

    #[test]
    fn query_token_count_increases_eager_work() {
        let one = estimate_maxsim_cost(MaxsimCostInput {
            query_tokens: 1,
            ..input(MaxsimCostBackend::CpuExact)
        });
        let many = estimate_maxsim_cost(MaxsimCostInput {
            query_tokens: 64,
            ..input(MaxsimCostBackend::CpuExact)
        });
        assert!(many.startup_cost > one.startup_cost);
        assert!(many.index_pages > one.index_pages);
    }

    #[test]
    fn exact_candidate_limit_bounds_rows_and_cost() {
        let small = estimate_maxsim_cost(MaxsimCostInput {
            candidate_limit: Some(128),
            ..input(MaxsimCostBackend::CpuExact)
        });
        let large = estimate_maxsim_cost(MaxsimCostInput {
            candidate_limit: Some(2048),
            ..input(MaxsimCostBackend::CpuExact)
        });
        assert!(small.selectivity < large.selectivity);
        assert!(small.startup_cost < large.startup_cost);
    }

    #[test]
    fn auto_prices_more_than_gpu_for_fallback_risk() {
        let gpu = estimate_maxsim_cost(input(MaxsimCostBackend::Gpu));
        let auto = estimate_maxsim_cost(input(MaxsimCostBackend::Auto));
        assert!(auto.startup_cost > gpu.startup_cost);
    }

    #[test]
    fn missing_stats_remain_finite() {
        let estimate = estimate_maxsim_cost(MaxsimCostInput {
            heap_rows: -1.0,
            index_tokens: 0.0,
            token_nodes_per_query: 0.0,
            base_index_pages: 0.0,
            filter_selectivity: 0.0,
            limit_tuples: None,
            candidate_limit: None,
            ..input(MaxsimCostBackend::CoarseOnly)
        });
        assert!(estimate.startup_cost.is_finite());
        assert!(estimate.total_cost.is_finite());
        assert!(estimate.selectivity.is_finite());
        assert!(estimate.index_pages.is_finite());
    }
}
