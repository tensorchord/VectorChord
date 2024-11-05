use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

pub use rayon::iter::ParallelIterator;

pub trait Parallelism: Send + Sync {
    fn check(&self);

    #[allow(clippy::wrong_self_convention)]
    fn into_par_iter<I: rayon::iter::IntoParallelIterator>(&self, x: I) -> I::Iter;
}

struct ParallelismCheckPanic(Box<dyn Any + Send>);

pub struct RayonParallelism {
    stop: Arc<dyn Fn() + Send + Sync>,
}

impl RayonParallelism {
    pub fn scoped<R>(
        num_threads: usize,
        stop: Arc<dyn Fn() + Send + Sync>,
        f: impl FnOnce(&Self) -> R,
    ) -> Result<R, rayon::ThreadPoolBuildError> {
        match std::panic::catch_unwind(AssertUnwindSafe(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .panic_handler(|e| {
                    if e.downcast_ref::<ParallelismCheckPanic>().is_some() {
                        return;
                    }
                    log::error!("Asynchronous task panickied.");
                })
                .build_scoped(
                    |thread| thread.run(),
                    |_| {
                        let pool = Self { stop: stop.clone() };
                        f(&pool)
                    },
                )
        })) {
            Ok(x) => x,
            Err(e) => match e.downcast::<ParallelismCheckPanic>() {
                Ok(payload) => std::panic::resume_unwind((*payload).0),
                Err(e) => std::panic::resume_unwind(e),
            },
        }
    }
}

impl Parallelism for RayonParallelism {
    fn check(&self) {
        match std::panic::catch_unwind(AssertUnwindSafe(|| (self.stop)())) {
            Ok(()) => (),
            Err(payload) => std::panic::panic_any(ParallelismCheckPanic(payload)),
        }
    }

    fn into_par_iter<I: rayon::iter::IntoParallelIterator>(&self, x: I) -> I::Iter {
        x.into_par_iter()
    }
}
