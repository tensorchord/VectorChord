use half::f16;
use random_orthogonal_matrix::random_orthogonal_matrix;
use std::sync::OnceLock;
use vector::vect::{VectBorrowed, VectOwned};

fn matrix(n: usize) -> Option<&'static Vec<Vec<f32>>> {
    static MATRIXS: [OnceLock<Vec<Vec<f32>>>; 1 + 60000] = [const { OnceLock::new() }; 1 + 60000];
    MATRIXS
        .get(n)
        .map(|x| x.get_or_init(|| random_orthogonal_matrix(n)))
}

pub fn prewarm(n: usize) {
    let _ = matrix(n);
}

pub fn project(vector: &[f32]) -> Vec<f32> {
    use simd::Floating;
    let n = vector.len();
    let matrix = matrix(n).expect("dimension too large");
    (0..n)
        .map(|i| f32::reduce_sum_of_xy(vector, &matrix[i]))
        .collect()
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        VectOwned::new(project(self.slice()))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use simd::Floating;
        VectOwned::new(f16::vector_from_f32(&project(&f16::vector_to_f32(
            self.slice(),
        ))))
    }
}
