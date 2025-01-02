use nalgebra::DMatrix;

#[ignore]
#[test]
fn check_full_rank_matrix() {
    let parallelism = std::thread::available_parallelism().unwrap().get();
    std::thread::scope(|scope| {
        let mut threads = vec![];
        for remainder in 0..parallelism {
            threads.push(scope.spawn(move || {
                for n in (0..=60000).filter(|x| x % parallelism == remainder) {
                    let matrix = random_full_rank_matrix(n);
                    assert!(matrix.is_invertible());
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
    });
}

fn random_full_rank_matrix(n: usize) -> DMatrix<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;
    use rand_distr::StandardNormal;
    let mut rng = ChaCha12Rng::from_seed([7; 32]);
    DMatrix::from_fn(n, n, |_, _| rng.sample(StandardNormal))
}

#[test]
fn check_random_orthogonal_matrix() {
    assert_eq!(random_orthogonal_matrix(2), vec![
        vec![-0.5424608, -0.8400813],
        vec![0.8400813, -0.54246056]
    ]);
    assert_eq!(random_orthogonal_matrix(3), vec![
        vec![-0.5309615, -0.69094884, -0.49058124],
        vec![0.8222731, -0.56002235, -0.10120347],
        vec![0.20481002, 0.45712686, -0.86549866]
    ]);
}

pub fn random_orthogonal_matrix(n: usize) -> Vec<Vec<f32>> {
    use nalgebra::QR;
    let matrix = random_full_rank_matrix(n);
    // QR decomposition is unique if the matrix is full rank
    let qr = QR::new(matrix);
    let q = qr.q();
    let mut projection = Vec::new();
    for row in q.row_iter() {
        projection.push(row.iter().copied().collect::<Vec<f32>>());
    }
    projection
}
