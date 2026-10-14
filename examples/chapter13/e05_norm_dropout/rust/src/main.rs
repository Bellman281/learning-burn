use burn::module::{Module, Param};
use burn::nn::{BatchNormConfig, DropoutConfig, LayerNormConfig};
use burn::tensor::{Device, Tensor};

// A layer behaves as in training when its input is on an AUTODIFF device AND its training
// flag is on (it is on after init; .valid() returns a copy with the flag off, on a plain device).
fn train_device() -> Device {
    Device::flex().autodiff() // autodiff    -> BatchNorm uses batch stats, Dropout is on
}
fn eval_device() -> Device {
    Device::flex() //            no autodiff -> BatchNorm uses running stats, Dropout is off
}

fn vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec().unwrap()
}

// x: [N=2, C=2, 2, 2]. Channel 0 holds 1..8, channel 1 holds four 1s then four 3s.
fn bn_input(device: &Device) -> Tensor<4> {
    Tensor::from_floats(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[5.0, 6.0], [7.0, 8.0]], [[3.0, 3.0], [3.0, 3.0]]],
        ],
        device,
    )
}

/// BatchNorm2d: (train output, running mean, running var, eval output).
fn batch_norm() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = train_device();
    let mut bn = BatchNormConfig::new(2).init(&device); // momentum 0.1, eps 1e-5
    bn.gamma = Param::from_tensor(Tensor::from_floats([2.0, 0.5], &device));
    bn.beta = Param::from_tensor(Tensor::from_floats([1.0, -1.0], &device));

    // One training-mode forward pass: normalise with THIS batch's statistics,
    // and nudge the running statistics towards them.
    let y_train = bn.forward(bn_input(&device));

    // .valid(): same weights, same running stats, training flag off, no autodiff.
    let bn_eval = bn.valid();
    let running_mean = vec(bn_eval.running_mean.value());
    let running_var = vec(bn_eval.running_var.value());
    let y_eval = bn_eval.forward(bn_input(&eval_device()));

    (vec(y_train), running_mean, running_var, vec(y_eval))
}

/// LayerNorm over the last axis of a [2, 4] input, with a learned scale and shift.
fn layer_norm() -> Vec<f32> {
    let device = eval_device();
    let mut ln = LayerNormConfig::new(4).init(&device); // eps 1e-5
    ln.gamma = Param::from_tensor(Tensor::from_floats([1.0, 0.5, 2.0, 1.0], &device));
    ln.beta = Some(Param::from_tensor(Tensor::from_floats(
        [0.0, 0.0, 0.0, 1.0],
        &device,
    )));
    let x = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], &device);
    vec(ln.forward(x))
}

/// Dropout(p = 0.25) on 1000 ones: (train output, eval output).
fn dropout() -> (Vec<f32>, Vec<f32>) {
    let (train_dev, eval_dev) = (train_device(), eval_device());
    let drop = DropoutConfig::new(0.25).init(); // no weights, no device: just p and a flag
    train_dev.seed(42);
    let train = drop.forward(Tensor::<1>::ones([1000], &train_dev));
    let eval = drop.forward(Tensor::<1>::ones([1000], &eval_dev));
    (vec(train), vec(eval))
}

fn fmt(v: &[f32]) -> String {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", s.join(", "))
}

fn main() {
    let (y_train, mean, var, y_eval) = batch_norm();
    println!("BatchNorm2d, training mode (batch statistics):");
    println!("  image 0, channel 0 = {}", fmt(&y_train[0..4]));
    println!("  image 0, channel 1 = {}", fmt(&y_train[4..8]));
    println!(
        "  running_mean = {}   running_var = {}",
        fmt(&mean),
        fmt(&var)
    );
    println!("BatchNorm2d, eval mode (running statistics):");
    println!("  image 0, channel 0 = {}", fmt(&y_eval[0..4]));
    println!("  image 0, channel 1 = {}", fmt(&y_eval[4..8]));

    let ln = layer_norm();
    println!("LayerNorm row 0 = {}", fmt(&ln[0..4]));
    println!("LayerNorm row 1 = {}", fmt(&ln[4..8]));

    let (train, eval) = dropout();
    let kept = train.iter().filter(|v| **v != 0.0).count();
    let mean = train.iter().sum::<f32>() / 1000.0;
    let scaled = train.iter().copied().fold(0.0f32, f32::max);
    println!(
        "Dropout p=0.25, autodiff device: kept {kept}/1000, kept value {scaled:.4}, mean {mean:.4}"
    );
    let untouched = eval.iter().all(|v| *v == 1.0);
    println!("Dropout p=0.25, plain device:    output == input: {untouched}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{x} vs {y}");
        }
    }

    #[test]
    fn matches_pytorch() {
        let (y_train, mean, var, y_eval) = batch_norm();
        // From main.py: nn.BatchNorm2d in train() mode, first 4 values of each channel (image 0).
        close(
            &y_train[0..8],
            &[-2.0551, -1.1822, -0.3093, 0.5636, -1.5, -1.5, -1.5, -1.5],
            1e-4,
        );
        close(&mean, &[0.45, 0.2], 1e-6);
        // PyTorch's running_var is [1.5, 1.0142857]: it uses the UNBIASED batch variance
        // (6.0 and 8/7). Burn uses the biased one (5.25 and 1.0): 0.9 + 0.1 * 5.25 = 1.425.
        close(&var, &[1.425, 1.0], 1e-6);
        // main.py reproduces this eval output once its running_var is set to Burn's value.
        close(&y_eval[0..4], &[1.9215, 3.5969, 5.2723, 6.9477], 1e-4);
        close(&y_eval[4..8], &[-0.6, -0.6, -0.6, -0.6], 1e-4);

        close(
            &layer_norm(),
            &[
                -1.3416, -0.2236, 0.8944, 2.3416, -1.3416, -0.2236, 0.8944, 2.3416,
            ],
            1e-4,
        );

        let (train, eval) = dropout();
        assert!(
            train
                .iter()
                .all(|v| *v == 0.0 || (v - 4.0 / 3.0).abs() < 1e-6)
        );
        let kept = train.iter().filter(|v| **v != 0.0).count();
        assert!((700..=800).contains(&kept), "kept {kept}"); // 750 +- 3.7 sigma
        assert!(eval.iter().all(|v| *v == 1.0));
    }
}
