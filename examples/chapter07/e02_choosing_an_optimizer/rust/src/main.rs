use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamConfig, AdamWConfig, SgdConfig};

// An optimizer is no longer typed by the model it will train: every `init()`
// returns the same `ModuleOptimizer`, and the model is handed over at `step`.

fn main() {
    // Plain stochastic gradient descent.
    let _sgd = SgdConfig::new().init();

    // SGD with momentum.
    let _sgd_m = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
        .init();

    // Adam uses adaptive learning rates.
    let _adam = AdamConfig::new().init();

    // AdamW uses decoupled weight decay.
    let _adamw = AdamWConfig::new().with_weight_decay(0.01).init();

    println!("Optimizers created: SGD, SGD with momentum, Adam, AdamW");
}
