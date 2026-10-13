use burn::tensor::{Device, Int, Tensor};

fn main() {
    let device = Device::flex();

    let z = Tensor::<2>::zeros([2, 3], &device);
    let o = Tensor::<2>::ones([2, 3], &device);
    let f = Tensor::<2>::full([2, 3], 7.0, &device);

    // identity matrix
    let eye = Tensor::<2>::eye(3, &device);

    // a range as an Int tensor: [5, 6, 7, 8, 9]
    let r = Tensor::<1, Int>::arange(5..10, &device);

    println!("zeros = {}", z);
    println!("ones = {}", o);
    println!("full = {}", f);
    println!("eye = {}", eye);
    println!("range = {}", r);
}
