use rustygrad::{
    mlp::{Layer, Neuron, MLP},
    RefValue,
};

fn main() {
    let x1 = RefValue::new(2.0);
    let x2 = RefValue::new(0.0);
    let w1 = RefValue::new(-3.0);
    let w2 = RefValue::new(1.0);

    let b = RefValue::new(6.8813735870195432);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let e = (2 * n).exp();
    let o = (e.clone() - 1) / (e + 1);

    // println!("{:#?}", (*o).borrow().value);

    let l = o.backward();
    // println!("{l:#?}");

    let mlp = MLP::new(2, &mut vec![4, 4, 1]);
    let x = vec![RefValue::new(2.0), RefValue::new(3.0)];
    println!("{:?}", mlp(x, ()));
}
