use std::borrow::{Borrow, BorrowMut};

use rustygrad::{
    mlp::{Layer, Neuron, MLP},
    RefValue,
};
use tqdm::tqdm;

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
    // let o = n.tanh();
    let e = (2 * n).exp();
    let o = (e.clone() - 1) / (e + 1);

    // println!("{:#?}", (*o).borrow().value);

    let l = o.backward();
    // println!("{l:#?}");

    let example_dataset = vec![
        vec![RefValue::new(2.0), RefValue::new(3.0), RefValue::new(-1.0)],
        vec![RefValue::new(3.0), RefValue::new(-1.0), RefValue::new(0.5)],
        vec![RefValue::new(0.5), RefValue::new(1.0), RefValue::new(1.0)],
        vec![RefValue::new(1.0), RefValue::new(1.0), RefValue::new(-1.0)],
    ];

    let expect_res = vec![
        RefValue::new(1.0),
        RefValue::new(-1.0),
        RefValue::new(-1.0),
        RefValue::new(1.0),
    ];

    let mlp = MLP::new(3, &mut vec![4, 4, 1]);
    println!("--------------- [ TRY ] ---------------");
    example_dataset.iter().for_each(|example| {
        let res = mlp.call(example.clone());
        println!("Res - {}", res.get(0).unwrap().item());
    });

    println!("--------------- [ TRAIN ] ---------------");
    for _ in tqdm(0..1500) {
        //Forward Pass
        let mut ypred = vec![];
        example_dataset.iter().for_each(|example| {
            let res = mlp.call(example.clone());
            ypred.push(res.get(0).unwrap().clone());
        });
        //println!("{:?}", ypred.clone());

        let loss: RefValue = expect_res
            .iter()
            .zip(ypred.iter())
            .map(|(ygt, yout)| (yout.clone() - ygt.clone()).powf(2.0))
            .sum();

        // Backward Pass
        mlp.parameters().iter().for_each(|param| {
            (**param).borrow_mut().grad = 0.0;
        });
        loss.backward();

        // Update
        mlp.parameters().iter().for_each(|param| {
            let grad;
            {
                grad = (**param).borrow().grad;
            }
            (**param).borrow_mut().value += -0.01 * grad;
        });
    }

    println!("--------------- [ RES ] ---------------");
    example_dataset.iter().for_each(|example| {
        let res = mlp.call(example.clone());
        println!("Res - {}", res.get(0).unwrap().item());
    });
}
