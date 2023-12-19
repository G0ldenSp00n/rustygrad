use std::{
    collections::HashSet,
    ops::{Add, Mul},
};

use ordered_float::NotNan;

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
enum Op {
    Add(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct Value {
    value: NotNan<f64>,
    grad: NotNan<f64>,
    prev: Option<Box<Op>>,
}

impl Value {
    pub fn new(value: f64) -> Self {
        Value {
            value: NotNan::new(value).expect("Should be a Non-NAN Value"),
            grad: NotNan::new(0.0).expect("Should support 0.0"),
            prev: None,
        }
    }

    fn calc_grad(self) -> Self {
        if let Some(prev_val) = self.prev {
            match *prev_val {
                Op::Add(mut val, mut val2) => {
                    val.grad += NotNan::new(1.0).expect("Should support 1.0") * self.grad;
                    val2.grad += NotNan::new(1.0).expect("Should support 1.0") * self.grad;
                    Self {
                        value: self.value,
                        grad: self.grad,
                        prev: Some(Box::new(Op::Add(val, val2))),
                    }
                }
                Op::Mul(mut val, mut val2) => {
                    val.grad += val2.value * self.grad;
                    val2.grad += val.value * self.grad;
                    Self {
                        value: self.value,
                        grad: self.grad,
                        prev: Some(Box::new(Op::Mul(val, val2))),
                    }
                }
                Op::Tanh(mut val) => {
                    val.grad +=
                        NotNan::new(1.0).expect("Should support 1.0") - (self.value.powi(2));
                    Self {
                        value: self.value,
                        grad: self.grad,
                        prev: Some(Box::new(Op::Tanh(val))),
                    }
                }
            }
        } else {
            self
        }
    }

    fn backward_internal(self) -> Self {
        let mut visited: HashSet<Value> = HashSet::new();
        if !visited.contains(&self) {
            visited.insert(self.clone());

            let mut res = self.calc_grad();
            if let Some(prev_val) = res.prev {
                match *prev_val {
                    Op::Add(val, val2) => {
                        let val = val.backward_internal();
                        let val2 = val2.backward_internal();
                        res.prev = Some(Box::new(Op::Add(val, val2)));
                        res
                    }
                    Op::Mul(val, val2) => {
                        let val = val.backward_internal();
                        let val2 = val2.backward_internal();
                        res.prev = Some(Box::new(Op::Add(val, val2)));
                        res
                    }
                    Op::Tanh(val) => {
                        let val = val.backward_internal();
                        res.prev = Some(Box::new(Op::Tanh(val)));
                        res
                    }
                }
            } else {
                res
            }
        } else {
            self
        }
    }

    pub fn backward(mut self) -> Self {
        self.grad = NotNan::new(1.0).expect("Should support 1.0");
        self.backward_internal()
    }

    pub fn tanh(&self) -> Self {
        Value {
            value: NotNan::new(self.value.tanh()).unwrap(),
            grad: NotNan::new(0.0).expect("Should support 0.0"),
            prev: Some(Box::new(Op::Tanh(self.clone()))),
        }
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;
    fn add(self, rhs: &'b Value) -> Self::Output {
        Value {
            value: self.value + rhs.value,
            grad: NotNan::new(0.0).expect("Should support 0.0"),
            prev: Some(Box::new(Op::Add(self.clone(), rhs.clone()))),
        }
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;
    fn mul(self, rhs: &'b Value) -> Self::Output {
        Value {
            value: self.value * rhs.value,
            grad: NotNan::new(0.0).expect("Should support 0.0"),
            prev: Some(Box::new(Op::Mul(self.clone(), rhs.clone()))),
        }
    }
}
