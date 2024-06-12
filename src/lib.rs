#![feature(unboxed_closures)]
#![feature(fn_traits)]

pub mod mlp;

static COUNTER: AtomicUsize = AtomicUsize::new(1);

use std::{
    borrow::{Borrow, BorrowMut},
    cell::{Cell, RefCell},
    collections::HashSet,
    ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub},
    rc::Rc,
    sync::atomic::AtomicUsize,
};

use ordered_float::{Float, NotNan};

#[derive(Debug, Clone)]
enum Op {
    Add(RefValue, RefValue),
    Mul(RefValue, RefValue),
    Tanh(RefValue),
    Exp(RefValue),
    Relu(RefValue),
    Pow(RefValue, f64),
}

#[derive(Debug, Clone)]
pub struct Value {
    id: usize,
    pub value: f64,
    grad: f64,
    prev: Option<Box<Op>>,
}

#[derive(Debug, Clone)]
pub struct RefValue(Rc<RefCell<Value>>);

impl RefValue {
    pub fn new(value: f64) -> Self {
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value,
            grad: 0.0,
            prev: None,
        })))
    }

    fn calc_grad(self) -> Self {
        let self_unwrapped = (*self).borrow();
        if let Some(prev_val) = self_unwrapped.prev.clone() {
            match *prev_val {
                Op::Add(val, val2) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    let mut val2_unwraped = (*val2).borrow_mut();
                    val_unwrapped.grad = 1.0 * self_unwrapped.grad;
                    val2_unwraped.grad = 1.0 * self_unwrapped.grad;
                    self.clone()
                }
                Op::Mul(val, val2) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    let mut val2_unwraped = (*val2).borrow_mut();

                    val_unwrapped.grad = val2_unwraped.value * self_unwrapped.grad;
                    val2_unwraped.grad = val_unwrapped.value * self_unwrapped.grad;
                    self.clone()
                }
                Op::Tanh(val) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    val_unwrapped.grad +=
                        (1.0 - (self_unwrapped.value.powi(2))) * self_unwrapped.grad;
                    self.clone()
                }
                Op::Exp(val) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    val_unwrapped.grad += self_unwrapped.value * self_unwrapped.grad;
                    self.clone()
                }
                Op::Pow(val, rhs) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    val_unwrapped.grad +=
                        rhs * (val_unwrapped.value.powf(rhs - 1.0)) * self_unwrapped.grad;
                    self.clone()
                }
                Op::Relu(val) => {
                    let mut val_unwrapped = (*val).borrow_mut();
                    let value;
                    if self_unwrapped.value > 0.0 {
                        value = 1.0;
                    } else {
                        value = 0.0;
                    }
                    val_unwrapped.grad += value * self_unwrapped.grad;
                    self.clone()
                }
            }
        } else {
            self.clone()
        }
    }

    fn backward_internal(self) -> Self {
        let mut visited: HashSet<usize> = HashSet::new();
        let id;
        {
            let self_unwrapped = (*self).borrow_mut();
            id = self_unwrapped.id;
        }

        if !visited.contains(&id) {
            visited.insert(id);

            let res = self.calc_grad();
            let prev;
            {
                let res_unwrapped = (*res).borrow();
                prev = res_unwrapped.prev.clone();
            }
            if let Some(prev_val) = prev {
                match *prev_val {
                    Op::Add(val, val2) => {
                        val.backward_internal();
                        val2.backward_internal();
                        res.clone()
                    }
                    Op::Mul(val, val2) => {
                        val.backward_internal();
                        val2.backward_internal();
                        res.clone()
                    }
                    Op::Tanh(val) => {
                        val.backward_internal();
                        res.clone()
                    }
                    Op::Exp(val) => {
                        val.backward_internal();
                        res.clone()
                    }
                    Op::Pow(val, _) => {
                        val.backward_internal();
                        res.clone()
                    }
                    Op::Relu(val) => {
                        val.backward_internal();
                        res.clone()
                    }
                }
            } else {
                res.clone()
            }
        } else {
            self.clone()
        }
    }

    pub fn backward(self) -> Self {
        {
            let mut self_unwrapped = (*self).borrow_mut();
            self_unwrapped.grad = 1.0;
        }
        self.backward_internal()
    }

    pub fn tanh(self) -> Self {
        let value;
        {
            let self_unwrapped = (*self).borrow();
            value = self_unwrapped.value.tanh();
        }
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value,
            grad: 0.0,
            prev: Some(Box::new(Op::Tanh(self))),
        })))
    }

    pub fn exp(self) -> Self {
        let value;
        {
            let self_unwrapped = (*self).borrow();
            value = self_unwrapped.value.exp();
        }
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value,
            grad: 0.0,
            prev: Some(Box::new(Op::Exp(self))),
        })))
    }

    pub fn powf(self, rhs: f64) -> Self {
        let value;
        {
            let self_unwrapped = (*self).borrow();
            value = self_unwrapped.value.powf(rhs);
        }
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value,
            grad: 0.0,
            prev: Some(Box::new(Op::Pow(self, rhs))),
        })))
    }

    pub fn relu(self) -> Self {
        let mut value = 0.0;
        {
            let self_unwrapped = (*self).borrow();
            if self_unwrapped.value >= 0.0 {
                value = self_unwrapped.value;
            }
        }
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value,
            grad: 0.0,
            prev: Some(Box::new(Op::Relu(self))),
        })))
    }
}

impl Neg for RefValue {
    type Output = RefValue;
    fn neg(self: RefValue) -> Self::Output {
        self * -1
    }
}

impl Sub<f64> for RefValue {
    type Output = RefValue;
    fn sub(self, rhs: f64) -> Self::Output {
        self - RefValue::new(rhs)
    }
}

impl Sub<RefValue> for f64 {
    type Output = RefValue;
    fn sub(self, rhs: RefValue) -> Self::Output {
        RefValue::new(self) - rhs
    }
}

impl Sub<i64> for RefValue {
    type Output = RefValue;
    fn sub(self, rhs: i64) -> Self::Output {
        self - (rhs as f64)
    }
}

impl Sub<RefValue> for i64 {
    type Output = RefValue;
    fn sub(self, rhs: RefValue) -> Self::Output {
        (self as f64) - rhs
    }
}

impl Sub<RefValue> for RefValue {
    type Output = RefValue;
    fn sub(self, rhs: RefValue) -> Self::Output {
        self + (-rhs)
    }
}

impl Add<f64> for RefValue {
    type Output = RefValue;
    fn add(self, rhs: f64) -> Self::Output {
        let rhs = RefValue::new(rhs);
        self + rhs
    }
}

impl Add<RefValue> for f64 {
    type Output = RefValue;
    fn add(self, rhs: RefValue) -> Self::Output {
        rhs + self
    }
}

impl Add<i64> for RefValue {
    type Output = RefValue;
    fn add(self, rhs: i64) -> Self::Output {
        self + (rhs as f64)
    }
}

impl Add<RefValue> for i64 {
    type Output = RefValue;
    fn add(self, rhs: RefValue) -> Self::Output {
        rhs + self
    }
}

impl Add<RefValue> for RefValue {
    type Output = RefValue;
    fn add(self, rhs: RefValue) -> Self::Output {
        let val = (*self).borrow();
        let val2 = (*rhs).borrow();
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value: val.value + val2.value,
            grad: 0.0,
            prev: Some(Box::new(Op::Add(self.clone(), rhs.clone()))),
        })))
    }
}

impl Mul<f64> for RefValue {
    type Output = RefValue;
    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = RefValue::new(rhs);
        self * rhs
    }
}

impl Mul<RefValue> for f64 {
    type Output = RefValue;
    fn mul(self, rhs: RefValue) -> Self::Output {
        rhs * self
    }
}

impl Mul<i64> for RefValue {
    type Output = RefValue;
    fn mul(self, rhs: i64) -> Self::Output {
        let rhs = RefValue::new(rhs as f64);
        self * rhs
    }
}

impl Mul<RefValue> for i64 {
    type Output = RefValue;
    fn mul(self, rhs: RefValue) -> Self::Output {
        rhs * self
    }
}

impl Mul<RefValue> for RefValue {
    type Output = RefValue;
    fn mul(self, rhs: RefValue) -> Self::Output {
        let val = (*self).borrow();
        let val2 = (*rhs).borrow();
        RefValue(Rc::new(RefCell::new(Value {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            value: val.value * val2.value,
            grad: 0.0,
            prev: Some(Box::new(Op::Mul(self.clone(), rhs.clone()))),
        })))
    }
}

impl Div<RefValue> for RefValue {
    type Output = RefValue;
    fn div(self, rhs: RefValue) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

impl Deref for RefValue {
    type Target = RefCell<Value>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
