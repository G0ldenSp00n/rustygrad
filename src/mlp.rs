use rand::Rng;

use crate::RefValue;

#[derive(Debug, Clone)]
pub struct Neuron {
    w: Vec<RefValue>,
    b: RefValue,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut ref_values: Vec<RefValue> = Vec::new();
        for _ in 0..nin {
            ref_values.push(RefValue::new(rng.gen_range(-1.0..=1.0)));
        }
        Neuron {
            w: ref_values,
            b: RefValue::new(rng.gen_range(-1.0..=1.0)),
        }
    }
}

impl FnOnce<(Vec<RefValue>, ())> for Neuron {
    type Output = RefValue;
    extern "rust-call" fn call_once(self, value: (Vec<RefValue>, ())) -> Self::Output {
        let x = value.0;
        let out: RefValue = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(w, x)| w.clone() * x.clone())
            .sum();
        out + self.b
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(nin));
        }
        Layer { neurons }
    }
}

impl FnOnce<(Vec<RefValue>, ())> for Layer {
    type Output = Vec<RefValue>;
    extern "rust-call" fn call_once(self, value: (Vec<RefValue>, ())) -> Self::Output {
        let res: Vec<RefValue> = self
            .neurons
            .iter()
            .map(|neuron| neuron.clone()(value.0.clone(), ()))
            .collect();
        res
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &mut Vec<usize>) -> Self {
        let mut layers = Vec::new();
        let mut sz = vec![nin];
        sz.append(nouts);
        sz.windows(2).for_each(|sz| {
            layers.push(Layer::new(sz[0], sz[1]));
        });
        MLP { layers }
    }
}

impl FnOnce<(Vec<RefValue>, ())> for MLP {
    type Output = Vec<RefValue>;
    extern "rust-call" fn call_once(self, value: (Vec<RefValue>, ())) -> Self::Output {
        let mut val = value.0;
        self.layers.iter().for_each(|layer| {
            val = layer.clone()(val.clone(), ());
        });
        val
    }
}
