use rand::Rng;

use crate::RefValue;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub w: Vec<RefValue>,
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

    pub fn parameters(&self) -> Vec<RefValue> {
        let mut params = vec![self.b.clone()];
        params.append(&mut self.w.clone());
        params
    }

    pub fn call(&self, x: Vec<RefValue>) -> RefValue {
        let out: RefValue = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(w, x)| w.clone() * x.clone())
            .sum();
        out + self.b.clone().tanh()
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(nin));
        }
        Layer { neurons }
    }

    pub fn parameters(&self) -> Vec<RefValue> {
        let mut params = Vec::new();
        self.neurons
            .iter()
            .for_each(|neuron| params.append(&mut neuron.parameters()));
        params
    }

    pub fn call(&self, x: Vec<RefValue>) -> Vec<RefValue> {
        let res: Vec<RefValue> = self
            .neurons
            .iter()
            .map(|neuron| neuron.call(x.clone()))
            .collect();
        res
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
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

    pub fn parameters(&self) -> Vec<RefValue> {
        let mut params = Vec::new();
        self.layers
            .iter()
            .for_each(|layer| params.append(&mut layer.parameters()));
        params
    }

    pub fn call(&self, x: Vec<RefValue>) -> Vec<RefValue> {
        let mut val = x.clone();
        self.layers.iter().for_each(|layer| {
            val = layer.call(val.clone());
        });
        val
    }
}
