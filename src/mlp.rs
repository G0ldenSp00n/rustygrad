pub struct Neuron {
    w: Vec<f64>,
    b: f64,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        Neuron {
            w: Vec::new(),
            b: 0.0,
        }
    }
}

impl FnOnce<()> for Neuron {
    type Output = f64;
    extern "rust-call" fn call_once(self, _: ()) -> Self::Output {
        0.0
    }
}
