use std::ops::Mul;

use super::System;

pub struct Gain<Data> {
    gain: f64,
    output: Data,
}

impl<Data> System<Data, Data> for Gain<Data>
where
    Data: Clone,
    for<'a> &'a Data: Mul<f64, Output = Data>,
{
    fn update(&mut self, _: f64, input: &Data) -> f64
    where
        Data: Clone,
    {
        self.output = input * self.gain;
        f64::INFINITY
    }

    fn get_output(&self, _time: f64) -> Data {
        self.output.clone()
    }
}

impl<Data> Gain<Data> {
    pub fn new(gain: f64) -> Self
    where
        Data: Default,
    {
        Self {
            gain,
            output: Data::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Param;
    use nalgebra::{vector, Const, Owned, Vector};
    
    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    #[test]
    fn test_gain() {
        let mut sys = Gain::<VecN<1>>::new(2.0);

        assert_eq!(sys.output[0], 0.0);
        sys.update(0.1, &vector![3.]);
        assert_eq!(sys.output[0], 6.0);
        sys.update(0.2, &vector![2.]);
        assert_eq!(sys.output[0], 4.0);
        sys.update(0.3, &vector![1.]);
        assert_eq!(sys.output[0], 2.0);
    }

    #[test]
    fn test_gain_timestep() {
        let mut sys = Gain::<VecN<1>>::new(2.0);

        let input = Param::new(VecN::<1>::from_column_slice(&[3.]));

        let mut count = 0;
        sys.simulate(0.4, 0.1, input, |_| count += 1);

        assert_eq!(count, 4);
    }
}
