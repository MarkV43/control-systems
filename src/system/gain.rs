use nalgebra::{Const, OVector, Storage};

use crate::utils::VecN;

use super::System;

pub struct Gain<const N: usize> {
    gain: f64,
    output: VecN<N>,
}

impl<const N: usize> System<N, N> for Gain<N> {
    fn update<S>(&mut self, _: f64, input: &VecN<N, S>) -> f64
    where
        S: Storage<f64, Const<N>>,
    {
        self.output.copy_from(&(self.gain * input));
        f64::INFINITY
    }

    fn get_output(&self, _time: f64) -> &OVector<f64, Const<N>> {
        &self.output
    }
}

impl<const N: usize> Gain<N> {
    pub fn new(gain: f64) -> Self {
        Self {
            gain,
            output: OVector::zeros_generic(Const, Const),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Param;
    use nalgebra::vector;

    #[test]
    fn test_gain() {
        let mut sys = Gain::new(2.0);

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
        let mut sys = Gain::new(2.0);

        let input = Param::new(VecN::<1>::from_column_slice(&[3.]));

        let mut count = 0;
        sys.simulate(0.4, 0.1, input, |_| count += 1);

        assert_eq!(count, 4);
    }
}
