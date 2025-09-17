use std::f64;

use nalgebra::{Const, OVector, Storage, Vector};

pub mod cloop;
pub mod gain;
pub mod series;

use crate::utils::Param;

/// A model for any kind of system with `INPUTS` inputs and `OUTPUTS` outputs.
///
/// Instead of the normal system implementations, where the overall simulation
/// has a single timestep for all its components, making it hard for fast system
/// to work in tandem with slower system, this implementation uses an event
/// request system, where each system requests an instant for its next update.
///
/// This way, slower system will be updated less often, while faster systems can
/// have different frequencies, and still work without issue.
pub trait System<const INPUTS: usize, const OUTPUTS: usize> {
    /// Updates the system as if it were on instant `time` receiving inputs `input`.
    /// Returns the next instant the system will be updated at.
    fn update<S>(&mut self, time: f64, input: &Vector<f64, Const<INPUTS>, S>) -> f64
    where
        S: Storage<f64, Const<INPUTS>>;

    /// Returns the system's current output. Should be called after `update`ing the system.
    fn get_output(&self) -> &OVector<f64, Const<OUTPUTS>>;

    /// Simulates the system for a full `total_time` time units.
    fn simulate<F>(
        &mut self,
        total_time: f64,
        max_timestep: f64,
        mut input: Param<OVector<f64, Const<INPUTS>>>,
        mut callback: F,
    ) where
        F: FnMut(Sample<INPUTS, OUTPUTS>) -> (),
    {
        let mut time = 0.0;

        while time < total_time {
            input.update(time);
            let next_time = self.update(time, &*input);

            let sample = Sample {
                instant: time,
                input: (*input).clone_owned(),
                output: self.get_output().clone_owned(),
            };
            callback(sample);

            time = next_time.min(time + max_timestep);
        }
    }
}

pub struct Sample<const INPUTS: usize, const OUTPUTS: usize> {
    pub instant: f64,
    pub input: OVector<f64, Const<INPUTS>>,
    pub output: OVector<f64, Const<OUTPUTS>>,
}

/// A simple system that directly transfer the input to the output.
/// Its transfer function is represented by $F(s) = 1$.
pub struct UnitSystem<const N: usize> {
    output: OVector<f64, Const<N>>,
}

impl<const N: usize> System<N, N> for UnitSystem<N> {
    fn update<S>(&mut self, _: f64, input: &Vector<f64, Const<N>, S>) -> f64
    where
        S: Storage<f64, Const<N>>,
    {
        self.output.copy_from(input);
        f64::INFINITY
    }

    fn get_output(&self) -> &OVector<f64, Const<N>> {
        &self.output
    }
}

impl<const N: usize> Default for UnitSystem<N> {
    fn default() -> Self {
        Self {
            output: OVector::zeros_generic(Const, Const),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Param;

    #[test]
    fn test_max_timestep() {
        let mut sys = UnitSystem::default();

        let input = Param::new(OVector::<f64, Const<1>>::from_column_slice(&[3.]));

        let mut count = 0;
        sys.simulate(0.4, 0.1, input, |_| count += 1);

        assert_eq!(count, 4);
    }
}
