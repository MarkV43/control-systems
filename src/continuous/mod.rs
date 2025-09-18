use std::ops::{Deref, DerefMut};

use nalgebra::{Const, Storage};

use crate::{continuous::integrator::Integrator, system::System, utils::VecN};

pub mod integrator;

pub trait ContinuousSystem<const INPUTS: usize, const STATES: usize, const OUTPUTS: usize> {
    fn get_derivative<Ss, Si>(
        &self,
        time: f64,
        state: &VecN<STATES, Ss>,
        input: &VecN<INPUTS, Si>,
    ) -> VecN<STATES>
    where
        Ss: Storage<f64, Const<STATES>>,
        Si: Storage<f64, Const<INPUTS>>;

    fn get_output(&self, time: f64) -> VecN<OUTPUTS>;

    fn state(&self) -> &VecN<STATES>;
    fn state_mut(&mut self) -> &mut VecN<STATES>;

    fn max_timestep(&self) -> f64;

    fn with_integrator<Int>(
        self,
        integrator: Int,
    ) -> IntegratedSystem<Self, Int, INPUTS, STATES, OUTPUTS>
    where
        Self: Sized,
        Int: Integrator<Self, INPUTS, STATES, OUTPUTS>,
    {
        IntegratedSystem {
            system: self,
            integrator,
            last_time: 0.0,
        }
    }
}

pub struct IntegratedSystem<
    Sys,
    Int,
    const INPUTS: usize,
    const STATES: usize,
    const OUTPUTS: usize,
> {
    system: Sys,
    integrator: Int,
    last_time: f64,
}

impl<Sys, Int, const INPUTS: usize, const STATES: usize, const OUTPUTS: usize>
    System<INPUTS, OUTPUTS> for IntegratedSystem<Sys, Int, INPUTS, STATES, OUTPUTS>
where
    Sys: ContinuousSystem<INPUTS, STATES, OUTPUTS>,
    Int: Integrator<Sys, INPUTS, STATES, OUTPUTS>,
{
    fn update<S>(&mut self, time: f64, input: &VecN<INPUTS, S>) -> f64
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let max_dt = self.system.max_timestep();
        let dt = time - self.last_time;
        self.last_time = time;

        self.integrator.integrate(&mut self.system, time, dt, input);

        time + max_dt
    }

    fn get_output(&self, time: f64) -> VecN<OUTPUTS> {
        self.system.get_output(time)
    }
}

pub struct PureIntegrator<const N: usize> {
    state: VecN<N>,
    output: VecN<N>,
    max_timestep: f64,
}

impl<const N: usize> PureIntegrator<N> {
    pub fn new(max_timestep: f64) -> Self {
        Self {
            state: VecN::zeros(),
            output: VecN::zeros(),
            max_timestep,
        }
    }
}

impl<const N: usize> ContinuousSystem<N, N, N> for PureIntegrator<N> {
    fn get_derivative<Ss, Si>(
        &self,
        _time: f64,
        _state: &VecN<N, Ss>,
        input: &VecN<N, Si>,
    ) -> VecN<N>
    where
        Ss: Storage<f64, Const<N>>,
        Si: Storage<f64, Const<N>>,
    {
        input.clone_owned()
    }

    fn get_output(&self, _time: f64) -> VecN<N> {
        self.output.clone_owned()
    }

    fn state(&self) -> &VecN<N> {
        &self.state
    }

    fn state_mut(&mut self) -> &mut VecN<N> {
        &mut self.state
    }

    fn max_timestep(&self) -> f64 {
        self.max_timestep
    }
}

/// A simple `System` represented by $\dot{x} = u$ and $y = x$
pub struct PureIntegratorSystem<Int, const N: usize>(
    IntegratedSystem<PureIntegrator<N>, Int, N, N, N>,
);

impl<Int, const N: usize> PureIntegratorSystem<Int, N> {
    pub fn new(max_timestep: f64, integrator: Int) -> Self {
        Self(IntegratedSystem {
            system: PureIntegrator::new(max_timestep),
            integrator,
            last_time: 0.0,
        })
    }
}

impl<Int, const N: usize> Deref for PureIntegratorSystem<Int, N> {
    type Target = IntegratedSystem<PureIntegrator<N>, Int, N, N, N>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Int, const N: usize> DerefMut for PureIntegratorSystem<Int, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_integrator_derivative_is_input() {
        const N: usize = 3;
        let sys = PureIntegrator::<N>::new(0.1);
        let input = VecN::<N>::from_row_slice(&[1.0, -2.0, 3.5]);
        let state = VecN::<N>::from_row_slice(&[0.0, 0.0, 0.0]);
        let d = sys.get_derivative(0.0, &state, &input);
        assert_eq!(d, input);
    }

    #[test]
    fn test_pure_integrator_initial_state_and_output_are_zero() {
        const N: usize = 4;
        let sys = PureIntegrator::<N>::new(0.2);
        assert_eq!(sys.state(), &VecN::<N>::zeros());
        assert_eq!(sys.get_output(0.0), VecN::<N>::zeros());
    }

    #[test]
    fn test_pure_integrator_state_mutation_reflects_in_state() {
        const N: usize = 2;
        let mut sys = PureIntegrator::<N>::new(0.05);
        {
            let s = sys.state_mut();
            s.copy_from(&VecN::<N>::from_row_slice(&[1.5, -0.5]));
        }
        assert_eq!(sys.state(), &VecN::<N>::from_row_slice(&[1.5, -0.5]));
        // Output is independent from state and remains zero unless updated externally
        assert_eq!(sys.get_output(0.0), VecN::<N>::zeros());
    }

    #[test]
    fn test_pure_integrator_max_timestep() {
        const N: usize = 1;
        let sys = PureIntegrator::<N>::new(0.123);
        assert_eq!(sys.max_timestep(), 0.123);
    }

    #[test]
    fn test_pure_integrator_system_new_initializes_inner_fields() {
        const N: usize = 3;
        // Int can be any type here since we don't call methods that require the Integrator bound
        let sys = PureIntegratorSystem::<(), N>::new(0.3, ());
        // Accessing private fields is allowed within this module hierarchy for tests
        assert_eq!(sys.0.last_time, 0.0);
        assert_eq!(sys.0.system.max_timestep, 0.3);
        assert_eq!(sys.0.system.state, VecN::<N>::zeros());
        assert_eq!(sys.0.system.output, VecN::<N>::zeros());
    }
}
