use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{continuous::integrator::Integrator, system::System};

pub mod integrator;

pub trait ContinuousSystem<Input, State, Output> {
    fn get_derivative(&self, time: f64, state: &State, input: &Input) -> State;

    fn get_output(&self, time: f64) -> Output;

    fn state(&self) -> &State;
    fn state_mut(&mut self) -> &mut State;

    fn max_timestep(&self) -> f64;

    fn with_integrator<Int>(
        self,
        integrator: Int,
    ) -> IntegratedSystem<Self, Int, Input, State, Output>
    where
        Self: Sized,
        Int: Integrator<Self, Input, State, Output>,
    {
        IntegratedSystem {
            system: self,
            integrator,
            last_time: 0.0,
            _dummy: PhantomData,
        }
    }
}

pub struct IntegratedSystem<Sys, Int, Input, State, Output> {
    system: Sys,
    integrator: Int,
    last_time: f64,
    _dummy: PhantomData<(Input, State, Output)>,
}

impl<Sys, Int, Input, State, Output> System<Input, Output>
    for IntegratedSystem<Sys, Int, Input, State, Output>
where
    Sys: ContinuousSystem<Input, State, Output>,
    Int: Integrator<Sys, Input, State, Output>,
{
    fn update(&mut self, time: f64, input: &Input) -> f64 {
        let max_dt = self.system.max_timestep();
        let dt = time - self.last_time;
        self.last_time = time;

        self.integrator.integrate(&mut self.system, time, dt, input);

        time + max_dt
    }

    fn get_output(&self, time: f64) -> Output {
        self.system.get_output(time)
    }
}

pub struct PureIntegrator<Data> {
    state: Data,
    output: Data,
    max_timestep: f64,
}

impl<Data> PureIntegrator<Data> {
    pub fn new(max_timestep: f64) -> Self
    where
        Data: Default,
    {
        Self {
            state: Data::default(),
            output: Data::default(),
            max_timestep,
        }
    }
}

impl<Data> ContinuousSystem<Data, Data, Data> for PureIntegrator<Data>
where
    Data: Clone,
{
    fn get_derivative(&self, _time: f64, _state: &Data, input: &Data) -> Data {
        input.clone()
    }

    fn get_output(&self, _time: f64) -> Data {
        self.output.clone()
    }

    fn state(&self) -> &Data {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Data {
        &mut self.state
    }

    fn max_timestep(&self) -> f64 {
        self.max_timestep
    }
}

/// A simple `System` represented by $\dot{x} = u$ and $y = x$
pub struct PureIntegratorSystem<Int, Data>(
    IntegratedSystem<PureIntegrator<Data>, Int, Data, Data, Data>,
);

impl<Int, Data> PureIntegratorSystem<Int, Data>
where
    Data: Default,
{
    pub fn new(max_timestep: f64, integrator: Int) -> Self {
        Self(IntegratedSystem {
            system: PureIntegrator::new(max_timestep),
            integrator,
            last_time: 0.0,
            _dummy: PhantomData,
        })
    }
}

impl<Int, Data> Deref for PureIntegratorSystem<Int, Data> {
    type Target = IntegratedSystem<PureIntegrator<Data>, Int, Data, Data, Data>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Int, Data> DerefMut for PureIntegratorSystem<Int, Data> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Const, Owned, Vector};

    pub type VecN<const N: usize, S = Owned<f64, Const<N>, Const<1>>> = Vector<f64, Const<N>, S>;

    #[test]
    fn test_pure_integrator_derivative_is_input() {
        const N: usize = 3;
        let sys = PureIntegrator::<VecN<N>>::new(0.1);
        let input = VecN::<N>::from_row_slice(&[1.0, -2.0, 3.5]);
        let state = VecN::<N>::from_row_slice(&[0.0, 0.0, 0.0]);
        let d = sys.get_derivative(0.0, &state, &input);
        assert_eq!(d, input);
    }

    #[test]
    fn test_pure_integrator_initial_state_and_output_are_zero() {
        const N: usize = 4;
        let sys = PureIntegrator::<VecN<N>>::new(0.2);
        assert_eq!(sys.state(), &VecN::<N>::zeros());
        assert_eq!(sys.get_output(0.0), VecN::<N>::zeros());
    }

    #[test]
    fn test_pure_integrator_state_mutation_reflects_in_state() {
        const N: usize = 2;
        let mut sys = PureIntegrator::<VecN<N>>::new(0.05);
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
        let sys = PureIntegrator::<VecN<N>>::new(0.123);
        assert_eq!(sys.max_timestep(), 0.123);
    }

    #[test]
    fn test_pure_integrator_system_new_initializes_inner_fields() {
        const N: usize = 3;
        // Int can be any type here since we don't call methods that require the Integrator bound
        let sys = PureIntegratorSystem::<(), VecN<N>>::new(0.3, ());
        // Accessing private fields is allowed within this module hierarchy for tests
        assert_eq!(sys.0.last_time, 0.0);
        assert_eq!(sys.0.system.max_timestep, 0.3);
        assert_eq!(sys.0.system.state, VecN::<N>::zeros());
        assert_eq!(sys.0.system.output, VecN::<N>::zeros());
    }
}
