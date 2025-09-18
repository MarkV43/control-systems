use nalgebra::{Const, Storage};

use crate::{continuous::ContinuousSystem, utils::VecN};

pub trait Integrator<
    Sys: ContinuousSystem<INPUTS, STATES, OUTPUTS>,
    const INPUTS: usize,
    const STATES: usize,
    const OUTPUTS: usize,
>
{
    /// Integrate the system with the given input and return the output.
    fn integrate<S>(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &VecN<INPUTS, S>)
    where
        S: Storage<f64, Const<INPUTS>>;
}

pub struct RectangularIntegrator;

impl<
    Sys: ContinuousSystem<INPUTS, STATES, OUTPUTS>,
    const INPUTS: usize,
    const STATES: usize,
    const OUTPUTS: usize,
> Integrator<Sys, INPUTS, STATES, OUTPUTS> for RectangularIntegrator
{
    fn integrate<S>(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &VecN<INPUTS, S>)
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let state = *sys.state();
        let der = sys.get_derivative(t, &state, input);

        *sys.state_mut() = state + der * dt;
    }
}

pub struct TrapezoidalIntegrator<const N: usize> {
    previous_derivative: VecN<N>,
}

impl<
    Sys: ContinuousSystem<INPUTS, STATES, OUTPUTS>,
    const INPUTS: usize,
    const STATES: usize,
    const OUTPUTS: usize,
> Integrator<Sys, INPUTS, STATES, OUTPUTS> for TrapezoidalIntegrator<STATES>
{
    fn integrate<S>(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &VecN<INPUTS, S>)
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let state = *sys.state();
        let der = sys.get_derivative(t, &state, input);

        *sys.state_mut() = (self.previous_derivative + der) * dt * 0.5;
        self.previous_derivative = der;
    }
}

pub struct RungeKutta4;

impl<
    Sys: ContinuousSystem<INPUTS, STATES, OUTPUTS>,
    const INPUTS: usize,
    const STATES: usize,
    const OUTPUTS: usize,
> Integrator<Sys, INPUTS, STATES, OUTPUTS> for RungeKutta4
{
    fn integrate<S>(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &VecN<INPUTS, S>)
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let state = *sys.state();

        let h2 = dt * 0.5;

        let k1 = sys.get_derivative(t, &state, input);
        let k2 = sys.get_derivative(t + h2, &(state + k1 * h2), input);
        let k3 = sys.get_derivative(t + h2, &(state + k2 * h2), input);
        let k4 = sys.get_derivative(t + dt, &(state + k3 * dt), input);
        *sys.state_mut() = state + (k1 + 2.0 * &k2 + 2.0 * &k3 + k4) * (dt / 6.0);
    }
}
