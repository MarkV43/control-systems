use std::ops::{Add, Mul};

use crate::continuous::ContinuousSystem;

pub trait Integrator<Sys: ContinuousSystem<Input, State, Output>, Input, State, Output> {
    fn integrate(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &Input);
}

pub struct RectangularIntegrator;

impl<
    Sys: ContinuousSystem<Input, State, Output>,
    Input, State, Output
> Integrator<Sys, Input, State, Output> for RectangularIntegrator
where 
    for<'a> State: Mul<f64, Output = State> + Add<State, Output = State> + Add<&'a State, Output = State>
{
    fn integrate(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &Input) {
        let state = sys.state();
        let der = sys.get_derivative(t, state, input);

        *sys.state_mut() = der * dt + state;
    }
}

pub struct TrapezoidalIntegrator<State> {
    previous_derivative: State,
}

impl<
    Sys: ContinuousSystem<Input, State, Output>,
    Input, State, Output
> Integrator<Sys, Input, State, Output> for TrapezoidalIntegrator<State>
where
    for<'a> State: Clone + Mul<f64, Output = State> + Add<State, Output = State> + Add<&'a State, Output = State>,
{
    fn integrate(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &Input) {
        let state = sys.state();
        let der = sys.get_derivative(t, state, input);

        *sys.state_mut() = (der.clone() + &self.previous_derivative) * dt * 0.5;
        self.previous_derivative = der;
    }
}

pub struct RungeKutta4;

impl<
    Sys: ContinuousSystem<Input, State, Output>,
    Input, State, Output
> Integrator<Sys, Input, State, Output> for RungeKutta4
where
    for<'a> State: Mul<f64, Output = State> + Add<State, Output = State> + Add<&'a State, Output = State>,
    for<'a> &'a State: Mul<f64, Output = State>
{
    fn integrate(&mut self, sys: &mut Sys, t: f64, dt: f64, input: &Input) {
        let state = sys.state();

        let h2 = dt * 0.5;

        let k1 = sys.get_derivative(t, state, input);
        let k2 = sys.get_derivative(t + h2, &(&k1 * h2 + state), input);
        let k3 = sys.get_derivative(t + h2, &(&k2 * h2 + state), input);
        let k4 = sys.get_derivative(t + dt, &(&k3 * dt + state), input);
        *sys.state_mut() = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0) + state;
    }
}
