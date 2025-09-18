use nalgebra::{Const, Storage};

use crate::{discrete::holder::Holder, system::System, utils::VecN};

pub mod holder;

pub trait DiscreteSystem<const INPUTS: usize, const STATES: usize, const OUTPUTS: usize> {
    fn next_state<Ss, Si>(
        &self,
        time: f64,
        state: &VecN<STATES, Ss>,
        input: &VecN<INPUTS, Si>,
    ) -> VecN<STATES>
    where
        Ss: Storage<f64, Const<STATES>>,
        Si: Storage<f64, Const<INPUTS>>;

    fn get_output(&self) -> &VecN<OUTPUTS>;

    fn state(&self) -> &VecN<STATES>;
    fn state_mut(&mut self) -> &mut VecN<STATES>;

    fn timestep(&self) -> f64;

    fn with_holder<Hol>(self, holder: Hol) -> HeldSystem<Self, Hol, INPUTS, STATES, OUTPUTS>
    where
        Self: Sized,
        Hol: Holder<OUTPUTS>,
    {
        HeldSystem {
            system: self,
            holder,
            last_time: 0.0,
        }
    }
}

pub struct HeldSystem<Sys, Hol, const INPUTS: usize, const STATES: usize, const OUTPUTS: usize> {
    system: Sys,
    holder: Hol,
    last_time: f64,
}

impl<Sys, Hol, const INPUTS: usize, const STATES: usize, const OUTPUTS: usize>
    System<INPUTS, OUTPUTS> for HeldSystem<Sys, Hol, INPUTS, STATES, OUTPUTS>
where
    Sys: DiscreteSystem<INPUTS, STATES, OUTPUTS>,
    Hol: Holder<OUTPUTS>,
{
    fn update<S>(&mut self, time: f64, input: &VecN<INPUTS, S>) -> f64
    where
        S: Storage<f64, Const<INPUTS>>,
    {
        let req_dt = self.system.timestep();
        let dt = time - self.last_time;

        if dt < req_dt {
            return self.last_time + dt;
        }

        // If the request time
        assert!(
            dt - req_dt < req_dt * 1e-5,
            "Requested event was not triggered"
        );

        *self.system.state_mut() = self.system.next_state(time, self.system.state(), input);
        self.holder.hold(time, self.system.get_output());

        self.last_time + 2.0 * dt
    }

    fn get_output(&self, time: f64) -> VecN<OUTPUTS> {
        self.holder.get_output(time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::holder::{FirstOrderHold, ImpulseHold, ZeroOrderHold};
    use crate::utils::VecN;

    // A minimal discrete system for tests:
    // - STATES = OUTPUTS = INPUTS = N
    // - next_state = state + input
    // - output = state
    struct MockDiscrete<const N: usize> {
        state: VecN<N>,
        step: f64,
    }

    impl<const N: usize> MockDiscrete<N> {
        fn new(step: f64) -> Self {
            Self {
                state: VecN::<N>::zeros(),
                step,
            }
        }
    }

    impl<const N: usize> DiscreteSystem<N, N, N> for MockDiscrete<N> {
        fn next_state<Ss, Si>(
            &self,
            _time: f64,
            state: &VecN<N, Ss>,
            input: &VecN<N, Si>,
        ) -> VecN<N>
        where
            Ss: Storage<f64, Const<N>>,
            Si: Storage<f64, Const<N>>,
        {
            state.clone_owned() + input.clone_owned()
        }

        fn get_output(&self) -> &VecN<N> {
            &self.state
        }

        fn state(&self) -> &VecN<N> {
            &self.state
        }

        fn state_mut(&mut self) -> &mut VecN<N> {
            &mut self.state
        }

        fn timestep(&self) -> f64 {
            self.step
        }
    }

    #[test]
    fn heldsystem_no_trigger_returns_last_time_plus_dt_and_does_not_update() {
        const N: usize = 1;
        let sys = MockDiscrete::<N>::new(0.2);
        let mut held = sys.with_holder(ZeroOrderHold::<N>::new());

        let input = VecN::<N>::from_row_slice(&[1.0]);

        // time < timestep -> no trigger, return last_time + dt
        let ret = held.update(0.1, &input);
        assert_eq!(ret, 0.1);

        // state must remain zero
        assert_eq!(held.system.state(), &VecN::<N>::zeros());

        // holder output must still be zeros
        assert_eq!(held.get_output(0.1), VecN::<N>::zeros());
    }

    #[test]
    fn heldsystem_trigger_updates_state_and_holder_zoh() {
        const N: usize = 2;
        let sys = MockDiscrete::<N>::new(0.1);
        let mut held = sys.with_holder(ZeroOrderHold::<N>::new());

        let input = VecN::<N>::from_row_slice(&[0.5, -0.5]);

        // trigger at exactly timestep
        let ret = held.update(0.1, &input);

        // impl returns last_time + 2.0 * dt; last_time starts 0.0
        assert_eq!(ret, 0.0 + 2.0 * 0.1);

        // internal system state updated to previous state + input = input
        assert_eq!(held.system.state(), &input);

        // holder should have stored the system output (which equals the new state)
        assert_eq!(held.get_output(0.1), input);
    }

    #[test]
    #[should_panic(expected = "Requested event was not triggered")]
    fn heldsystem_asserts_when_dt_not_close_to_req_dt() {
        const N: usize = 1;
        let sys = MockDiscrete::<N>::new(0.1);
        let mut held = sys.with_holder(ZeroOrderHold::<N>::new());
        let input = VecN::<N>::from_row_slice(&[1.0]);

        // dt = 0.25, req_dt = 0.1 -> assertion should fire
        let _ = held.update(0.25, &input);
    }

    #[test]
    fn first_order_hold_interpolates_between_two_samples() {
        const N: usize = 2;
        let mut foh = FirstOrderHold::<N>::new();

        let s0 = VecN::<N>::from_row_slice(&[0.0, 0.0]);
        let s1 = VecN::<N>::from_row_slice(&[2.0, 4.0]);

        foh.hold(0.0, &s0);
        foh.hold(1.0, &s1);

        // midpoint interpolation
        let mid_expected = VecN::<N>::from_row_slice(&[1.0, 2.0]);
        assert_eq!(foh.get_output(0.5), mid_expected);

        // clamping outside interval
        assert_eq!(foh.get_output(-1.0), s0);
        assert_eq!(foh.get_output(2.0), s1);
    }

    #[test]
    fn impulse_hold_only_returns_at_instant() {
        const N: usize = 1;
        let mut ih = ImpulseHold::<N>::new();
        let sample = VecN::<N>::from_row_slice(&[3.0]);

        ih.hold(2.0, &sample);
        assert_eq!(ih.get_output(2.0), sample);
        assert_eq!(ih.get_output(2.0 + 1e-12), VecN::<N>::zeros());
    }
}
