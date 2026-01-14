use std::f32::consts::{PI, SQRT_2};

use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum EpisodeStatus {
    InProgress,
    Success,
    Crash,
    OutOfBounds,
    Timeout,
}

impl EpisodeStatus {
    fn as_str(&self) -> &'static str {
        match self {
            EpisodeStatus::InProgress => "in_progress",
            EpisodeStatus::Success => "success",
            EpisodeStatus::Crash => "crash",
            EpisodeStatus::OutOfBounds => "out_of_bounds",
            EpisodeStatus::Timeout => "timeout",
        }
    }
}

#[pyclass]
pub struct PyEnvironment {
    world: World,
    prev_potential: f32,
    steps: u32,
    tot_steps: u32,
    max_steps: u32,
    obs_dim: u32,
    act_dim: u32,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    pub fn new(max_steps: u32) -> Self {
        let world = World::new();
        Self {
            world,
            prev_potential: 0.0,
            steps: 0,
            tot_steps: 0,
            max_steps,
            obs_dim: 6,
            act_dim: 2,
        }
    }

    // world doesn't need a getter
    #[getter]
    pub fn get_steps(&self) -> u32 {
        self.steps
    }

    #[getter]
    pub fn get_tot_steps(&self) -> u32 {
        self.tot_steps
    }

    #[getter]
    pub fn get_max_steps(&self) -> u32 {
        self.max_steps
    }

    #[getter]
    pub fn get_prev_potential(&self) -> f32 {
        self.prev_potential
    }

    #[getter]
    pub fn get_obs_dim(&self) -> u32 {
        self.obs_dim
    }

    #[getter]
    pub fn get_act_dim(&self) -> u32 {
        self.act_dim
    }

    #[setter]
    pub fn set_tot_steps(&mut self, tot_steps: u32) {
        self.tot_steps = tot_steps;
    }

    pub fn reset(&mut self) -> PyResult<[f32; 6]> {
        self.world = World::new();
        self.steps = 0;

        let init_state = self._sample(); // Random valid state
        self.prev_potential = self.calculate_potential(
            init_state[0],
            init_state[1],
            init_state[2],
            init_state[3],
            init_state[4],
            init_state[5],
        );

        // Set the state to the rocket in the world
        let rocket = self
            .world
            .rigid_body_set
            .get_mut(self.world.rocket_body_handle)
            .unwrap();
        rocket.set_position(
            Isometry2::new(vector![init_state[0], init_state[1]], init_state[2]),
            true,
        );
        rocket.set_linvel(vector![init_state[3], init_state[4]], true);
        rocket.set_angvel(init_state[5], true);

        Ok(self._normalize(init_state))
    }

    pub fn step(&mut self, action: [f32; 2]) -> PyResult<([f32; 6], f32, bool, &'static str)> {
        // Run the physics step in rapier for the next state
        let [thrust, gimbal_angle] = action;
        self.world.apply_thruster_forces(thrust, gimbal_angle);
        self.world.step();
        self.steps += 1;

        let (x, y, theta) = self.world.get_rocket_state();
        let (vx, vy, omega) = self.world.get_rocket_dynamics();
        let next_state = self._normalize([x, y, theta, vx, vy, omega]);

        let (reason, done) = self._episode_status(x, y, theta, vx, vy, omega);
        let reward = self.calculate_reward(x, y, theta, vx, vy, omega);

        Ok((next_state, reward, done, reason))
    }

    fn calculate_potential(&self, x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> f32 {
        let [nx, ny, ntheta, nvx, nvy, nomega] = self._normalize([x, y, theta, vx, vy, omega]);

        // center is (max_x/2, 0) => potential should be min there
        let ndist = nx.powi(2) + ny.powi(2);
        let dist_score = 1.0 - (ndist / 2.0);

        // slow velocity preferred
        let speed = (nvx.powi(2) + nvy.powi(2)).sqrt();
        let speed_score = 1.0 - (speed / SQRT_2).min(1.0);

        // reward being upright and not spinning too much
        let angle_norm = (ntheta.powi(2) + nomega.powi(2).min(1.0)).sqrt(); // [0, sqrt2]
        let angle_score = 1.0 - (angle_norm / SQRT_2);

        let potential = 0.5 * dist_score + 0.2 * angle_score + 0.3 * speed_score;

        100.0 * potential
    }

    fn calculate_reward(
        &mut self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> f32 {
        let current_potential = self.calculate_potential(x, y, theta, vx, vy, omega);
        let shaping_reward = current_potential - self.prev_potential;
        self.prev_potential = current_potential;

        let mut terminal_reward = 0.0;
        let base_success = 100.0;

        if self._is_crash_landing(x, y, theta, vx, vy, omega) || self._is_oob(x, y) {
            terminal_reward = -base_success;
        } else if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            let ndx = (2.0 * x - MAX_POS_X) / MAX_POS_X;
            terminal_reward = base_success * (-2.0 * ndx.powi(2)).exp(); // gaussian reward
        }

        let time_penalty = 5e-3;
        shaping_reward + terminal_reward - time_penalty
    }

    fn _normalize(&self, obs: [f32; 6]) -> [f32; 6] {
        let nx = (2.0 * obs[0] - MAX_POS_X) / MAX_POS_X;
        let ny = (obs[1] - _MIN_POS_Y) / (MAX_POS_Y - _MIN_POS_Y);
        let ntheta = obs[2] / PI;

        let nvx = (2.0 * obs[3]) / MAX_POS_X;
        let nvy = obs[4] / (MAX_POS_Y - _MIN_POS_Y);
        let nomega = obs[5] / PI;

        [nx, ny, ntheta, nvx, nvy, nomega]
    }

    fn _sample(&self) -> [f32; 6] {
        let mut rng = rand::rng();

        let center_x = MAX_POS_X / 2.0;

        let (spawn_width, box_bottom, box_top) = if self.tot_steps < 100_000 {
            (0.0, 5.0, 5.0)
        } else if self.tot_steps < 250_000 {
            (5.0, 10.0, 20.0)
        } else if self.tot_steps < 500_000 {
            (10.0, 15.0, 30.0)
        } else if self.tot_steps < 1_000_000 {
            (20.0, 20.0, 35.0)
        } else {
            (35.0, 20.0, 42.0)
        };

        let box_left = center_x - spawn_width;
        let box_right = center_x + spawn_width;

        let start_x: f32 = rng.random_range(box_left..=box_right);
        let start_y: f32 = rng.random_range(box_bottom..=box_top);
        let start_angle: f32 = rng.random_range(-MAX_ANGLE_DEFLECTION..=MAX_ANGLE_DEFLECTION);

        // From the implememtation in base/src/world.rs
        // rotation is prevented when dragging, and velocities
        // is forcefully set to 0 when the drag ends
        // All starting sequences must have v & omega = 0

        [start_x, start_y, start_angle, 0.0, 0.0, 0.0]
    }

    pub fn sample(&self) -> PyResult<[f32; 6]> {
        Ok(self._sample())
    }

    fn _is_crash_landing(&self, _x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> bool {
        let landed = y <= _MIN_POS_Y;
        let bad_angle = theta.abs() > MAX_LANDING_ANGLE;
        let fast_land = vy.abs() > MAX_LANDING_VY;
        let fast_horiz = vx.abs() > MAX_LANDING_VX;
        let fast_spin = omega.abs() > MAX_LANDING_ANGULAR_VELOCITY;

        landed && (bad_angle || fast_land || fast_horiz || fast_spin)
    }

    fn _is_successful_landing(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> bool {
        let landed = y <= _MIN_POS_Y;
        let in_x_range = x > 0.0 && x < MAX_POS_X;
        let gentle_angle = theta.abs() <= MAX_LANDING_ANGLE;
        let gentle_vy = vy.abs() <= MAX_LANDING_VY;
        let gentle_vx = vx.abs() <= MAX_LANDING_VX;
        let gentle_omega = omega.abs() <= MAX_LANDING_ANGULAR_VELOCITY;

        landed && in_x_range && gentle_angle && gentle_vy && gentle_vx && gentle_omega
    }

    fn _is_oob(&self, x: f32, y: f32) -> bool {
        x <= 0.0 || x >= MAX_POS_X || y >= MAX_POS_Y
    }

    fn _episode_status(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> (&'static str, bool) {
        let status = if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            EpisodeStatus::Success
        } else if self._is_crash_landing(x, y, theta, vx, vy, omega) {
            EpisodeStatus::Crash
        } else if self._is_oob(x, y) {
            EpisodeStatus::OutOfBounds
        } else if self.steps >= self.max_steps {
            EpisodeStatus::Timeout
        } else {
            EpisodeStatus::InProgress
        };
        (status.as_str(), status != EpisodeStatus::InProgress)
    }

    pub fn render_info(&self, py: Python) -> PyResult<PyObject> {
        let (rocket_x, rocket_y, rocket_angle) = self.world.get_rocket_state();

        let state_dict = PyDict::new_bound(py);
        state_dict.set_item("rocket_x", rocket_x)?;
        state_dict.set_item("rocket_y", rocket_y)?;
        state_dict.set_item("rocket_angle", rocket_angle)?;

        Ok(state_dict.into())
    }
}

#[pymodule]
fn gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironment>()?;
    Ok(())
}
