use std::f32::consts::{PI, SQRT_2};

use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;
use rayon::prelude::*;

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
    prev_potentials: Vec<f32>,
    steps: u32,
    max_steps: u32,
    obs_dim: u32,
    act_dim: u32,
    group_size: usize,
    spawn_x_range: (f32, f32),
    spawn_y_range: (f32, f32),
    spawn_angle_range: (f32, f32),
}

#[pymethods]
impl PyEnvironment {
    #[new]
    pub fn new(max_steps: u32, group_size: usize) -> Self {
        let world = World::new(group_size);
        Self {
            world,
            prev_potentials: vec![0.0; group_size],
            steps: 0,
            max_steps,
            obs_dim: 6,
            act_dim: 2,
            group_size,
            spawn_x_range: (MAX_POS_X / 2.0, MAX_POS_X / 2.0),
            spawn_y_range: (5.0, 5.0),
            spawn_angle_range: (0.0, 0.0),
        }
    }

    #[getter]
    pub fn get_steps(&self) -> u32 {
        self.steps
    }

    #[getter]
    pub fn get_max_steps(&self) -> u32 {
        self.max_steps
    }

    #[getter]
    pub fn get_obs_dim(&self) -> u32 {
        self.obs_dim
    }

    #[getter]
    pub fn get_act_dim(&self) -> u32 {
        self.act_dim
    }

    #[getter]
    pub fn get_group_size(&self) -> usize {
        self.group_size
    }

    pub fn set_spawn_x_range(&mut self, min: f32, max: f32) {
        self.spawn_x_range = (min, max);
    }

    pub fn set_spawn_y_range(&mut self, min: f32, max: f32) {
        self.spawn_y_range = (min, max);
    }

    pub fn set_spawn_angle_range(&mut self, min: f32, max: f32) {
        self.spawn_angle_range = (min, max);
    }

    pub fn reset(&mut self) -> PyResult<Vec<[f32; 6]>> {
        self.world = World::new(self.group_size);
        self.steps = 0;

        // All rockets start at the same sample for simplicity in parallel batching,
        // but we can also sample independently if needed.
        let init_state = self._sample();
        let potential = self._calculate_potential(
            init_state[0],
            init_state[1],
            init_state[2],
            init_state[3],
            init_state[4],
            init_state[5],
        );
        self.prev_potentials = vec![potential; self.group_size];

        for &handle in &self.world.rocket_handles {
            let rocket = self.world.rigid_body_set.get_mut(handle).unwrap();
            rocket.set_position(
                Isometry2::new(vector![init_state[0], init_state[1]], init_state[2]),
                true,
            );
            rocket.set_linvel(vector![init_state[3], init_state[4]], true);
            rocket.set_angvel(init_state[5], true);
        }

        let normalized = self._normalize(init_state);
        Ok(vec![normalized; self.group_size])
    }

    pub fn step(
        &mut self,
        actions: Vec<[f32; 2]>,
    ) -> PyResult<(Vec<[f32; 6]>, Vec<f32>, Vec<bool>, Vec<&'static str>)> {
        self.world.apply_multi_thruster_forces(&actions);
        self.world.step();
        self.steps += 1;

        let states = self.world.get_multi_rocket_state();
        let dynamics = self.world.get_multi_rocket_dynamics();

        let results: Vec<([f32; 6], f32, f32, bool, &'static str)> = (0..self.group_size)
            .into_par_iter()
            .map(|i| {
                let (x, y, theta) = states[i];
                let (vx, vy, omega) = dynamics[i];

                let obs = self._normalize([x, y, theta, vx, vy, omega]);
                let (reason, done) = self._episode_status(x, y, theta, vx, vy, omega);
                let (reward, potential) = self._calculate_reward(i, x, y, theta, vx, vy, omega);

                (obs, reward, potential, done, reason)
            })
            .collect();

        let mut next_obs = Vec::with_capacity(self.group_size);
        let mut rewards = Vec::with_capacity(self.group_size);
        let mut dones = Vec::with_capacity(self.group_size);
        let mut reasons = Vec::with_capacity(self.group_size);

        for (i, (obs, reward, potential, done, reason)) in results.into_iter().enumerate() {
            next_obs.push(obs);
            rewards.push(reward);
            dones.push(done);
            reasons.push(reason);
            self.prev_potentials[i] = potential;
        }

        Ok((next_obs, rewards, dones, reasons))
    }

    fn _calculate_potential(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> f32 {
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

    fn _calculate_reward(
        &self,
        idx: usize,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> (f32, f32) {
        let current_potential = self._calculate_potential(x, y, theta, vx, vy, omega);
        let shaping_reward = current_potential - self.prev_potentials[idx];

        let mut terminal_reward = 0.0;
        let base_success = 100.0;

        if self._is_crash_landing(x, y, theta, vx, vy, omega) || self._is_oob(x, y) {
            terminal_reward = -base_success;
        } else if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            let ndx = (2.0 * x - MAX_POS_X) / MAX_POS_X;
            terminal_reward = base_success * (-2.0 * ndx.powi(2)).exp();
        }

        let time_penalty = 5e-3;
        (
            shaping_reward + terminal_reward - time_penalty,
            current_potential,
        )
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
        let start_x: f32 = rng.random_range(self.spawn_x_range.0..=self.spawn_x_range.1);
        let start_y: f32 = rng.random_range(self.spawn_y_range.0..=self.spawn_y_range.1);
        let start_angle: f32 =
            rng.random_range(self.spawn_angle_range.0..=self.spawn_angle_range.1);
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
        let states = self.world.get_multi_rocket_state();
        let (rocket_x, rocket_y, rocket_angle) = states[0]; // Render the first one for simplicity

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
