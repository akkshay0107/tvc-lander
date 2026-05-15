use std::f32::consts::PI;

use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;
use rayon::prelude::*;

const BASE_REWARD_SCALE: f32 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq)]
enum EpisodeStatus {
    InProgress,
    Success,
    MissingTarget,
    Crash,
    OutOfBounds,
    Timeout,
}

impl EpisodeStatus {
    fn as_str(&self) -> &'static str {
        match self {
            EpisodeStatus::InProgress => "in_progress",
            EpisodeStatus::Success => "success",
            EpisodeStatus::MissingTarget => "missing_target",
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

        let mut next_obs = Vec::with_capacity(self.group_size);
        self.prev_potentials = Vec::with_capacity(self.group_size);

        for &handle in &self.world.rocket_handles {
            let state = self._sample();
            let potential = self
                ._calculate_potential(state[0], state[1], state[2], state[3], state[4], state[5]);
            self.prev_potentials.push(potential);

            let rocket = self.world.rigid_body_set.get_mut(handle).unwrap();
            rocket.set_position(Isometry2::new(vector![state[0], state[1]], state[2]), true);
            rocket.set_linvel(vector![state[3], state[4]], true);
            rocket.set_angvel(state[5], true);

            next_obs.push(self._normalize(state));
        }

        Ok(next_obs)
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
                let (reward, potential) =
                    self._calculate_reward(i, x, y, theta, vx, vy, omega, actions[i]);

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

        // bias towards x
        let dist_sq = 3.0 * nx.powi(2) + ny.powi(2);
        let vel_sq = nvx.powi(2) + nvy.powi(2);
        let angle_sq = ntheta.powi(2) + nomega.powi(2);

        // normalize scores
        let dist_score = (1.0 - (dist_sq / 4.0)).max(0.0);
        let vel_score = (1.0 - (vel_sq / 2.0)).max(0.0);
        let angle_score = (1.0 - (angle_sq / 2.0)).max(0.0);

        let potential = 0.5 * dist_score + 0.2 * vel_score + 0.3 * angle_score;

        BASE_REWARD_SCALE * potential
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
        action: [f32; 2],
    ) -> (f32, f32) {
        let current_potential = self._calculate_potential(x, y, theta, vx, vy, omega);
        let shaping_reward = current_potential - self.prev_potentials[idx];

        let landed = y <= _MIN_POS_Y;
        let terminal_reward = if self._is_oob(x, y) {
            -BASE_REWARD_SCALE
        } else if !landed {
            0.0
        } else {
            if self._is_crash(theta, vx, vy, omega) {
                -BASE_REWARD_SCALE
            } else {
                let nx = (2.0 * x - MAX_POS_X) / MAX_POS_X;
                let centering_bonus = BASE_REWARD_SCALE * (1.0 - nx.abs());
                let precision_bonus = BASE_REWARD_SCALE * (-nx.powi(2)).exp();

                BASE_REWARD_SCALE + 0.5 * centering_bonus + 0.5 * precision_bonus
            }
        };

        // fuel and time penalty to prefer more efficient trajs
        let time_penalty = 0.02;
        let thrust_norm = (action[0] + 1.0) / 2.0;
        let fuel_penalty = 5e-3 * thrust_norm;

        (
            shaping_reward + terminal_reward - time_penalty - fuel_penalty,
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

    fn _is_crash(&self, theta: f32, vx: f32, vy: f32, omega: f32) -> bool {
        let bad_angle = theta.abs() > MAX_LANDING_ANGLE;
        let fast_land = vy.abs() > MAX_LANDING_VY;
        let fast_horiz = vx.abs() > MAX_LANDING_VX;
        let fast_spin = omega.abs() > MAX_LANDING_ANGULAR_VELOCITY;

        bad_angle || fast_land || fast_horiz || fast_spin
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
        let landed = y <= _MIN_POS_Y;

        let status = if self._is_oob(x, y) {
            EpisodeStatus::OutOfBounds
        } else if self.steps >= self.max_steps {
            EpisodeStatus::Timeout
        } else if !landed {
            EpisodeStatus::InProgress
        } else {
            let left_flag = (MAX_POS_X / 2.0) - FLAG_DELTA;
            let right_flag = (MAX_POS_X / 2.0) + FLAG_DELTA;
            if self._is_crash(theta, vx, vy, omega) {
                EpisodeStatus::Crash
            } else if left_flag <= x && x <= right_flag {
                EpisodeStatus::Success
            } else {
                EpisodeStatus::MissingTarget
            }
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
