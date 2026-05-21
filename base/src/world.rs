//! A 2D physics simulation world controlling multiple rockets
//!
//! This module contains the physics simulation world for the rocket landing scenario.
//! It supports multiple rockets ghosting through each other while colliding with the ground.

use rapier2d::prelude::*;
use rayon::prelude::*;

use crate::constants::{MAX_GIMBAL_ANGLE, MAX_POS_X, MAX_THRUST, ROCKET_HEIGHT_M, ROCKET_WIDTH_M};

const GROUND_RESTITUTION: f32 = 0.5;
const ROCKET_RESTITUTION: f32 = 0.1;
const ROCKET_MASS: f32 = 1.0;
const GROUND_SIZE_Y: f32 = 6.0;
const ANGULAR_DRAG_COEFFICIENT: f32 = 2.5;
const LINEAR_DRAG_COEFFICIENT: f32 = 1.75;

// Collision Groups
// Group 1: Ground
// Group 2: Rockets
// Rockets should collide with ground (Group 1) but not with each other.
const GROUP_GROUND: InteractionGroups = InteractionGroups::new(Group::GROUP_1, Group::ALL);
const GROUP_ROCKET: InteractionGroups = InteractionGroups::new(Group::GROUP_2, Group::GROUP_1);

pub struct World {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub gravity: Vector<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub rocket_handles: Vec<RigidBodyHandle>,
    pub group_size: usize,
    pub is_dragging: bool,
    pub drag_start_world: Vector<f32>,
    pub drag_current_world: Vector<f32>,
}

impl World {
    pub fn new(group_size: usize) -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        // Physical ground is at y=0. Collider is centered at -GROUND_SIZE_Y.
        let ground_position = vector![MAX_POS_X / 2.0, -GROUND_SIZE_Y];
        Self::create_ground(&mut rigid_body_set, &mut collider_set, ground_position);

        let mut rocket_handles = Vec::with_capacity(group_size);
        let rocket_start_position = vector![MAX_POS_X / 2.0, 30.0];
        for _ in 0..group_size {
            let handle = Self::create_rocket(
                &mut rigid_body_set,
                &mut collider_set,
                rocket_start_position,
            );
            rocket_handles.push(handle);
        }

        Self {
            rigid_body_set,
            collider_set,
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            gravity: vector![0.0, -9.81],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            rocket_handles,
            group_size,
            is_dragging: false,
            drag_start_world: vector![0.0, 0.0],
            drag_current_world: vector![0.0, 0.0],
        }
    }

    fn create_ground(
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        position: Vector<f32>,
    ) -> RigidBodyHandle {
        let ground = RigidBodyBuilder::fixed().translation(position).build();
        let ground_handle = rigid_body_set.insert(ground);

        let collider = ColliderBuilder::cuboid(MAX_POS_X / 2.0, GROUND_SIZE_Y)
            .restitution(GROUND_RESTITUTION)
            .collision_groups(GROUP_GROUND)
            .build();
        collider_set.insert_with_parent(collider, ground_handle, rigid_body_set);

        ground_handle
    }

    fn create_rocket(
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        position: Vector<f32>,
    ) -> RigidBodyHandle {
        let rocket_body = RigidBodyBuilder::dynamic()
            .translation(position)
            .linear_damping(LINEAR_DRAG_COEFFICIENT)
            .angular_damping(ANGULAR_DRAG_COEFFICIENT)
            .build();
        let rocket_handle = rigid_body_set.insert(rocket_body);

        let half_width = ROCKET_WIDTH_M / 2.0;
        let half_height = ROCKET_HEIGHT_M / 2.0;

        let body_collider = ColliderBuilder::cuboid(half_width, half_height)
            .restitution(ROCKET_RESTITUTION)
            .mass(ROCKET_MASS)
            .collision_groups(GROUP_ROCKET)
            .build();
        collider_set.insert_with_parent(body_collider, rocket_handle, rigid_body_set);

        rocket_handle
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    pub fn apply_thruster_forces(&mut self, thrust: f32, gimbal_angle: f32) {
        self.apply_multi_thruster_forces(&[[thrust, gimbal_angle]]);
    }

    pub fn apply_multi_thruster_forces(&mut self, actions: &[[f32; 2]]) {
        // Parallelize force calculation
        let forces: Vec<(Vector<f32>, f32)> = self
            .rocket_handles
            .par_iter()
            .enumerate()
            .map(|(i, &handle)| {
                let [thrust, gimbal_angle] = actions[i];

                let raw_thrust = (MAX_THRUST * (1.0 + thrust)) / 2.0;
                let raw_thrust = raw_thrust.clamp(0.0, MAX_THRUST);

                if raw_thrust <= 0.0001 {
                    return (vector![0.0, 0.0], 0.0);
                }
                let raw_gimbal_angle = gimbal_angle * MAX_GIMBAL_ANGLE;

                let rocket_body = &self.rigid_body_set[handle];
                let angle_from_vertical = raw_gimbal_angle + rocket_body.rotation().angle();

                let thrust_force_world = vector![
                    raw_thrust * angle_from_vertical.sin(),
                    raw_thrust * angle_from_vertical.cos()
                ];

                let y_offset = ROCKET_HEIGHT_M / 2.0;
                let offset = (0.0, -y_offset);
                let thrust_force_body = (
                    raw_thrust * raw_gimbal_angle.sin(),
                    raw_thrust * raw_gimbal_angle.cos(),
                );

                let torque = offset.0 * thrust_force_body.1 - offset.1 * thrust_force_body.0;
                (thrust_force_world, torque)
            })
            .collect();

        // Apply forces in serial (RigidBodySet is not thread-safe for mutation in this way)
        for (i, (force, torque)) in forces.into_iter().enumerate() {
            let rb = self.rigid_body_set.get_mut(self.rocket_handles[i]).unwrap();
            rb.reset_forces(true);
            rb.add_force(force, true);
            rb.reset_torques(true);
            if torque != 0.0 {
                rb.add_torque(torque, true);
            }
        }
    }

    pub fn get_multi_rocket_state(&self) -> Vec<(f32, f32, f32)> {
        self.rocket_handles
            .par_iter()
            .map(|&h| {
                let rb = &self.rigid_body_set[h];
                (
                    rb.translation().x,
                    rb.translation().y,
                    rb.rotation().angle(),
                )
            })
            .collect()
    }

    pub fn get_multi_rocket_dynamics(&self) -> Vec<(f32, f32, f32)> {
        self.rocket_handles
            .par_iter()
            .map(|&h| {
                let rb = &self.rigid_body_set[h];
                let linvel = rb.linvel();
                let angvel = rb.angvel();
                (linvel.x, linvel.y, angvel)
            })
            .collect()
    }

    pub fn get_rocket_state(&self) -> (f32, f32, f32) {
        let (x, y, a) = self.get_multi_rocket_state()[0];
        (x, y, a)
    }

    pub fn get_rocket_dynamics(&self) -> (f32, f32, f32) {
        let (vx, vy, w) = self.get_multi_rocket_dynamics()[0];
        (vx, vy, w)
    }

    // Drag-and-drop logic for a single rocket (only meant for the graphical sims)
    pub fn start_drag(&mut self, pos: Vector<f32>) {
        let rb = &self.rigid_body_set[self.rocket_handles[0]];
        let rocket_pos = rb.translation();
        let dist = (rocket_pos - pos).norm();

        // Only start drag if close to the rocket (within 5 meters)
        if dist < 5.0 {
            self.is_dragging = true;
            self.drag_start_world = pos;
            self.drag_current_world = pos;
        }
    }

    pub fn update_drag(&mut self, pos: Vector<f32>) {
        if self.is_dragging {
            self.drag_current_world = pos;
            for &handle in &self.rocket_handles {
                let rb = self.rigid_body_set.get_mut(handle).unwrap();
                let rocket_pos = *rb.translation();
                let rocket_vel = *rb.linvel();

                // spring force: pull towards mouse
                let spring_k = 150.0;
                let damping_c = 15.0;

                let force = (pos - rocket_pos) * spring_k - rocket_vel * damping_c;
                rb.apply_impulse(force * 0.016, true);

                // torque to pull upright
                let angle = rb.rotation().angle();
                let torque_k = 10.0;
                let torque_d = 2.0;
                let torque = -angle * torque_k - rb.angvel() * torque_d;
                rb.apply_torque_impulse(torque * 0.016, true);
            }
        }
    }

    pub fn end_drag(&mut self) {
        if self.is_dragging {
            for &handle in &self.rocket_handles {
                let rb = self.rigid_body_set.get_mut(handle).unwrap();
                rb.set_linvel(vector![0.0, 0.0], true);
                rb.set_angvel(0.0, true);
            }
            self.is_dragging = false;
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new(1)
    }
}

pub fn pixels_per_meter() -> f32 {
    macroquad::window::screen_width() / crate::constants::MAX_POS_X
}

pub fn ground_y_px() -> f32 {
    crate::constants::MAX_POS_Y * pixels_per_meter()
}

pub fn world_to_pixel(x: f32, y: f32) -> (f32, f32) {
    let ppm = pixels_per_meter();
    (x * ppm, ground_y_px() - y * ppm)
}

pub fn pixel_to_world(x: f32, y: f32) -> (f32, f32) {
    let ppm = pixels_per_meter();
    (x / ppm, (ground_y_px() - y) / ppm)
}
