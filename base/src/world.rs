//! A 2D physics simulation world controlling the rocket
//!
//! This module contains the physics simulation world for the rocket landing scenario
//! Contains the physical rocket body and the ground and has methods to which modify
//! the velocity and position of the rocket rigid body and support drag and drop of the
//! rocket body

use macroquad::prelude::*;
use rapier2d::prelude::*;

use crate::constants::{MAX_GIMBAL_ANGLE, ROCKET_HEIGHT_M, ROCKET_WIDTH_M};

const GROUND_RESTITUTION: f32 = 0.5;
const ROCKET_RESTITUTION: f32 = 0.1;
const ROCKET_MASS: f32 = 1.0;
const GROUND_SIZE: Vector<f32> = vector![40.0, 6.0];
const ANGULAR_DRAG_COEFFICIENT: f32 = 2.0;
const LINEAR_DRAG_COEFFICIENT: f32 = 1.5;

pub const MAX_THRUST: f32 = 15.0; // Thruster can offset gravity

const TOLERANCE_RADIUS: f32 = 1.5; // Set to the average of the dimensions of the rocket body

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
    pub rocket_body_handle: RigidBodyHandle,
    pub is_dragging: bool,
    pub drag_start_pos: Option<Vector<f32>>,
    pub drag_anchor: Option<Vector<f32>>,
}

impl World {
    pub fn new() -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        let ground_position = vector![40.0, -6.0];
        let _ground_handle =
            Self::create_ground(&mut rigid_body_set, &mut collider_set, ground_position);

        let rocket_start_position = vector![40.0, 40.0];
        let rocket_body_handle = Self::create_rocket(
            &mut rigid_body_set,
            &mut collider_set,
            rocket_start_position,
        );

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
            rocket_body_handle,
            is_dragging: false,
            drag_start_pos: None,
            drag_anchor: None,
        }
    }

    fn create_ground(
        rigid_body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
        position: Vector<f32>,
    ) -> RigidBodyHandle {
        let ground = RigidBodyBuilder::fixed().translation(position).build();
        let ground_handle = rigid_body_set.insert(ground);

        let collider = ColliderBuilder::cuboid(GROUND_SIZE.x, GROUND_SIZE.y)
            .restitution(GROUND_RESTITUTION)
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
        // Both thrust and gimbal angle assumed to be normalized
        if thrust < -1.0 || thrust > 1.0 {
            panic!(
                "Thrust isn't in the normalized range. Normalize thrust to [-1, 1] before passing it."
            )
        }

        if gimbal_angle < -1.0 || gimbal_angle > 1.0 {
            panic!(
                "Gimbal angle isn't in the normalized range. Normalize thrust to [-1, 1] before passing it."
            )
        }

        if thrust == 0.0 {
            return;
        }

        let raw_thrust = (MAX_THRUST * (1.0 + thrust)) / 2.0;
        let raw_gimbal_angle = gimbal_angle * MAX_GIMBAL_ANGLE;

        let rocket_body = self
            .rigid_body_set
            .get_mut(self.rocket_body_handle)
            .unwrap();
        let angle_from_vertical = raw_gimbal_angle + rocket_body.rotation().angle();

        // Find thrust in world coordinates
        let thrust_force_world = vector![
            raw_thrust * angle_from_vertical.sin(),
            raw_thrust * angle_from_vertical.cos()
        ];

        // equivalent to updating force
        rocket_body.reset_forces(true);
        rocket_body.add_force(thrust_force_world, true);

        // Find torque on rocket body
        let y_offset = ROCKET_HEIGHT_M / 2.0;
        let offset = (0.0, -y_offset);
        let thrust_force_body = (
            raw_thrust * raw_gimbal_angle.sin(),
            raw_thrust * raw_gimbal_angle.cos(),
        );

        let torque = Self::cross_product(offset, thrust_force_body);
        rocket_body.reset_torques(true);
        if torque != 0.0 {
            rocket_body.add_torque(torque, true);
        }
    }

    fn cross_product(a: (f32, f32), b: (f32, f32)) -> f32 {
        // k component of a cross b where a and b lie in the xy plane
        a.0 * b.1 - a.1 * b.0
    }

    pub fn get_rocket_state(&self) -> (f32, f32, f32) {
        let rocket_body = &self.rigid_body_set[self.rocket_body_handle];
        (
            rocket_body.translation().x,
            rocket_body.translation().y,
            rocket_body.rotation().angle(),
        )
    }

    pub fn get_rocket_dynamics(&self) -> (f32, f32, f32) {
        let rocket_body = &self.rigid_body_set[self.rocket_body_handle];
        let lin_vel = rocket_body.linvel();
        let ang_vel = rocket_body.angvel();
        (lin_vel.x, lin_vel.y, ang_vel)
    }

    pub fn start_drag(&mut self, mouse_world_pos: Vector<f32>) -> bool {
        let rocket_body = &self.rigid_body_set[self.rocket_body_handle];
        let rocket_pos = rocket_body.translation();

        let dx = mouse_world_pos.x - rocket_pos.x;
        let dy = mouse_world_pos.y - rocket_pos.y;
        let sq_dist = dx * dx + dy * dy;

        // Only start drag if within tolerance limit
        if sq_dist < TOLERANCE_RADIUS * TOLERANCE_RADIUS {
            self.is_dragging = true;
            self.drag_start_pos = Some(rocket_pos.clone());
            self.drag_anchor = Some(vector![
                mouse_world_pos.x - rocket_pos.x,
                mouse_world_pos.y - rocket_pos.y
            ]);
            return true;
        }

        false
    }

    pub fn update_drag(&mut self, mouse_world_pos: Vector<f32>) {
        if !self.is_dragging {
            return;
        }

        if let Some(anchor) = &self.drag_anchor {
            if let Some(rocket_body) = self.rigid_body_set.get_mut(self.rocket_body_handle) {
                // Calculate target position based on mouse position and anchor point
                let target_pos =
                    vector![mouse_world_pos.x - anchor.x, mouse_world_pos.y - anchor.y];

                let current_pos = rocket_body.translation();
                let to_target = target_pos - current_pos;

                // Give the rocket enough velocity to just lag behind the mouse
                // 60 fps => speed / distance should be 60 for perfect following
                // coefficient should be slightly lower for the lag effect
                let factor = 20.0;
                let velocity = to_target * factor;

                rocket_body.set_linvel(velocity, true);
                // Prevent rotation while dragging
                rocket_body.set_angvel(0.0, true);
                rocket_body.wake_up(true);
            }
        }
    }

    pub fn end_drag(&mut self) {
        self.is_dragging = false;
        self.drag_start_pos = None;
        self.drag_anchor = None;

        // Reset angular velocity when drag ends
        if let Some(rocket_body) = self.rigid_body_set.get_mut(self.rocket_body_handle) {
            rocket_body.set_angvel(0.0, true);
            rocket_body.set_linvel(vector![0.0, 0.0], true);
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

pub fn pixels_per_meter() -> f32 {
    screen_width() / crate::constants::MAX_POS_X
}

pub fn world_to_pixel(x: f32, y: f32) -> (f32, f32) {
    let ppm = pixels_per_meter();
    let ground_y = screen_height() * 0.8;
    let screen_x = x * ppm;
    let screen_y = ground_y - y * ppm;
    (screen_x, screen_y)
}

pub fn pixel_to_world(screen_x: f32, screen_y: f32) -> (f32, f32) {
    let ppm = pixels_per_meter();
    let ground_y = screen_height() * 0.8;
    let world_x = screen_x / ppm;
    let world_y = (ground_y - screen_y) / ppm;
    (world_x, world_y)
}
