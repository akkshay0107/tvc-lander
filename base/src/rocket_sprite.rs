//! Module for managing the rocket sprite
//!
//! This module contains the implementation to draw the rocket sprite
//! based on its internal state and the functions to update its internal
//! state
use crate::{
    constants::{ROCKET_HEIGHT_M, ROCKET_WIDTH_M},
    world::pixels_per_meter,
};
use macroquad::prelude::*;

fn rocket_width() -> f32 {
    ROCKET_WIDTH_M * pixels_per_meter()
}

fn rocket_height() -> f32 {
    ROCKET_HEIGHT_M * pixels_per_meter()
}

fn flame_max_length() -> f32 {
    rocket_height() * 0.3
}

fn flame_width() -> f32 {
    rocket_width() * 0.6
}

fn window_radius() -> f32 {
    rocket_width() * 0.25
}

const THRUSTER_WIDTH_RATIO: f32 = 0.4;
const THRUSTER_HEIGHT_RATIO: f32 = 0.4;

const BODY_BASE_COLOR: Color = Color::from_rgba(180, 180, 190, 255);
const BODY_HIGHLIGHT_COLOR: Color = Color::from_rgba(220, 220, 235, 255);
const BODY_SHADOW_COLOR: Color = Color::from_rgba(140, 140, 150, 255);

const THRUSTER_BASE_COLOR: Color = Color::from_rgba(160, 160, 170, 255);
const THRUSTER_HIGHLIGHT_COLOR: Color = Color::from_rgba(190, 190, 200, 255);
const THRUSTER_SHADOW_COLOR: Color = Color::from_rgba(130, 130, 140, 255);

const NOSE_HIGHLIGHT_COLOR: Color = Color {
    r: 200.0 / 255.0,
    g: 200.0 / 255.0,
    b: 210.0 / 255.0,
    a: 120.0 / 255.0,
};

const WINDOW_FRAME_COLOR: Color = Color::from_rgba(160, 160, 170, 255);
const WINDOW_GLASS_COLOR: Color = Color::from_rgba(20, 30, 60, 255);
const WINDOW_REFLECTION_COLOR: Color = Color {
    r: 150.0 / 255.0,
    g: 180.0 / 255.0,
    b: 220.0 / 255.0,
    a: 180.0 / 255.0,
};

const FLAME_OUTER_COLOR: Color = RED;
const FLAME_MIDDLE_COLOR: Color = ORANGE;
const FLAME_CORE_COLOR: Color = YELLOW;

pub struct RocketSprite {
    x: f32,
    y: f32,
    angle: f32,        // angle of rocket from vertical (radians)
    thrust: f32,       // thrust magnitude (normalized to [-1, 1])
    gimbal_angle: f32, // gimbal angle relative to rocket body (normalized to [-1, 1])
}

impl RocketSprite {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            angle: 0.0,
            thrust: 0.0,
            gimbal_angle: 0.0,
        }
    }

    pub fn get_state(&self) -> (f32, f32, f32, (f32, f32)) {
        (self.x, self.y, self.angle, (self.thrust, self.gimbal_angle))
    }

    pub fn set_state(&mut self, x: f32, y: f32, angle: f32, thruster: (f32, f32)) {
        self.x = x;
        self.y = y;
        self.angle = angle;
        self.thrust = thruster.0.clamp(-1.0, 1.0);
        self.gimbal_angle = thruster.1.clamp(-1.0, 1.0);
    }

    pub fn draw(&self) {
        let half_width = rocket_width() / 2.0;
        let half_height = rocket_height() / 2.0;
        let thruster_width = rocket_width() * THRUSTER_WIDTH_RATIO;
        let thruster_height = rocket_height() * THRUSTER_HEIGHT_RATIO;
        let center = vec2(self.x, self.y);

        let left_thruster_vertices = [
            vec2(-half_width - thruster_width, half_height - thruster_height),
            vec2(-half_width, half_height - thruster_height),
            vec2(-half_width, half_height),
            vec2(-half_width - thruster_width, half_height),
        ];

        let right_thruster_vertices = [
            vec2(half_width, half_height - thruster_height),
            vec2(half_width + thruster_width, half_height - thruster_height),
            vec2(half_width + thruster_width, half_height),
            vec2(half_width, half_height),
        ];

        // Draw flame effect behind rocket
        if self.thrust != -1.0 {
            self.draw_flame_effect(center);
        }

        self.draw_metallic_body(center);
        self.draw_metallic_thruster(&left_thruster_vertices, center, true);
        self.draw_metallic_thruster(&right_thruster_vertices, center, false);
        self.draw_window(center);
    }

    fn draw_metallic_thruster(&self, vertices: &[Vec2], center: Vec2, is_left: bool) {
        self.draw_rotated_polygon(vertices, center, THRUSTER_BASE_COLOR);

        let thruster_width = rocket_width() * THRUSTER_WIDTH_RATIO;
        let half_width = rocket_width() / 2.0;
        let half_height = rocket_height() / 2.0;
        let thruster_height = rocket_height() * THRUSTER_HEIGHT_RATIO;

        if is_left {
            let highlight_vertices = [
                vec2(-half_width - thruster_width, half_height - thruster_height),
                vec2(
                    -half_width - thruster_width * 0.4,
                    half_height - thruster_height,
                ),
                vec2(-half_width - thruster_width * 0.4, half_height),
                vec2(-half_width - thruster_width, half_height),
            ];
            self.draw_rotated_polygon(&highlight_vertices, center, THRUSTER_HIGHLIGHT_COLOR);

            let shadow_vertices = [
                vec2(
                    -half_width - thruster_width * 0.2,
                    half_height - thruster_height,
                ),
                vec2(-half_width, half_height - thruster_height),
                vec2(-half_width, half_height),
                vec2(-half_width - thruster_width * 0.2, half_height),
            ];
            self.draw_rotated_polygon(&shadow_vertices, center, THRUSTER_SHADOW_COLOR);
        } else {
            let highlight_vertices = [
                vec2(half_width, half_height - thruster_height),
                vec2(
                    half_width + thruster_width * 0.4,
                    half_height - thruster_height,
                ),
                vec2(half_width + thruster_width * 0.4, half_height),
                vec2(half_width, half_height),
            ];
            self.draw_rotated_polygon(&highlight_vertices, center, THRUSTER_HIGHLIGHT_COLOR);

            let shadow_vertices = [
                vec2(
                    half_width + thruster_width * 0.6,
                    half_height - thruster_height,
                ),
                vec2(half_width + thruster_width, half_height - thruster_height),
                vec2(half_width + thruster_width, half_height),
                vec2(half_width + thruster_width * 0.6, half_height),
            ];
            self.draw_rotated_polygon(&shadow_vertices, center, THRUSTER_SHADOW_COLOR);
        }
    }

    fn draw_metallic_body(&self, center: Vec2) {
        let half_width = rocket_width() / 2.0;
        let half_height = rocket_height() / 2.0;
        let nose_radius = half_width;

        let mut body_vertices = Vec::new();

        // Add points for the rounded top (semicircle)
        let segments = 8;
        for i in 0..=segments {
            let angle = std::f32::consts::PI * i as f32 / segments as f32;
            let x = nose_radius * angle.cos();
            let y = -half_height + nose_radius * (1.0 - angle.sin());
            body_vertices.push(vec2(-x, y));
        }

        // Add straight sides and bottom
        body_vertices.push(vec2(half_width, half_height));
        body_vertices.push(vec2(-half_width, half_height));

        // Draw main body
        self.draw_rotated_polygon(&body_vertices, center, BODY_BASE_COLOR);

        // Draw metallic highlight on left side
        let highlight_vertices = [
            vec2(-half_width, -half_height + nose_radius),
            vec2(-half_width * 0.3, -half_height + nose_radius),
            vec2(-half_width * 0.3, half_height),
            vec2(-half_width, half_height),
        ];
        self.draw_rotated_polygon(&highlight_vertices, center, BODY_HIGHLIGHT_COLOR);

        // Draw shadow on right side
        let shadow_vertices = [
            vec2(half_width * 0.3, -half_height + nose_radius),
            vec2(half_width, -half_height + nose_radius),
            vec2(half_width, half_height),
            vec2(half_width * 0.3, half_height),
        ];
        self.draw_rotated_polygon(&shadow_vertices, center, BODY_SHADOW_COLOR);

        self.draw_rounded_nose_highlight(center, nose_radius);
    }

    fn draw_rounded_nose_highlight(&self, center: Vec2, radius: f32) {
        let half_height = rocket_height() / 2.0;
        let nose_center_local = vec2(0.0, -half_height + radius);

        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let nose_center_world = vec2(
            nose_center_local.x * cos_a - nose_center_local.y * sin_a + center.x,
            nose_center_local.x * sin_a + nose_center_local.y * cos_a + center.y,
        );

        draw_circle(
            nose_center_world.x,
            nose_center_world.y,
            radius * 0.6,
            NOSE_HIGHLIGHT_COLOR,
        );
    }

    fn draw_window(&self, center: Vec2) {
        let window_offset_y = -rocket_height() * 0.15;
        let window_local = vec2(0.0, window_offset_y);

        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let window_world = vec2(
            window_local.x * cos_a - window_local.y * sin_a + center.x,
            window_local.x * sin_a + window_local.y * cos_a + center.y,
        );

        draw_circle(
            window_world.x,
            window_world.y,
            window_radius() + 0.5,
            WINDOW_FRAME_COLOR,
        );
        draw_circle(
            window_world.x,
            window_world.y,
            window_radius(),
            WINDOW_GLASS_COLOR,
        );

        let highlight_offset = vec2(-0.8, -0.8);
        let highlight_world = vec2(
            highlight_offset.x * cos_a - highlight_offset.y * sin_a + window_world.x,
            highlight_offset.x * sin_a + highlight_offset.y * cos_a + window_world.y,
        );
        draw_circle(
            highlight_world.x,
            highlight_world.y,
            0.8,
            WINDOW_REFLECTION_COLOR,
        );
    }

    fn draw_flame_effect(&self, center: Vec2) {
        let half_height = rocket_height() / 2.0;
        let flame_length = flame_max_length() * self.thrust.abs();
        let flame_half_width = flame_width() / 2.0;

        let gimbal_effect = self.gimbal_angle * 0.1;
        let total_angle = self.angle + gimbal_effect;

        let flame_vertices = [
            vec2(-flame_half_width, half_height),
            vec2(flame_half_width, half_height),
            vec2(0.0, half_height + flame_length),
        ];

        let inner_flame_vertices = [
            vec2(-flame_half_width * 0.6, half_height),
            vec2(flame_half_width * 0.6, half_height),
            vec2(0.0, half_height + flame_length * 0.8),
        ];

        let core_flame_vertices = [
            vec2(-flame_half_width * 0.3, half_height),
            vec2(flame_half_width * 0.3, half_height),
            vec2(0.0, half_height + flame_length * 0.6),
        ];

        self.draw_rotated_flame(&flame_vertices, center, total_angle, FLAME_OUTER_COLOR);
        self.draw_rotated_flame(
            &inner_flame_vertices,
            center,
            total_angle,
            FLAME_MIDDLE_COLOR,
        );
        self.draw_rotated_flame(&core_flame_vertices, center, total_angle, FLAME_CORE_COLOR);
    }

    fn draw_rotated_flame(&self, vertices: &[Vec2], center: Vec2, angle: f32, color: Color) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let transformed_vertices: Vec<Vec2> = vertices
            .iter()
            .map(|v| {
                vec2(
                    v.x * cos_a - v.y * sin_a + center.x,
                    v.x * sin_a + v.y * cos_a + center.y,
                )
            })
            .collect();

        if transformed_vertices.len() == 3 {
            draw_triangle(
                transformed_vertices[0],
                transformed_vertices[1],
                transformed_vertices[2],
                color,
            );
        }
    }

    fn draw_rotated_polygon(&self, vertices: &[Vec2], center: Vec2, color: Color) {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        let transformed_vertices: Vec<Vec2> = vertices
            .iter()
            .map(|v| {
                vec2(
                    v.x * cos_a - v.y * sin_a + center.x,
                    v.x * sin_a + v.y * cos_a + center.y,
                )
            })
            .collect();

        for i in 0..transformed_vertices.len() {
            let current = transformed_vertices[i];
            let next = transformed_vertices[(i + 1) % transformed_vertices.len()];
            draw_line(current.x, current.y, next.x, next.y, 2.0, color);
        }

        if transformed_vertices.len() >= 3 {
            let first_vertex = transformed_vertices[0];
            for i in 1..transformed_vertices.len() - 1 {
                draw_triangle(
                    first_vertex,
                    transformed_vertices[i],
                    transformed_vertices[i + 1],
                    color,
                );
            }
        }
    }
}
