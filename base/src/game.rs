use crate::constants::FLAG_DELTA;
use crate::world::{ground_y_px, pixels_per_meter};
use macroquad::prelude::*;

pub const PARALLAX_SCROLL_SPEED: f32 = 0.08;

pub const DRAG_POINTER_COLOR: Color = Color::new(1.0, 1.0, 0.0, 1.0);
pub const DRAG_POINTER_RADIUS: f32 = 4.0;

pub struct Game {
    stars: Vec<(f32, f32, f32)>,
    pub rocket: crate::rocket_sprite::RocketSprite,
}

impl Game {
    pub fn new() -> Self {
        let mut stars = Vec::new();
        let ground_y = ground_y_px();
        for _ in 0..50 {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, ground_y); // Stars only in sky
            let radius = rand::gen_range(1.0, 2.0);
            stars.push((x, y, radius));
        }

        let rocket_x = screen_width() / 2.0;
        let rocket = crate::rocket_sprite::RocketSprite::new(rocket_x, ground_y / 2.0);
        Self { stars, rocket }
    }

    pub fn update(&mut self) {
        // Parallax effect implementation
        let ground_y = ground_y_px();
        for star in self.stars.iter_mut() {
            star.0 -= PARALLAX_SCROLL_SPEED;
            if star.0 < 0.0 {
                star.0 = screen_width();
                star.1 = rand::gen_range(0.0, ground_y);
            }
        }
    }

    pub fn draw_space_atmos(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = ground_y_px();
        let num_steps = 50;
        let strip_height = ground_y / num_steps as f32;

        let top_color = Color::new(0.05, 0.02, 0.15, 1.0); // Deep purple
        let mid_color = Color::new(0.02, 0.05, 0.25, 1.0); // Dark blue
        let bottom_color = Color::new(0.01, 0.02, 0.08, 1.0); // Very dark blue

        for i in 0..num_steps {
            let y = i as f32 * strip_height;
            let t = i as f32 / (num_steps - 1) as f32;

            // Linear interpolation of colors to get a gradient
            // Fades from top to mid then mid to bottom in a 60:40 ratio
            let color = if t < 0.6 {
                // Top to mid transition
                let local_t = t / 0.6;
                Color::new(
                    top_color.r + (mid_color.r - top_color.r) * local_t,
                    top_color.g + (mid_color.g - top_color.g) * local_t,
                    top_color.b + (mid_color.b - top_color.b) * local_t,
                    1.0,
                )
            } else {
                // Mid to bottom transition
                let local_t = (t - 0.6) / 0.4;
                Color::new(
                    mid_color.r + (bottom_color.r - mid_color.r) * local_t,
                    mid_color.g + (bottom_color.g - mid_color.g) * local_t,
                    mid_color.b + (bottom_color.b - mid_color.b) * local_t,
                    1.0,
                )
            };

            draw_rectangle(0.0, y, screen_w, strip_height + 1.0, color);
        }

        // Draw deep space below ground
        if screen_h > ground_y {
            draw_rectangle(
                0.0,
                ground_y,
                screen_w,
                screen_h - ground_y,
                Color::new(0.01, 0.02, 0.05, 1.0),
            );
        }
    }

    pub fn draw_ground(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = ground_y_px();

        let base_color = Color::from_rgba(74, 48, 30, 255);
        let shadow_color = Color::from_rgba(57, 36, 23, 255);
        let grass_color = Color::from_rgba(43, 86, 29, 255);
        let grass_edge_color = Color::from_rgba(34, 69, 20, 255);

        let ppm = pixels_per_meter();
        let block_size = (1.6 * ppm).max(1.0);

        // Ground base
        let grass_height = block_size;
        draw_rectangle(0.0, ground_y, screen_w, grass_height, grass_color);
        draw_rectangle(
            0.0,
            ground_y + grass_height,
            screen_w,
            screen_h - (ground_y + grass_height),
            base_color,
        );

        // Top grass block
        let grass_edge_height = (block_size * 0.25).max(1.0);
        draw_rectangle(
            0.0,
            ground_y + grass_height - grass_edge_height,
            screen_w,
            grass_edge_height,
            grass_edge_color,
        );

        // Dirt texture
        let num_blocks_x = (screen_w / block_size).ceil() as i32;
        let num_blocks_y = ((screen_h - (ground_y + grass_height)) / block_size).ceil() as i32;

        for i in 0..num_blocks_x {
            for j in 0..num_blocks_y {
                let x = i as f32 * block_size;
                let y = ground_y + grass_height + j as f32 * block_size;

                let noise_val = (x * 0.1).sin() + (y * 0.07).cos();
                if noise_val <= 0.0 {
                    draw_rectangle(x, y, block_size, block_size, shadow_color);
                }
            }
        }
    }

    pub fn draw_landing_flags(&self, dist_from_center_m: f32) {
        let ppm = pixels_per_meter();
        let dist_from_center = dist_from_center_m * ppm;

        let screen_w = screen_width();
        let ground_y = ground_y_px();
        let center_x = screen_w / 2.0;

        let pole_height = 4.0 * ppm;
        let pole_thickness = 0.2 * ppm;
        let flag_width = 2.5 * ppm;
        let flag_height = 1.5 * ppm;

        let pole_color = WHITE;
        let flag_color = RED;

        // The landing pad is between the two flags.
        // Flags at center +- dist
        let positions = [center_x - dist_from_center, center_x + dist_from_center];

        for &x_pos in &positions {
            // Pole
            draw_line(
                x_pos,
                ground_y,
                x_pos,
                ground_y - pole_height,
                pole_thickness,
                pole_color,
            );

            let direction = if x_pos < center_x { 1.0 } else { -1.0 };

            // Draw flag triangle
            let v1 = Vec2::new(x_pos, ground_y - pole_height);
            let v2 = Vec2::new(
                x_pos + direction * flag_width,
                ground_y - pole_height + flag_height / 2.0,
            );
            let v3 = Vec2::new(x_pos, ground_y - pole_height + flag_height);

            draw_triangle(v1, v2, v3, flag_color);
        }
    }

    pub fn draw(&self) {
        self.draw_space_atmos();

        // Draw stars
        for (x, y, radius) in &self.stars {
            draw_circle(*x, *y, *radius, WHITE);
        }

        self.draw_ground();
        self.draw_landing_flags(FLAG_DELTA);
    }
}
