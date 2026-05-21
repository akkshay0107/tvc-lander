use std::f32::consts::PI;

pub const MAX_POS_X: f32 = 80.0; // Min pos x is 0
pub const _MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
pub const MAX_POS_Y: f32 = 40.0;
pub const MAX_ANGLE_DEFLECTION: f32 = PI / 12.0; // 15 degrees
pub const GROUND_THRESHOLD: f32 = 1.999; // Slightly under min possible y

pub const MAX_LANDING_ANGLE: f32 = PI / 18.0; // 10 degrees tolerance when landing
pub const MAX_LANDING_VX: f32 = 0.7;
pub const MAX_LANDING_VY: f32 = 0.7;
pub const MAX_LANDING_ANGULAR_VELOCITY: f32 = 0.05;

pub const MAX_GIMBAL_ANGLE: f32 = PI / 12.0; // 15 degress
pub const MAX_THRUST: f32 = 20.0; // Thruster can offset gravity

pub const ROCKET_WIDTH_M: f32 = 2.0;
pub const ROCKET_HEIGHT_M: f32 = 4.0;

pub const FLAG_DELTA: f32 = 10.0;
