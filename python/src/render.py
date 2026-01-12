import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

MAX_POS_X = 80
MAX_POS_Y = 45
MIN_POS_Y = 2

ROCKET_W = 2
ROCKET_H = 4

# direct copy of code from rust
class Renderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.setup_plot()
        self.fig.show()

    def setup_plot(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.tight_layout(pad=0)

    def macroquad_rectangle(self, x, y, w, h, color):
        mpl_y = self.height - y - h
        rect = Rectangle((x, mpl_y), w, h, facecolor=color, edgecolor="none")
        self.ax.add_patch(rect)

    def macroquad_circle(self, x, y, r, color):
        mpl_y = self.height - y
        circle = Circle((x, mpl_y), r, facecolor=color, edgecolor="none")
        self.ax.add_patch(circle)

    def macroquad_line(self, x1, y1, x2, y2, thickness, color):
        mpl_y1 = self.height - y1
        mpl_y2 = self.height - y2
        self.ax.plot([x1, x2], [mpl_y1, mpl_y2], color=color, linewidth=thickness)

    def macroquad_triangle(self, v1, v2, v3, color):
        points = [
            [v1[0], self.height - v1[1]],
            [v2[0], self.height - v2[1]],
            [v3[0], self.height - v3[1]],
        ]
        tri = Polygon(points, facecolor=color, edgecolor="none")
        self.ax.add_patch(tri)

    def world_to_pixel(self, x, y):
        ppm = self.width / MAX_POS_X
        ground_y = self.height * 0.8
        screen_x = x * ppm
        screen_y = ground_y - y * ppm
        return screen_x, screen_y

    def draw_space_atmosphere(self):
        num_steps = 50
        strip_height = self.height / num_steps

        top_color = np.array([13, 5, 38]) / 255.0
        mid_color = np.array([5, 13, 64]) / 255.0
        bottom_color = np.array([3, 5, 20]) / 255.0

        for i in range(num_steps):
            y = i * strip_height
            t = i / (num_steps - 1)

            if t < 0.6:
                local_t = t / 0.6
                color = top_color + (mid_color - top_color) * local_t
            else:
                local_t = (t - 0.6) / 0.4
                color = mid_color + (bottom_color - mid_color) * local_t

            self.macroquad_rectangle(0.0, y, self.width, strip_height + 1.0, color)

    def draw_stars(self):
        np.random.seed(42)
        for _ in range(50):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height * 0.5)
            radius = np.random.uniform(1, 2)
            self.macroquad_circle(x, y, radius, "white")

    def draw_ground(self):
        ground_y = self.height * 0.8
        ppm = self.width / MAX_POS_X

        block_size = max(1.6 * ppm, 1.0)
        grass_height = block_size

        grass_color = "#2b561d"
        base_color = "#4a301e"
        grass_edge_color = "#224514"
        shadow_color = "#392417"

        self.macroquad_rectangle(0.0, ground_y, self.width, grass_height, grass_color)

        dirt_height = self.height - (ground_y + grass_height)
        self.macroquad_rectangle(
            0.0, ground_y + grass_height, self.width, dirt_height, base_color
        )

        grass_edge_height = max(block_size * 0.25, 1.0)
        self.macroquad_rectangle(
            0.0,
            ground_y + grass_height - grass_edge_height,
            self.width,
            grass_edge_height,
            grass_edge_color,
        )

        num_blocks_x = int(self.width / block_size) + 1
        num_blocks_y = int(dirt_height / block_size) + 1

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                x = i * block_size
                y = ground_y + grass_height + j * block_size

                noise_val = math.sin(x * 0.1) + math.cos(y * 0.07)
                if noise_val <= 0.0:
                    self.macroquad_rectangle(x, y, block_size, block_size, shadow_color)

    def draw_landing_flags(self):
        ground_y = self.height * 0.8
        center_x = self.width / 2.0
        ppm = self.width / MAX_POS_X

        dist_from_center = 10.0 * ppm
        pole_height = 4.0 * ppm
        pole_thickness = 0.2 * ppm
        flag_width = 2.5 * ppm
        flag_height = 1.5 * ppm

        positions = [center_x - dist_from_center, center_x + dist_from_center]

        for x_pos in positions:
            self.macroquad_line(
                x_pos, ground_y, x_pos, ground_y - pole_height, pole_thickness, "white"
            )

            direction = 1.0 if x_pos < center_x else -1.0

            v1 = (x_pos, ground_y - pole_height)
            v2 = (
                x_pos + direction * flag_width,
                ground_y - pole_height + flag_height / 2.0,
            )
            v3 = (x_pos, ground_y - pole_height + flag_height)

            self.macroquad_triangle(v1, v2, v3, "red")

    def draw_rocket(self, state):
        x, y, angle = state["rocket_x"], state["rocket_y"], state["rocket_angle"]

        ppm = self.width / MAX_POS_X
        px, py = self.world_to_pixel(x, y)

        rocket_width_px = ROCKET_W * ppm * 0.5
        rocket_height_px = ROCKET_H * ppm
        half_w = rocket_width_px / 2.0
        half_h = rocket_height_px / 2.0

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        def rotate_point(lx, ly):
            rx = lx * cos_a - ly * sin_a + px
            ry = lx * sin_a + ly * cos_a + py
            return rx, self.height - ry

        nose_radius = half_w
        body_points = []

        segments = 8
        for i in range(segments + 1):
            seg_angle = math.pi * i / segments
            lx = -nose_radius * math.cos(seg_angle)
            ly = -half_h + nose_radius * (1.0 - math.sin(seg_angle))
            body_points.append(rotate_point(lx, ly))

        for lx, ly in [(half_w, half_h), (-half_w, half_h)]:
            body_points.append(rotate_point(lx, ly))

        body = Polygon(
            body_points, facecolor="#b4b4be", edgecolor="#8c8c96", linewidth=1
        )
        self.ax.add_patch(body)

        window_offset_y = -rocket_height_px * 0.15
        window_radius = rocket_width_px * 0.25
        win_x, win_y = rotate_point(0, window_offset_y)

        window_frame = Circle(
            (win_x, win_y), window_radius + 0.5, facecolor="#a0a0aa", edgecolor="none"
        )
        window_glass = Circle(
            (win_x, win_y), window_radius, facecolor="#141e3c", edgecolor="none"
        )
        self.ax.add_patch(window_frame)
        self.ax.add_patch(window_glass)

        thruster_w = rocket_width_px * 0.4
        thruster_h = rocket_height_px * 0.4

        for side in [-1, 1]:
            lx = side * (half_w + thruster_w / 2.0)
            ly = half_h - thruster_h / 2.0

            thruster_points = []
            for dx, dy in [
                (-thruster_w / 2.0, -thruster_h / 2.0),
                (thruster_w / 2.0, -thruster_h / 2.0),
                (thruster_w / 2.0, thruster_h / 2.0),
                (-thruster_w / 2.0, thruster_h / 2.0),
            ]:
                thruster_points.append(rotate_point(lx + dx, ly + dy))

            thruster = Polygon(
                thruster_points, facecolor="#a0a0aa", edgecolor="#828c8c", linewidth=1
            )
            self.ax.add_patch(thruster)

    def render(self, state):
        self.ax.clear()
        self.setup_plot()

        self.draw_space_atmosphere()
        self.draw_stars()
        self.draw_ground()
        self.draw_landing_flags()
        self.draw_rocket(state)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
