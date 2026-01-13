import math
import random
from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

MAX_POS_X = 80
MAX_POS_Y = 45
MIN_POS_Y = 2

ROCKET_W = 2
ROCKET_H = 4


class Renderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Rocket Landing")
        self.clock = pygame.time.Clock()

    def world_to_pixel(self, x, y):
        ppm = self.width / MAX_POS_X
        ground_y = self.height * 0.8
        screen_x = x * ppm
        screen_y = ground_y - y * ppm
        return screen_x, screen_y

    def draw_space_atmosphere(self):
        num_steps = 50
        strip_height = self.height / num_steps

        top_color = (13, 5, 38)  # Deep purple
        mid_color = (5, 13, 64)  # Dark blue
        bottom_color = (3, 5, 20)  # Very dark blue

        for i in range(num_steps):
            y = i * strip_height
            t = i / (num_steps - 1)

            if t < 0.6:
                local_t = t / 0.6
                color = tuple(
                    int(top_color[j] + (mid_color[j] - top_color[j]) * local_t)
                    for j in range(3)
                )
            else:
                local_t = (t - 0.6) / 0.4
                color = tuple(
                    int(mid_color[j] + (bottom_color[j] - mid_color[j]) * local_t)
                    for j in range(3)
                )

            pygame.draw.rect(
                self.screen, color, (0, int(y), self.width, int(strip_height + 1))
            )

    def draw_stars(self):
        random.seed(42)

        for _ in range(50):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height * 0.5)
            radius = random.uniform(1, 2)
            pygame.draw.circle(
                self.screen, (255, 255, 255), (int(x), int(y)), int(radius)
            )

    def draw_ground(self):
        ground_y = self.height * 0.8
        ppm = self.width / MAX_POS_X

        block_size = max(1.6 * ppm, 1.0)
        grass_height = block_size

        grass_color = (43, 86, 29)  # #2b561d
        base_color = (74, 48, 30)  # #4a301e
        grass_edge_color = (34, 69, 20)  # #224514
        shadow_color = (57, 36, 23)  # #392417

        # Grass layer
        pygame.draw.rect(
            self.screen, grass_color, (0, int(ground_y), self.width, int(grass_height))
        )

        # Dirt base
        dirt_height = self.height - (ground_y + grass_height)
        pygame.draw.rect(
            self.screen,
            base_color,
            (0, int(ground_y + grass_height), self.width, int(dirt_height)),
        )

        grass_edge_height = max(block_size * 0.25, 1.0)
        pygame.draw.rect(
            self.screen,
            grass_edge_color,
            (
                0,
                int(ground_y + grass_height - grass_edge_height),
                self.width,
                int(grass_edge_height),
            ),
        )

        num_blocks_x = int(self.width / block_size) + 1
        num_blocks_y = int(dirt_height / block_size) + 1

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                x = i * block_size
                y = ground_y + grass_height + j * block_size

                noise_val = math.sin(x * 0.1) + math.cos(y * 0.07)
                if noise_val <= 0.0:
                    pygame.draw.rect(
                        self.screen,
                        shadow_color,
                        (int(x), int(y), int(block_size), int(block_size)),
                    )

    def draw_landing_flags(self):
        ground_y = self.height * 0.8
        center_x = self.width / 2.0
        ppm = self.width / MAX_POS_X

        dist_from_center = 10.0 * ppm
        pole_height = 4.0 * ppm
        pole_thickness = max(int(0.2 * ppm), 2)
        flag_width = 2.5 * ppm
        flag_height = 1.5 * ppm

        positions = [center_x - dist_from_center, center_x + dist_from_center]

        for x_pos in positions:
            pygame.draw.line(
                self.screen,
                (255, 255, 255),  # White
                (int(x_pos), int(ground_y)),
                (int(x_pos), int(ground_y - pole_height)),
                pole_thickness,
            )

            direction = 1.0 if x_pos < center_x else -1.0

            v1 = (int(x_pos), int(ground_y - pole_height))
            v2 = (
                int(x_pos + direction * flag_width),
                int(ground_y - pole_height + flag_height / 2.0),
            )
            v3 = (int(x_pos), int(ground_y - pole_height + flag_height))

            pygame.draw.polygon(
                self.screen,
                (255, 0, 0),  # Red
                [v1, v2, v3],
            )

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
            return int(rx), int(ry)

        nose_radius = half_w
        body_points = []

        # Rounded top
        segments = 8
        for i in range(segments + 1):
            seg_angle = math.pi * i / segments
            lx = -nose_radius * math.cos(seg_angle)
            ly = -half_h + nose_radius * (1.0 - math.sin(seg_angle))
            body_points.append(rotate_point(lx, ly))

        # Bottom corners
        for lx, ly in [(half_w, half_h), (-half_w, half_h)]:
            body_points.append(rotate_point(lx, ly))

        pygame.draw.polygon(
            self.screen,
            (180, 180, 190),  # #b4b4be body color
            body_points,
        )
        pygame.draw.polygon(
            self.screen,
            (140, 140, 150),  # #8c8c96 outline color
            body_points,
            1,
        )

        window_offset_y = -rocket_height_px * 0.15
        window_radius = rocket_width_px * 0.25
        win_x, win_y = rotate_point(0, window_offset_y)

        pygame.draw.circle(
            self.screen,
            (160, 160, 170),  # #a0a0aa frame color
            (win_x, win_y),
            int(window_radius + 0.5),
        )
        pygame.draw.circle(
            self.screen,
            (20, 30, 60),  # #141e3c glass color
            (win_x, win_y),
            int(window_radius),
        )

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

            pygame.draw.polygon(
                self.screen,
                (160, 160, 170),  # #a0a0aa thruster color
                thruster_points,
            )
            pygame.draw.polygon(
                self.screen,
                (130, 140, 140),  # #828c8c outline
                thruster_points,
                1,
            )

    def render(self, state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        self.screen.fill((0, 0, 0))

        self.draw_space_atmosphere()
        self.draw_stars()
        self.draw_ground()
        self.draw_landing_flags()
        self.draw_rocket(state)

        pygame.display.flip()
        self.clock.tick(60)  # Cap at 60 FPS

    def close(self):
        pygame.quit()
