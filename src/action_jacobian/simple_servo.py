"""Analytic 3-DoF planar visual-servo benchmark for action conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE = 64
LINK_LENGTHS = np.asarray([0.9, 0.7, 0.5], dtype=np.float32)
TRAIN_SIGNS = np.asarray(
    [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32
)
OOD_SIGNS = np.asarray(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1]], dtype=np.float32
)
WORLD_SCALE = 10.0
WORLD_ORIGIN = np.asarray([32.0, 38.0], dtype=np.float32)


@dataclass(frozen=True)
class ServoSample:
    image: np.ndarray
    pixel_jacobian: np.ndarray
    global_jacobian: np.ndarray
    q: np.ndarray
    target: np.ndarray
    action: np.ndarray
    error_px: np.ndarray


def keypoints(q: np.ndarray) -> np.ndarray:
    angles = np.cumsum(np.asarray(q, dtype=np.float32))
    increments = LINK_LENGTHS[:, None] * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1
    )
    return np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), np.cumsum(increments, axis=0)], axis=0
    )


def point_jacobian(q: np.ndarray, link_index: int, fraction: float = 1.0) -> np.ndarray:
    """Return d(world xy)/d(q) for a point on one link."""
    q = np.asarray(q, dtype=np.float32)
    angles = np.cumsum(q)
    jacobian = np.zeros((2, 3), dtype=np.float32)
    for joint in range(link_index + 1):
        for link in range(joint, link_index + 1):
            length = LINK_LENGTHS[link]
            if link == link_index:
                length *= fraction
            jacobian[:, joint] += length * np.asarray(
                [-np.sin(angles[link]), np.cos(angles[link])], dtype=np.float32
            )
    return jacobian


def world_to_pixel(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    pixels = points * np.asarray([WORLD_SCALE, -WORLD_SCALE], dtype=np.float32)
    return pixels + WORLD_ORIGIN


def eef_pixel_jacobian(q: np.ndarray, signs: np.ndarray) -> np.ndarray:
    projection = np.diag([WORLD_SCALE, -WORLD_SCALE]).astype(np.float32)
    return projection @ point_jacobian(q, 2) @ np.diag(signs)


def oracle_action(
    q: np.ndarray,
    target: np.ndarray,
    signs: np.ndarray,
    damping: float = 0.5,
    max_action: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    eef_px = world_to_pixel(keypoints(q)[-1])
    target_px = world_to_pixel(target)
    error = target_px - eef_px
    jacobian = eef_pixel_jacobian(q, signs)
    inverse = jacobian.T @ np.linalg.inv(
        jacobian @ jacobian.T + damping * damping * np.eye(2, dtype=np.float32)
    )
    action = np.clip(inverse @ error, -max_action, max_action).astype(np.float32)
    return action, error.astype(np.float32)


def render(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (246, 246, 246))
    draw = ImageDraw.Draw(image)
    pixels = world_to_pixel(keypoints(q))
    for index in range(3):
        draw.line(
            [tuple(pixels[index]), tuple(pixels[index + 1])],
            fill=(45, 95 + index * 35, 210),
            width=5,
        )
    for x, y in pixels:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(24, 35, 55))
    tx, ty = world_to_pixel(target)
    draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(30, 200, 80))
    ex, ey = pixels[-1]
    draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(230, 55, 45))
    return np.asarray(image, dtype=np.uint8)


def pixel_jacobian_map(q: np.ndarray, signs: np.ndarray) -> np.ndarray:
    field = np.zeros((7, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    points = keypoints(q)
    projection = np.diag([WORLD_SCALE, -WORLD_SCALE]).astype(np.float32)
    for link in range(3):
        for fraction in np.linspace(0.0, 1.0, 48, dtype=np.float32):
            point = points[link] + fraction * (points[link + 1] - points[link])
            u, v = np.rint(world_to_pixel(point)).astype(int)
            jacobian = projection @ point_jacobian(q, link, float(fraction)) @ np.diag(signs)
            for dv in range(-2, 3):
                for du in range(-2, 3):
                    x, y = u + du, v + dv
                    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                        field[:6, y, x] = jacobian.T.reshape(-1)
                        field[6, y, x] = 1.0
    return field


def global_descriptor(field: np.ndarray) -> np.ndarray:
    mask = field[6] > 0
    values = field[:6, mask]
    if values.shape[1] == 0:
        raise ValueError("Jacobian map has no robot pixels")
    return np.concatenate([values.mean(axis=1), np.sqrt(np.mean(values**2, axis=1))])


def make_sample(q: np.ndarray, target: np.ndarray, signs: np.ndarray) -> ServoSample:
    field = pixel_jacobian_map(q, signs)
    action, error = oracle_action(q, target, signs)
    return ServoSample(
        image=render(q, target),
        pixel_jacobian=field,
        global_jacobian=global_descriptor(field).astype(np.float32),
        q=np.asarray(q, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        action=action,
        error_px=error,
    )


def sample_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.uniform([-1.25, -1.2, -1.2], [1.25, 1.2, 1.2]).astype(np.float32)
    for _ in range(100):
        q_goal = np.clip(
            q + rng.uniform(-0.55, 0.55, size=3),
            [-1.25, -1.2, -1.2],
            [1.25, 1.2, 1.2],
        ).astype(np.float32)
        target = keypoints(q_goal)[-1]
        pixel = world_to_pixel(target)
        initial_error = np.linalg.norm(
            world_to_pixel(target) - world_to_pixel(keypoints(q)[-1])
        )
        if 3.0 <= initial_error <= 10.0 and np.all((pixel >= 5) & (pixel <= 58)):
            return q, target
    raise RuntimeError("Failed to sample a visible target")


def rollout_oracle(
    q: np.ndarray,
    target: np.ndarray,
    signs: np.ndarray,
    horizon: int = 25,
    threshold_px: float = 2.5,
) -> tuple[bool, float, int]:
    q = np.asarray(q, dtype=np.float32).copy()
    for step in range(horizon + 1):
        error = np.linalg.norm(world_to_pixel(target) - world_to_pixel(keypoints(q)[-1]))
        if error <= threshold_px:
            return True, float(error), step
        action, _ = oracle_action(q, target, signs)
        q += signs * action
    return False, float(error), horizon
