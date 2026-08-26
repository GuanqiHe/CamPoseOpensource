"""Pixel-aligned Jacobian from raw joint-delta actions to robot image motion."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)


@dataclass(frozen=True)
class PixelActionJacobian:
    """A dense grid plus geometry needed for finite-difference validation."""

    field: np.ndarray
    body_ids: np.ndarray
    world_points: np.ndarray


def _robot_body_ids(model, root_body_id: int) -> set[int]:
    body_ids = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor != 0:
            if ancestor == root_body_id:
                body_ids.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    body_ids.add(root_body_id)
    return body_ids


def _continuous_project(
    world_points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Project world points to continuous ``(u, v)`` image coordinates."""

    camera_points = (
        world_points @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    z = camera_points[..., 2]
    u = intrinsic[0, 0] * camera_points[..., 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_points[..., 1] / z + intrinsic[1, 2]
    return np.stack([u, v], axis=-1)


def compute_pixel_action_jacobian(
    env,
    camera_name: str,
    grid_height: int = 16,
    grid_width: int = 16,
    action_to_qpos_scale: np.ndarray | None = None,
) -> PixelActionJacobian:
    """Compute ``[du/da_j, dv/da_j]`` at each visible robot cell.

    The current experiment uses the ``joint_delta`` wrapper, whose raw arm
    action is accumulated directly into an absolute joint-position goal. Its
    ``action_to_qpos_scale`` is therefore one. The optional scale keeps this
    derivative explicit for other action interfaces.
    """

    sim = env.sim
    robot = env.robots[0]
    qvel_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    if qvel_indexes.shape != (7,):
        raise ValueError(f"Expected a 7-DoF arm, got {qvel_indexes.shape}")
    if action_to_qpos_scale is None:
        action_to_qpos_scale = np.ones(7, dtype=np.float64)
    action_to_qpos_scale = np.asarray(action_to_qpos_scale, dtype=np.float64)
    if action_to_qpos_scale.shape != (7,):
        raise ValueError("action_to_qpos_scale must have shape (7,)")

    segmentation, normalized_depth = sim.render(
        camera_name=camera_name,
        height=grid_height,
        width=grid_width,
        depth=True,
        segmentation=True,
    )
    segmentation = segmentation[::-1]
    normalized_depth = normalized_depth[::-1]
    depth = get_real_depth_map(sim, normalized_depth)

    intrinsic = get_camera_intrinsic_matrix(
        sim, camera_name, grid_height, grid_width
    )
    camera_to_world = get_camera_extrinsic_matrix(sim, camera_name)
    world_to_camera = np.linalg.inv(camera_to_world)
    camera_rotation = world_to_camera[:3, :3]

    root_body_id = sim.model.body_name2id(robot.robot_model.root_body)
    robot_body_ids = _robot_body_ids(sim.model, root_body_id)
    geom_object_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    field = np.zeros((15, grid_height, grid_width), dtype=np.float32)
    body_ids = np.full((grid_height, grid_width), -1, dtype=np.int32)
    world_points = np.full(
        (grid_height, grid_width, 3), np.nan, dtype=np.float64
    )

    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
    native_model = sim.model._model
    native_data = sim.data._data

    for v in range(grid_height):
        for u in range(grid_width):
            object_type, geom_id = segmentation[v, u]
            if int(object_type) != geom_object_type or int(geom_id) < 0:
                continue
            body_id = int(sim.model.geom_bodyid[int(geom_id)])
            if body_id not in robot_body_ids:
                continue

            z = float(depth[v, u])
            point_camera = np.array(
                [(u - c_x) * z / f_x, (v - c_y) * z / f_y, z],
                dtype=np.float64,
            )
            point_world = (
                camera_to_world[:3, :3] @ point_camera
                + camera_to_world[:3, 3]
            )
            jacobian_world = np.zeros((3, sim.model.nv), dtype=np.float64)
            mujoco.mj_jac(
                native_model,
                native_data,
                jacobian_world,
                None,
                point_world,
                body_id,
            )
            velocity_camera = (
                camera_rotation
                @ jacobian_world[:, qvel_indexes]
                * action_to_qpos_scale[None, :]
            )
            x, y, z = point_camera
            du = (
                f_x / z * velocity_camera[0]
                - f_x * x / (z * z) * velocity_camera[2]
            )
            dv = (
                f_y / z * velocity_camera[1]
                - f_y * y / (z * z) * velocity_camera[2]
            )
            field[:14, v, u] = np.stack([du, dv], axis=-1).reshape(-1)
            field[14, v, u] = 1.0
            body_ids[v, u] = body_id
            world_points[v, u] = point_world

    return PixelActionJacobian(
        field=field,
        body_ids=body_ids,
        world_points=world_points,
    )


def finite_difference_pixel_action_jacobian(
    env,
    jacobian: PixelActionJacobian,
    camera_name: str,
    epsilon: float = 1e-5,
    action_to_qpos_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Differentiate projection of the same body-fixed surface points."""

    sim = env.sim
    qpos_indexes = np.asarray(
        env.robots[0]._ref_joint_pos_indexes, dtype=np.int64
    )
    if qpos_indexes.shape != (7,):
        raise ValueError(f"Expected a 7-DoF arm, got {qpos_indexes.shape}")
    if action_to_qpos_scale is None:
        action_to_qpos_scale = np.ones(7, dtype=np.float64)
    action_to_qpos_scale = np.asarray(action_to_qpos_scale, dtype=np.float64)

    mask = jacobian.field[14].astype(bool)
    valid_body_ids = jacobian.body_ids[mask]
    base_world_points = jacobian.world_points[mask]
    local_points = np.empty_like(base_world_points)
    for index, (body_id, point_world) in enumerate(
        zip(valid_body_ids, base_world_points)
    ):
        body_rotation = sim.data.body_xmat[body_id].reshape(3, 3)
        body_position = sim.data.body_xpos[body_id]
        local_points[index] = body_rotation.T @ (point_world - body_position)

    intrinsic = get_camera_intrinsic_matrix(
        sim,
        camera_name,
        jacobian.field.shape[1],
        jacobian.field.shape[2],
    )
    world_to_camera = np.linalg.inv(
        get_camera_extrinsic_matrix(sim, camera_name)
    )
    base_qpos = sim.data.qpos.copy()
    finite_difference = np.zeros_like(jacobian.field[:14], dtype=np.float64)

    def transformed_points() -> np.ndarray:
        points = np.empty_like(local_points)
        for index, (body_id, point_local) in enumerate(
            zip(valid_body_ids, local_points)
        ):
            body_rotation = sim.data.body_xmat[body_id].reshape(3, 3)
            points[index] = (
                body_rotation @ point_local + sim.data.body_xpos[body_id]
            )
        return points

    try:
        for joint_index, qpos_index in enumerate(qpos_indexes):
            qpos_step = epsilon * action_to_qpos_scale[joint_index]
            sim.data.qpos[:] = base_qpos
            sim.data.qpos[qpos_index] += qpos_step
            sim.forward()
            plus = _continuous_project(
                transformed_points(), world_to_camera, intrinsic
            )

            sim.data.qpos[:] = base_qpos
            sim.data.qpos[qpos_index] -= qpos_step
            sim.forward()
            minus = _continuous_project(
                transformed_points(), world_to_camera, intrinsic
            )

            derivative = (plus - minus) / (2.0 * epsilon)
            finite_difference[2 * joint_index][mask] = derivative[:, 0]
            finite_difference[2 * joint_index + 1][mask] = derivative[:, 1]
    finally:
        sim.data.qpos[:] = base_qpos
        sim.forward()

    return finite_difference.astype(np.float32)
