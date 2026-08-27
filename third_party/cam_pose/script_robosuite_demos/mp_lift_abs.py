import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
import time
import mujoco
import mujoco.viewer


class LiftAbsMotionPlanningController:
    """Minimalist absolute-pose motion planner for LiftRand (OSC_POSE, absolute, world frame)"""

    def __init__(self, env):
        self.env = env

        # State machine
        self.current_phase = 0  # 0=move_above, 1=descend, 2=wait, 3=close_gripper, 4=lift, 5=hold
        self.phase_start_step = 0

        # Success tracking
        self.success_detected = False
        self.success_start_step = 0
        self.done = False

        # Speed caps (per control step at 20 Hz)
        self.max_pos_step = 0.02  # meters
        self.max_ori_step = 0.10  # radians (axis-angle magnitude)

        # Latched pose for halt states
        self.freeze_pos = None
        self.freeze_ori_mat = None

    def plan_lift_trajectory(self, cube_pos, max_steps=200):
        return []

    def _compute_target_world_ori_axisangle(self, cube_quat_xyzw):
        """Compute absolute target gripper orientation (axis-angle in world frame).

        - Choose yaw aligning with the closest cube face (minimizing yaw change)
        - Preserve current gripper roll/pitch; only adjust yaw
        """
        # Cube world rotation
        cube_rot_mat = T.quat2mat(cube_quat_xyzw)
        cube_x_axis = cube_rot_mat[:, 0]
        cube_y_axis = cube_rot_mat[:, 1]

        # Current gripper world rotation from site
        eef_site_id = self.env.sim.model.site_name2id("gripper0_right_grip_site")
        current_rot_mat = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)

        # Extract current yaw from gripper world rotation
        current_yaw = np.arctan2(current_rot_mat[1, 0], current_rot_mat[0, 0])

        # Find desired yaw among 4 face directions (+/-X, +/-Y of cube) minimizing yaw change
        candidates = []
        for axis in (cube_x_axis, -cube_x_axis, cube_y_axis, -cube_y_axis):
            dir2d = axis[:2]
            desired_yaw = np.arctan2(dir2d[1], dir2d[0])
            yaw_diff = desired_yaw - current_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            candidates.append((abs(yaw_diff), desired_yaw))
        _, best_desired_yaw = min(candidates, key=lambda x: x[0])

        # Preserve roll/pitch from current orientation; replace yaw only
        def Rz(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        Rz_curr = Rz(current_yaw)
        rollpitch_mat = Rz_curr.T @ current_rot_mat
        R_target = Rz(best_desired_yaw) @ rollpitch_mat

        # Convert to world-frame axis-angle vector
        aa_vec = T.quat2axisangle(T.mat2quat(R_target))
        return aa_vec

    def _limit_abs_pose(self, curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
        """Step-limit absolute target to cap speed."""
        # Position step limit
        delta_pos = targ_pos - curr_pos
        dist = np.linalg.norm(delta_pos)
        if dist > self.max_pos_step and dist > 1e-12:
            delta_pos = delta_pos / dist * self.max_pos_step
        new_pos = curr_pos + delta_pos

        # Orientation step limit
        R_delta = curr_ori_mat.T @ targ_ori_mat
        delta_axisangle = T.quat2axisangle(T.mat2quat(R_delta))
        mag = np.linalg.norm(delta_axisangle)
        if mag > self.max_ori_step and mag > 1e-12:
            delta_axisangle = delta_axisangle / mag * self.max_ori_step
        R_step = curr_ori_mat @ T.quat2mat(T.axisangle2quat(delta_axisangle))
        new_axisangle_abs = T.quat2axisangle(T.mat2quat(R_step))
        return new_pos, new_axisangle_abs

    def get_real_time_action(self, step_num):
        """Compute absolute pose action for current phase"""
        # Current states
        eef_site_id = self.env.sim.model.site_name2id("gripper0_right_grip_site")
        current_gripper_pos = self.env.sim.data.site_xpos[eef_site_id].copy()
        current_cube_pos = self.env.sim.data.body_xpos[self.env.cube_body_id].copy()
        current_rot_mat = self.env.sim.data.site_xmat[eef_site_id].reshape(3, 3)
        table_z = float(self.env.table_offset[2])

        cube_quat_wxyz = self.env.sim.data.body_xquat[self.env.cube_body_id].copy()
        cube_quat_xyzw = T.convert_quat(cube_quat_wxyz, to="xyzw")

        action = np.zeros(self.env.action_dim)
        abs_ori_vec = self._compute_target_world_ori_axisangle(cube_quat_xyzw)
        target_ori_mat = T.quat2mat(T.axisangle2quat(abs_ori_vec))

        # Track success detection without reward: object height above table threshold
        if (not self.success_detected) and (current_cube_pos[2] > table_z + 0.22):
            self.success_detected = True
            self.success_start_step = step_num

        # Phase logic (absolute targets)
        if self.current_phase == 0:  # move above cube
            target_pos = current_cube_pos.copy()
            target_pos[2] += 0.10

            if np.linalg.norm(current_gripper_pos - target_pos) < 0.05:
                self.current_phase = 1
                self.phase_start_step = step_num

            step_pos, step_aa = self._limit_abs_pose(
                current_gripper_pos, current_rot_mat, target_pos, target_ori_mat
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        elif self.current_phase == 1:  # descend
            target_pos = current_cube_pos.copy()
            target_pos[2] = current_cube_pos[2] - 0.02

            if np.linalg.norm(current_gripper_pos - target_pos) < 0.03:
                self.current_phase = 2
                self.phase_start_step = step_num
                # Latch halt pose
                self.freeze_pos = current_gripper_pos.copy()
                self.freeze_ori_mat = current_rot_mat.copy()

            step_pos, step_aa = self._limit_abs_pose(
                current_gripper_pos, current_rot_mat, target_pos, target_ori_mat
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        elif self.current_phase == 2:  # wait / halt before closing (5 steps)
            if step_num - self.phase_start_step >= 5:
                self.current_phase = 3
                self.phase_start_step = step_num

            # Hold still at latched pose
            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = -1

        elif self.current_phase == 3:  # close gripper
            if step_num - self.phase_start_step > 15:
                self.current_phase = 4
                self.phase_start_step = step_num

            # Keep still while closing
            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = 1

        elif self.current_phase == 4:  # lift
            target_pos = current_cube_pos.copy()
            target_pos[2] += 0.30

            if self.success_detected and step_num - self.success_start_step >= 15:
                self.current_phase = 5
                self.phase_start_step = step_num
                self.done = True
            elif not self.success_detected and (current_cube_pos[2] > 1.0 or step_num - self.phase_start_step > 50):
                self.current_phase = 5
                self.phase_start_step = step_num

            step_pos, step_aa = self._limit_abs_pose(
                current_gripper_pos, current_rot_mat, target_pos, target_ori_mat
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        else:  # hold
            # Keep last absolute pose (no change) and gripper closed
            # Using current pose prevents large jumps if controller resets
            action[0:3] = current_gripper_pos
            action[3:6] = abs_ori_vec
            action[-1] = 1

        return action


def create_lift_abs_debug_env():
    """Create LiftRand env with absolute OSC_POSE (world frame)."""
    ctrl_cfg = load_composite_controller_config(robot="Panda")
    # Set absolute + world frame for right arm
    ctrl_cfg["body_parts"]["right"]["input_type"] = "absolute"
    ctrl_cfg["body_parts"]["right"]["input_ref_frame"] = "world"

    env = suite.make(
        env_name="LiftRand",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        reward_shaping=True,
        hard_reset=False,
        controller_configs=ctrl_cfg,
    )
    return env


def debug_lift_abs_motion_planning(num_runs=10, max_steps=200):
    env = create_lift_abs_debug_env()
    env.reset()

    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    if hasattr(viewer, "opt"):
        viewer.opt.geomgroup[0] = 0

    successes = 0
    sec_per_step = 1.0 / 20.0
    for ep in range(num_runs):
        mp_controller = LiftAbsMotionPlanningController(env)
        env.reset()
        viewer.sync()
        reward = 0
        for step in range(max_steps):
            if not viewer.is_running():
                break
            action = mp_controller.get_real_time_action(step)
            env.step(action)
            viewer.sync()
            if mp_controller.done:
                break
            time.sleep(sec_per_step)

        success = env._check_success()
        successes += int(success)
        print(f"Episode {ep+1}/{num_runs}: {'SUCCESS' if success else 'FAILED'}")
        if not viewer.is_running():
            break

    print(f"Finished {ep+1 if viewer.is_running() else ep+1} episode(s). Successes: {successes}")

    while viewer.is_running():
        mujoco.mj_step(env.sim.model._model, env.sim.data._data)
        viewer.sync()
        time.sleep(sec_per_step)

    env.close()
    return successes


if __name__ == "__main__":
    debug_lift_abs_motion_planning()

