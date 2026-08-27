import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
import time
import mujoco
import mujoco.viewer


class CanAbsMotionPlanningController:
    """Absolute-pose motion planner for CanRand (pick can and place at goal).

    Uses composite OSC_POSE with absolute, world-frame inputs. Position and
    orientation commands are step-limited each control step to cap speeds.
    """

    def __init__(self, env):
        self.env = env

        # Per-step speed caps at 20 Hz
        self.max_pos_step = 0.02  # meters
        self.max_ori_step = 0.20  # radians (axis-angle magnitude)

        # Phases:
        # 0=move_above_can, 1=descend_can, 2=wait, 3=close, 4=lift,
        # 5=move_above_goal, 6=descend_goal, 7=open, 8=retreat
        self.current_phase = 0
        self.phase_start_step = 0
        self.done = False
        self._transit_z = None

        # Latched pose for halt states
        self.freeze_pos = None
        self.freeze_ori_mat = None

    def _limit_abs_pose(self, curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
        """Step-limit absolute target to cap speed (isotropic)."""
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

    def _limit_abs_pose_aniso(self, curr_pos, curr_ori_mat, targ_pos, targ_ori_mat, max_step_xy, max_step_z):
        """Step-limit absolute target with anisotropic XY/Z caps."""
        delta_pos = targ_pos - curr_pos
        delta_xy = delta_pos[:2]
        n_xy = np.linalg.norm(delta_xy)
        if n_xy > max_step_xy and n_xy > 1e-12:
            delta_xy = delta_xy / n_xy * max_step_xy
        dz = float(delta_pos[2])
        if abs(dz) > max_step_z:
            dz = np.sign(dz) * max_step_z
        new_pos = curr_pos.copy()
        new_pos[0] += delta_xy[0]
        new_pos[1] += delta_xy[1]
        new_pos[2] += dz

        R_delta = curr_ori_mat.T @ targ_ori_mat
        delta_axisangle = T.quat2axisangle(T.mat2quat(R_delta))
        mag = np.linalg.norm(delta_axisangle)
        if mag > self.max_ori_step and mag > 1e-12:
            delta_axisangle = delta_axisangle / mag * self.max_ori_step
        R_step = curr_ori_mat @ T.quat2mat(T.axisangle2quat(delta_axisangle))
        new_axisangle_abs = T.quat2axisangle(T.mat2quat(R_step))
        return new_pos, new_axisangle_abs

    def _eef_site_id(self):
        return self.env.sim.model.site_name2id("gripper0_right_grip_site")

    def get_real_time_action(self, step_num):
        """Compute absolute pose action for current phase."""
        eef_id = self._eef_site_id()
        eef_pos = self.env.sim.data.site_xpos[eef_id].copy()
        eef_rot = self.env.sim.data.site_xmat[eef_id].reshape(3, 3).copy()
        can_pos = self.env.sim.data.body_xpos[self.env.can_body_id].copy()
        goal_pos = self.env.sim.data.body_xpos[self.env.target_body_id].copy()
        table_z = float(self.env.table_offset[2])

        action = np.zeros(self.env.action_dim)
        target_ori_mat = eef_rot  # keep current orientation throughout

        # Phase logic
        if self.current_phase == 0:  # move above the can
            target_pos = can_pos.copy()
            target_pos[2] = max(can_pos[2] + 0.15, table_z + 0.25)

            if np.linalg.norm(eef_pos - target_pos) < 0.05:
                self.current_phase = 1
                self.phase_start_step = step_num

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        elif self.current_phase == 1:  # descend to grasp height
            target_pos = can_pos.copy()
            target_pos[2] = can_pos[2] - 0.02

            if np.linalg.norm(eef_pos - target_pos) < 0.03:
                self.current_phase = 2
                self.phase_start_step = step_num
                self.freeze_pos = eef_pos.copy()
                self.freeze_ori_mat = eef_rot.copy()

            step_pos, step_aa = self._limit_abs_pose_aniso(
                eef_pos, eef_rot, target_pos, target_ori_mat,
                max_step_xy=float('inf'),
                max_step_z=self.max_pos_step,
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        elif self.current_phase == 2:  # brief wait before closing
            if step_num - self.phase_start_step >= 8:
                self.current_phase = 3
                self.phase_start_step = step_num

            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = -1

        elif self.current_phase == 3:  # close gripper
            if step_num - self.phase_start_step > 15:
                self.current_phase = 4
                self.phase_start_step = step_num

            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = 1

        elif self.current_phase == 4:  # lift the can
            target_pos = can_pos.copy()
            target_pos[2] = max(table_z + 0.3, can_pos[2] + 0.25)

            if (can_pos[2] > table_z + 0.22) or (step_num - self.phase_start_step > 40):
                self.current_phase = 5
                self.phase_start_step = step_num

            # Maintain grasp orientation during transport
            transport_ori = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot
            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, transport_ori)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 5:  # move above the goal
            if self._transit_z is None:
                self._transit_z = max(table_z + 0.25, can_pos[2])

            target_pos = goal_pos.copy()
            target_pos[2] = self._transit_z

            if np.linalg.norm(eef_pos[:2] - target_pos[:2]) < 0.03 and abs(eef_pos[2] - target_pos[2]) < 0.06:
                self.current_phase = 6
                self.phase_start_step = step_num
                self._transit_z = None

            transport_ori = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot
            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, transport_ori)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 6:  # descend to place height
            target_pos = goal_pos.copy()
            target_pos[2] = table_z + 0.06

            if np.linalg.norm(eef_pos - target_pos) < 0.03:
                self.current_phase = 7
                self.phase_start_step = step_num

            transport_ori = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot
            step_pos, step_aa = self._limit_abs_pose_aniso(
                eef_pos, eef_rot, target_pos, transport_ori,
                max_step_xy=0.04,
                max_step_z=0.012,
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 7:  # open to release
            target_pos = goal_pos.copy()
            target_pos[2] = table_z + 0.02

            if step_num - self.phase_start_step >= 15:
                self.current_phase = 8
                self.phase_start_step = step_num

            release_ori = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot
            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, release_ori)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        else:  # 8: retreat
            target_pos = goal_pos.copy()
            target_pos[2] = table_z + 0.25

            if step_num - self.phase_start_step >= 10:
                self.done = True

            retreat_ori = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot
            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, retreat_ori)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        return action


def create_can_abs_debug_env():
    """Create CanRand env configured for absolute OSC_POSE (world frame)."""
    ctrl_cfg = load_composite_controller_config(robot="Panda")
    ctrl_cfg["body_parts"]["right"]["input_type"] = "absolute"
    ctrl_cfg["body_parts"]["right"]["input_ref_frame"] = "world"

    env = suite.make(
        env_name="CanRand",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        reward_shaping=False,
        hard_reset=False,
        controller_configs=ctrl_cfg,
    )
    return env


def debug_can_abs_motion_planning(num_runs=10, max_steps=400):
    env = create_can_abs_debug_env()
    env.reset()

    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    if hasattr(viewer, "opt"):
        viewer.opt.geomgroup[0] = 0

    successes = 0
    sec_per_step = 1.0 / 20.0
    for ep in range(num_runs):
        ctrl = CanAbsMotionPlanningController(env)
        env.reset()
        viewer.sync()
        for step in range(max_steps):
            if not viewer.is_running():
                break
            action = ctrl.get_real_time_action(step)
            env.step(action)
            viewer.sync()
            if ctrl.done:
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
    debug_can_abs_motion_planning()



