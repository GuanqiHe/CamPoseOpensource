import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
import time
import mujoco
import mujoco.viewer


class SquareAbsMotionPlanningController:
    """Absolute-pose motion planner for SquareRand (grasp handle + place on peg)."""

    def __init__(self, env):
        self.env = env

        # Speed caps per 20 Hz step
        self.max_pos_step = 0.02
        self.max_ori_step = 0.20

        # Phases:
        # 0=move_above_handle, 1=descend, 2=wait_stable, 3=close,
        # 4=hold_after_grasp, 5=lift, 6=move_above_peg, 7=descend_peg, 8=open, 9=retreat
        self.current_phase = 0
        self.phase_start_step = 0
        self.done = False
        self._transit_z = None

        # Stabilization and alignment
        self.stability_required_steps = 5
        self.linear_vel_threshold = 0.05
        self.position_threshold = 0.02
        self.yaw_align_threshold = 0.1
        self._stability_counter = 0

        # Cache ids
        self.handle_site = self.env.square.important_sites["handle"]
        self.handle_site_id = self.env.sim.model.site_name2id(self.handle_site)
        self.center_site = f"{self.env.square.naming_prefix}center_site"
        self.center_site_id = self.env.sim.model.site_name2id(self.center_site)
        self.eef_site_name = "gripper0_right_grip_site"
        self.eef_site_id = self.env.sim.model.site_name2id(self.eef_site_name)

        # Latched pose used during wait / close
        self.freeze_pos = None
        self.freeze_ori_mat = None

    def _limit_abs_pose(self, curr_pos, curr_ori_mat, targ_pos, targ_ori_mat):
        """Step-limit absolute pose to cap speed."""
        # Position
        delta_pos = targ_pos - curr_pos
        dist = np.linalg.norm(delta_pos)
        if dist > self.max_pos_step and dist > 1e-12:
            delta_pos = delta_pos / dist * self.max_pos_step
        new_pos = curr_pos + delta_pos

        # Orientation
        R_delta = curr_ori_mat.T @ targ_ori_mat
        delta_axisangle = T.quat2axisangle(T.mat2quat(R_delta))
        mag = np.linalg.norm(delta_axisangle)
        if mag > self.max_ori_step and mag > 1e-12:
            delta_axisangle = delta_axisangle / mag * self.max_ori_step
        R_step = curr_ori_mat @ T.quat2mat(T.axisangle2quat(delta_axisangle))
        new_axisangle_abs = T.quat2axisangle(T.mat2quat(R_step))
        return new_pos, new_axisangle_abs

    def _limit_abs_pose_aniso(self, curr_pos, curr_ori_mat, targ_pos, targ_ori_mat, max_step_xy, max_step_z):
        """Anisotropic step-limit absolute pose: larger XY vs smaller Z step."""
        # Position (anisotropic)
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

        # Orientation (same limiter as isotropic)
        R_delta = curr_ori_mat.T @ targ_ori_mat
        delta_axisangle = T.quat2axisangle(T.mat2quat(R_delta))
        mag = np.linalg.norm(delta_axisangle)
        if mag > self.max_ori_step and mag > 1e-12:
            delta_axisangle = delta_axisangle / mag * self.max_ori_step
        R_step = curr_ori_mat @ T.quat2mat(T.axisangle2quat(delta_axisangle))
        new_axisangle_abs = T.quat2axisangle(T.mat2quat(R_step))
        return new_pos, new_axisangle_abs

    @staticmethod
    def _wrap_angle(a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _calc_yaw_from_fingers(self, finger_dir_xy, v_hc_xy):
        """Return small yaw correction (around z) from finger direction to be ⟂ to v_hc.

        The correction is signed and limited in magnitude per step.
        Also returns the absolute yaw error for thresholding.
        """
        v = v_hc_xy / (np.linalg.norm(v_hc_xy) + 1e-12)
        t1 = np.array([-v[1], v[0]])
        t2 = np.array([v[1], -v[0]])
        f = finger_dir_xy / (np.linalg.norm(finger_dir_xy) + 1e-12)

        def signed_angle(a, b):
            a = a / (np.linalg.norm(a) + 1e-12)
            b = b / (np.linalg.norm(b) + 1e-12)
            cr = a[0] * b[1] - a[1] * b[0]
            dp = a[0] * b[0] + a[1] * b[1]
            return np.arctan2(cr, dp)

        err1 = signed_angle(f, t1)
        err2 = signed_angle(f, t2)
        yaw_err = err1 if abs(err1) <= abs(err2) else err2
        if abs(yaw_err) < 0.02:
            return np.zeros(3), 0.0
        yaw_err = float(np.clip(yaw_err, -self.max_ori_step, self.max_ori_step))
        return np.array([0.0, 0.0, yaw_err]), min(abs(err1), abs(err2))

    def _finger_dir_xy(self):
        gr = self.env.robots[0].gripper
        if isinstance(gr, dict):
            gr = next(iter(gr.values()))
        left_geom = gr.important_geoms["left_fingerpad"]
        right_geom = gr.important_geoms["right_fingerpad"]
        left_name = left_geom[0] if isinstance(left_geom, (list, tuple)) else left_geom
        right_name = right_geom[0] if isinstance(right_geom, (list, tuple)) else right_geom
        left_gid = self.env.sim.model.geom_name2id(left_name)
        right_gid = self.env.sim.model.geom_name2id(right_name)
        pL = self.env.sim.data.geom_xpos[left_gid]
        pR = self.env.sim.data.geom_xpos[right_gid]
        return (pR - pL)[:2]

    def get_real_time_action(self, step_num):
        # Current states
        eef_pos = self.env.sim.data.site_xpos[self.eef_site_id].copy()
        eef_rot = self.env.sim.data.site_xmat[self.eef_site_id].reshape(3, 3).copy()
        handle_pos = self.env.sim.data.site_xpos[self.handle_site_id].copy()
        square_center = self.env.sim.data.site_xpos[self.center_site_id].copy()
        peg_pos = self.env.sim.data.body_xpos[self.env.peg_body_id].copy()
        table_z = float(self.env.table_offset[2])

        action = np.zeros(self.env.action_dim)

        # Default target orientation: keep current
        target_ori_mat = eef_rot

        # Phases
        if self.current_phase == 0:  # move above handle
            target_pos = handle_pos.copy()
            target_pos[2] = max(handle_pos[2] + 0.12, table_z + 0.25)

            # Yaw control using finger direction vs. handle->center direction
            v = (square_center - handle_pos)[:2]
            if np.linalg.norm(v) < 1e-6:
                v = np.array([1.0, 0.0])
            finger_dir = self._finger_dir_xy()
            # Print dot product (should be near 0 when perpendicular)
            # v_n = v / (np.linalg.norm(v) + 1e-12)
            # f_n = finger_dir / (np.linalg.norm(finger_dir) + 1e-12)
            # print(f"dot={np.dot(v_n, f_n):.3f}")
            yaw_ctrl, yaw_err_abs = self._calc_yaw_from_fingers(finger_dir, v)

            # Convert desired incremental yaw into absolute target orientation
            dR = T.quat2mat(T.axisangle2quat(yaw_ctrl))
            # Apply yaw in world frame (pre-multiply)
            target_ori_mat = dR @ eef_rot

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

            if np.linalg.norm(eef_pos - target_pos) < 0.05 and yaw_err_abs < self.yaw_align_threshold:
                self.current_phase = 1
                self.phase_start_step = step_num

        elif self.current_phase == 1:  # descend to handle height
            target_pos = handle_pos.copy()
            target_pos[2] = handle_pos[2] - 0.02

            v = (square_center - handle_pos)[:2]
            if np.linalg.norm(v) < 1e-6:
                v = np.array([1.0, 0.0])
            finger_dir = self._finger_dir_xy()
            yaw_ctrl, _ = self._calc_yaw_from_fingers(finger_dir, v)
            dR = T.quat2mat(T.axisangle2quat(yaw_ctrl))
            # Apply yaw in world frame (pre-multiply)
            target_ori_mat = dR @ eef_rot

            # Unlimited XY while descending in Z towards the handle
            step_pos, step_aa = self._limit_abs_pose_aniso(
                eef_pos, eef_rot, target_pos, target_ori_mat,
                max_step_xy=float('inf'),
                max_step_z=self.max_pos_step,
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

            if np.linalg.norm(eef_pos - target_pos) < 0.03:
                self.current_phase = 2
                self.phase_start_step = step_num
                self.freeze_pos = eef_pos.copy()
                self.freeze_ori_mat = eef_rot.copy()

        elif self.current_phase == 2:  # brief wait / halt before closing
            if step_num - self.phase_start_step >= 5:
                # Perpendicularity check using inner product (should be near 0)
                v_hc = (square_center - handle_pos)[:2]
                if np.linalg.norm(v_hc) > 1e-8:
                    v_hc = v_hc / np.linalg.norm(v_hc)
                    finger_vec = self._finger_dir_xy()
                    if np.linalg.norm(finger_vec) > 1e-8:
                        finger_dir = finger_vec / np.linalg.norm(finger_vec)
                        dot = float(abs(np.dot(v_hc, finger_dir)))
                        ang = float(np.degrees(np.arccos(np.clip(np.dot(v_hc, finger_dir), -1.0, 1.0))))
                        print(f"Perp check (grasp prep): dot={dot:.3f}, angle={ang:.1f} deg")
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

            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = 1

        elif self.current_phase == 4:  # brief hold after grasp
            if step_num - self.phase_start_step >= 5:
                self.current_phase = 5
                self.phase_start_step = step_num

            hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat))
            action[0:3] = self.freeze_pos
            action[3:6] = hold_aa
            action[-1] = 1

        elif self.current_phase == 5:  # lift square to transport height
            current_sq_z = float(self.env.sim.data.body_xpos[self.env.square_body_id][2])
            target_pos = np.array([
                square_center[0],
                square_center[1],
                max(table_z + 0.3, current_sq_z + 0.25),
            ])
            target_ori_mat = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot

            if (current_sq_z > table_z + 0.22) or (step_num - self.phase_start_step > 40):
                self.current_phase = 6
                self.phase_start_step = step_num

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 6:  # move above the peg
            if self._transit_z is None:
                current_sq_z = float(self.env.sim.data.body_xpos[self.env.square_body_id][2])
                self._transit_z = max(table_z + 0.25, current_sq_z)

            offset_sc_e = (square_center - eef_pos)[:2]
            desired_xy = peg_pos[:2] - offset_sc_e
            target_pos = np.array([desired_xy[0], desired_xy[1], self._transit_z])
            target_ori_mat = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot

            if np.linalg.norm(eef_pos[:2] - desired_xy) < 0.03 and abs(eef_pos[2] - target_pos[2]) < 0.06:
                self.current_phase = 7
                self.phase_start_step = step_num
                self._transit_z = None

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 7:  # descend to place above peg
            offset_sc_e = (square_center - eef_pos)[:2]
            desired_xy = peg_pos[:2] - offset_sc_e
            target_pos = np.array([desired_xy[0], desired_xy[1], table_z + 0.06])
            target_ori_mat = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot

            if np.linalg.norm(eef_pos[:2] - desired_xy) < 0.03 and abs(eef_pos[2] - target_pos[2]) < 0.03:
                self.current_phase = 8
                self.phase_start_step = step_num

            # Allow faster XY alignment while descending slowly in Z
            step_pos, step_aa = self._limit_abs_pose_aniso(
                eef_pos, eef_rot, target_pos, target_ori_mat,
                max_step_xy=0.04,  # relax XY limit for quicker centering
                max_step_z=0.012,   # conservative Z to avoid overshoot
            )
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = 1

        elif self.current_phase == 8:  # open to release on peg
            target_pos = peg_pos.copy()
            target_pos[2] = table_z + 0.02
            target_ori_mat = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot

            if step_num - self.phase_start_step >= 15:
                self.current_phase = 9
                self.phase_start_step = step_num

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        else:  # 9: retreat
            target_pos = peg_pos.copy()
            target_pos[2] = table_z + 0.25
            target_ori_mat = self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot

            if step_num - self.phase_start_step >= 10:
                self.done = True

            step_pos, step_aa = self._limit_abs_pose(eef_pos, eef_rot, target_pos, target_ori_mat)
            action[0:3] = step_pos
            action[3:6] = step_aa
            action[-1] = -1

        # else:  # hold
        #     hold_aa = T.quat2axisangle(T.mat2quat(self.freeze_ori_mat if self.freeze_ori_mat is not None else eef_rot))
        #     action[0:3] = self.freeze_pos if self.freeze_pos is not None else eef_pos
        #     action[3:6] = hold_aa
        #     action[-1] = 1

        return action


def create_square_abs_debug_env():
    ctrl_cfg = load_composite_controller_config(robot="Panda")
    ctrl_cfg["body_parts"]["right"]["input_type"] = "absolute"
    ctrl_cfg["body_parts"]["right"]["input_ref_frame"] = "world"

    env = suite.make(
        env_name="SquareRand",
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


def debug_square_abs_motion_planning(num_runs=10, max_steps=300):
    env = create_square_abs_debug_env()
    env.reset()

    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    if hasattr(viewer, "opt"):
        viewer.opt.geomgroup[0] = 0

    successes = 0
    sec_per_step = 1.0 / 20.0
    for ep in range(num_runs):
        ctrl = SquareAbsMotionPlanningController(env)
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

        grasped = ctrl.done
        successes += int(grasped)
        print(f"Episode {ep+1}/{num_runs}: {'GRASPED' if grasped else 'TIMEOUT'}")
        if not viewer.is_running():
            break

    print(f"Finished {ep+1 if viewer.is_running() else ep+1} episode(s). Grasp completions: {successes}")

    while viewer.is_running():
        mujoco.mj_step(env.sim.model._model, env.sim.data._data)
        viewer.sync()
        time.sleep(sec_per_step)

    env.close()
    return successes


if __name__ == "__main__":
    debug_square_abs_motion_planning()


