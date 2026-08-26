import numpy as np
import robosuite as suite
import h5py
import robosuite.utils.transform_utils as T
from collections import defaultdict
import json
from tqdm import tqdm
from robosuite.controllers import load_composite_controller_config

# Import the motion planning controller from the existing file
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mp_lift_abs import LiftAbsMotionPlanningController
from mp_can_abs import CanAbsMotionPlanningController
from mp_square_abs import SquareAbsMotionPlanningController

VERBOSE = False

# Constants
EEF_SITE_NAME = "gripper0_right_grip_site"

# Task to environment name mapping
TASK_TO_ENV = {
    "liftrand": "LiftRand",
    "canrand": "CanRand",
    "squarerand": "SquareRand",
}

def get_eef_site_pose(env):
    """Returns (pos, axis-angle) for the EEF site in world frame."""
    site_id = env.sim.model.site_name2id(EEF_SITE_NAME)
    pos = env.sim.data.site_xpos[site_id].copy()
    rot_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
    aa = T.quat2axisangle(T.mat2quat(rot_mat))
    return pos, aa

def create_demo_env(task: str):
    """Create selected environment configured for absolute, world-frame OSC_POSE."""

    if task not in TASK_TO_ENV:
        raise ValueError(f"Unknown task: {task}")
    env_name = TASK_TO_ENV[task]

    controller_configs = load_composite_controller_config(robot="Panda")
    controller_configs["body_parts"]["right"]["input_type"] = "absolute"
    controller_configs["body_parts"]["right"]["input_ref_frame"] = "world"

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        reward_shaping=False,
        camera_depths=False,
        camera_heights=84,
        camera_widths=84,
        controller_configs=controller_configs,
    )

    return env

def extract_observation_data(obs, env):
    """Extract observation data in the same format as reference dataset"""
    obs_data = {}
    
    # Map object-state to object for compatibility with reference dataset
    obs_data['object'] = obs['object-state']
    
    # Extract robot observations
    obs_data['robot0_eef_pos'] = obs['robot0_eef_pos'] 
    obs_data['robot0_eef_quat'] = obs['robot0_eef_quat']
    obs_data['robot0_gripper_qpos'] = obs['robot0_gripper_qpos']
    obs_data['robot0_gripper_qvel'] = obs['robot0_gripper_qvel']
    obs_data['robot0_joint_pos'] = obs['robot0_joint_pos']
    obs_data['robot0_joint_pos_cos'] = obs['robot0_joint_pos_cos']
    obs_data['robot0_joint_pos_sin'] = obs['robot0_joint_pos_sin']
    obs_data['robot0_joint_vel'] = obs['robot0_joint_vel']
    
    # Calculate velocity observations from robot state
    eef_site_id = env.sim.model.site_name2id(EEF_SITE_NAME)
    
    env.sim.forward()  # Ensure state is updated
    
    # Get body velocity
    body_id = env.sim.model.site_bodyid[eef_site_id]
    obs_data['robot0_eef_vel_lin'] = env.sim.data.cvel[body_id][:3].copy()
    obs_data['robot0_eef_vel_ang'] = env.sim.data.cvel[body_id][3:].copy()
    
    return obs_data


def generate_single_demo(demo_id, action_spaces, seed=None, task: str = "liftrand"):
    """Generate a single demo using motion planning for the selected task, recording actions for multiple spaces.

    Args:
        demo_id (int): index of the demo
        action_spaces (list[str]): list of action space names in {"eef_delta", "eef_abs", "joint_abs", "joint_delta"}
        seed (int|None): seed for reproducibility
        task (str): task name

    Returns:
        tuple[dict[str, dict], bool]: (episode data per action space, success flag)
    """
    if VERBOSE:
        print(f"Generating demo {demo_id}...")
    
    # Set seed for reproducible trajectories across action spaces
    if seed is not None:
        np.random.seed(seed + demo_id)
    
    # Always use absolute-pose controller for planning
    env = create_demo_env(task)
    env.reset()
    
    # Create absolute-pose motion planning controller per task
    if task == "liftrand":
        mp_controller = LiftAbsMotionPlanningController(env)
    elif task == "canrand":
        mp_controller = CanAbsMotionPlanningController(env)
    elif task == "squarerand":
        mp_controller = SquareAbsMotionPlanningController(env)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Data storage (common across action spaces)
    episode_common = {
        'obs': defaultdict(list),
        'next_obs': defaultdict(list),
        'rewards': [],
        'dones': [],
        'states': [],
    }
    # Actions per requested space
    actions_by_space = {space: [] for space in action_spaces}

    # Track previous absolute pose command for delta construction (pos+aa combined)
    prev_abs_pose = None
    starting_abs_eef_action = None
    
    # Get initial observation
    obs = env._get_observations()
    initial_obs_data = extract_observation_data(obs, env)
    
    # Initialize previous joint positions for joint_delta computation
    prev_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
    
    success = False
    max_steps = 400
    reward = 0
    
    for step in range(max_steps):
        # Get real-time action and measure current EEF site pose BEFORE applying action
        osc_action = mp_controller.get_real_time_action(step)
        pre_site_pos, pre_site_aa = get_eef_site_pose(env)
        if starting_abs_eef_action is None:
            starting_abs_eef_action = osc_action.copy()
        
        # Store current observation
        for key, value in initial_obs_data.items():
            episode_common['obs'][key].append(value.copy())
        
        # Store current state
        current_state = env.sim.get_state().flatten()
        episode_common['states'].append(current_state.copy())
        
        # Execute OSC_POSE action
        next_obs, reward, done, info = env.step(osc_action)
        next_obs_data = extract_observation_data(next_obs, env)
        
        # Store next observation
        for key, value in next_obs_data.items():
            episode_common['next_obs'][key].append(value.copy())
        
        # Record actions for all requested spaces
        for space in action_spaces:
            if space == "joint_abs":
                joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
                gripper_action = osc_action[-1:]
                joint_action = np.concatenate([joint_pos, gripper_action])
                actions_by_space[space].append(joint_action)
            elif space == "joint_delta":
                current_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
                joint_delta = current_joint_pos - prev_joint_pos
                gripper_action = osc_action[-1:]
                joint_delta_action = np.concatenate([joint_delta, gripper_action])
                actions_by_space[space].append(joint_delta_action)
            elif space == "eef_abs":
                actions_by_space[space].append(osc_action.copy())
            elif space == "eef_delta":
                abs_pose = osc_action[0:6]
                if prev_abs_pose is None:
                    # Initialize previous pose from measured site for correct first delta
                    prev_abs_pose = np.concatenate([pre_site_pos, pre_site_aa])
                delta_pose = abs_pose - prev_abs_pose
                delta_in = np.concatenate([delta_pose, osc_action[-1:]])
                actions_by_space[space].append(delta_in)
            else:
                raise ValueError(f"Invalid action space: {space}")

        current_joint_pos_for_update = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
        prev_joint_pos = current_joint_pos_for_update
        prev_abs_pose = osc_action[0:6].copy()
        
        # Store reward and done
        episode_common['rewards'].append(reward)
        episode_common['dones'].append(0)  # Reference dataset uses 0 for all dones
        
        # Update for next iteration
        initial_obs_data = next_obs_data
        
        # Check success
        if env._check_success():
            if not success:  # First time success detected
                success = True
                success_step = step
            elif step >= success_step + 20:  # End 20 steps after success
                break
        
        # Also check controller's done flag
        if mp_controller.done:
            break
    
    env.close()
    
    # Pad the last action to match states length (per action space)
    for space in action_spaces:
        if len(actions_by_space[space]) < len(episode_common['states']):
            last_action = actions_by_space[space][-1].copy()
            actions_by_space[space].append(last_action)
    
    # Convert lists to numpy arrays
    # Convert common data
    common_np = {}
    common_np['obs'] = {k: np.array(v) for k, v in episode_common['obs'].items()}
    common_np['next_obs'] = {k: np.array(v) for k, v in episode_common['next_obs'].items()}
    common_np['rewards'] = np.array(episode_common['rewards'])
    common_np['dones'] = np.array(episode_common['dones'], dtype=np.int64)
    common_np['states'] = np.array(episode_common['states'])

    # Build per-space final episode data
    final_episode_data_by_space = {}
    for space in action_spaces:
        final_episode_data_by_space[space] = {
            'obs': common_np['obs'],
            'next_obs': common_np['next_obs'],
            'actions': np.array(actions_by_space[space]),
            'rewards': common_np['rewards'],
            'dones': common_np['dones'],
            'states': common_np['states'],
        }
        if space == 'eef_delta':
            final_episode_data_by_space[space]['starting_abs_action'] = starting_abs_eef_action

    if VERBOSE:
        lengths = {s: len(a) for s, a in actions_by_space.items()}
        print(f"Demo {demo_id}: {'SUCCESS' if success else 'FAILED'}, lengths: {lengths}")
    return final_episode_data_by_space, success


def generate_demos(
    num_demos=10,
    output_files=None,
    action_spaces=None,
    seed=None,
    task: str = "liftrand",
    output_dir=None,
):
    """Generate multiple demos and save separate HDF5 files per action space.

    Args:
        num_demos (int): number of demos
        output_files (list[str]): output filenames per action space
        action_spaces (list[str]): list of action spaces to record
        seed (int|None): seed
        task (str): task name
        output_dir (str|None): destination directory. Defaults to generated_demos
            next to this script.

    Returns:
        list[str]: absolute output paths aligned with action_spaces order
    """

    assert len(output_files) == len(action_spaces), "output_files must have the same length as action_spaces"

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_demos")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_paths = [os.path.join(output_dir, fname) for fname in output_files]

    # Generate demos once per demo id; collect per-space outputs
    all_demos_by_space = {space: {} for space in action_spaces}
    successful_demos = 0

    for i in tqdm(range(num_demos), desc=f"Generating {task} demos for {action_spaces}", unit="demo"):
        demo_data_by_space, success = generate_single_demo(i, action_spaces, seed=seed, task=task)
        for space in action_spaces:
            all_demos_by_space[space][f"demo_{i}"] = demo_data_by_space[space]
        if success:
            successful_demos += 1

    if VERBOSE:
        print(f"\nGenerated {num_demos} demos, {successful_demos} successful")

    # Save per action space
    env_name_meta = {"liftrand": "LiftRand", "canrand": "CanRand", "squarerand": "SquareRand"}[task]
    for space, out_path in zip(action_spaces, output_paths):
        # Controller config per space
        if space in ("joint_abs", "joint_delta"):
            controller_config = {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "JOINT_POSITION",
                        "input_type": "absolute",
                        "interpolation": None,
                        "gripper": {"type": "GRIP"},
                    }
                },
            }
        elif space in ("eef_delta", "eef_abs"):  # both use absolute controller metadata
            controller_config = {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "OSC_POSE",
                        "input_type": "absolute",
                        "input_ref_frame": "world",
                        "interpolation": None,
                        "gripper": {"type": "GRIP"},
                    }
                },
            }
        else:
            raise ValueError(f"Unknown action_space: {space}")

        env_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "controller_configs": controller_config,
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": 84,
            "camera_widths": 84,
            "reward_shaping": False,
        }
        env_args = json.dumps({"env_name": env_name_meta, "env_version": "1.4.1", "type": 1, "env_kwargs": env_kwargs})

        demos_for_space = all_demos_by_space[space]
        total_timesteps = sum(len(d['actions']) for d in demos_for_space.values())

        with h5py.File(out_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.attrs['env_args'] = env_args
            data_group.attrs['total'] = np.int64(total_timesteps)
            # Record the action space / type for this dataset (e.g., eef_abs, eef_delta, joint_abs, joint_delta)
            data_group.attrs['action_space'] = space

            for demo_name, demo_data in demos_for_space.items():
                demo_group = data_group.create_group(demo_name)
                demo_group.create_dataset('actions', data=demo_data['actions'])
                demo_group.create_dataset('rewards', data=demo_data['rewards'])
                demo_group.create_dataset('dones', data=demo_data['dones'])
                demo_group.create_dataset('states', data=demo_data['states'])
                # For delta action space, save the starting absolute EEF action
                if space == 'eef_delta':
                    demo_group.create_dataset('starting_abs_action', data=demo_data['starting_abs_action'])

                obs_group = demo_group.create_group('obs')
                next_obs_group = demo_group.create_group('next_obs')
                for key, value in demo_data['obs'].items():
                    obs_group.create_dataset(key, data=value)
                for key, value in demo_data['next_obs'].items():
                    next_obs_group.create_dataset(key, data=value)

    if VERBOSE:
        for out_path in output_paths:
            print(f"Saved {num_demos} demos to {out_path}")

    return output_paths



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="squarerand", choices=["liftrand", "canrand", "squarerand"])
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument(
        "--output_files", type=str, nargs="+", default=["eef_abs.hdf5", "eef_delta.hdf5", "joint_abs.hdf5", "joint_delta.hdf5"],
        help="List of output filenames (one per action space). If omitted, defaults to {task}_{space}.hdf5"
    )
    parser.add_argument(
        "--action_spaces", type=str, nargs="+", default=["eef_abs", "eef_delta", "joint_abs", "joint_delta"],
        choices=["eef_delta", "eef_abs", "joint_abs", "joint_delta"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination directory. Defaults to script_robosuite_demos/generated_demos.",
    )
    args = parser.parse_args()

    generate_demos(
        num_demos=args.num_demos,
        output_files=args.output_files,
        action_spaces=args.action_spaces,
        seed=args.seed,
        task=args.task,
        output_dir=args.output_dir,
    )
