import numpy as np
import robosuite as suite
import h5py
import os
import json
import argparse
import time
import mujoco
import mujoco.viewer

def create_replay_env_from_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        
        env_args_str = data_group.attrs['env_args']
        env_config = json.loads(env_args_str)
        # controller_config = env_config['env_kwargs']['controller_configs']
    
        
        env = suite.make(
            env_name="DoorRand",
            robots="Panda",
            # controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            ignore_done=True,
            reward_shaping=False,
            control_freq=20,
        )
        
        return env

def get_demo_keys(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = [key for key in data_group.keys() if key.startswith('demo')]
        return sorted(demo_keys)

def replay_demo_interactive(demo_key, dataset_path, playback_speed, env, viewer):
    with h5py.File(dataset_path, 'r') as f:
        demo_group = f['data'][demo_key]
        actions = np.array(demo_group['actions'])
        states = np.array(demo_group['states'])
    
    print(f"Playing {demo_key}, {len(actions)} actions")
    
    viewer.sync()
    
    success_achieved = False
    current_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
    
    for step, action in enumerate(actions):
        if not viewer.is_running():
            break
            
        if "joint_delta" in dataset_path:
            current_joint_pos += action[:-1]
            action = action.copy()
            action[:-1] = current_joint_pos
        
        env.step(action)
        viewer.sync()
        
        if env._check_success() and not success_achieved:
            success_achieved = True
        
        time.sleep(0.05 / playback_speed)
    
    print(f"Finished {demo_key}")
    return success_achieved

def replay_dataset_interactive(dataset_path, demo_indices, playback_speed):
    demo_keys = get_demo_keys(dataset_path)
    demos_to_replay = [demo_keys[i] for i in demo_indices]
    
    env = create_replay_env_from_dataset(dataset_path)
    
    env.reset()
    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    if hasattr(viewer, "opt"):
        viewer.opt.geomgroup[0] = 0
    
    print(f"Starting replay of {len(demos_to_replay)} demos")
    
    successful_replays = 0
    
    for i, demo_key in enumerate(demos_to_replay):
        with h5py.File(dataset_path, 'r') as f:
            demo_group = f['data'][demo_key]
            states = np.array(demo_group['states'])
        
        # env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        
        if not viewer.is_running():
            break
        
        success = replay_demo_interactive(demo_key, dataset_path, playback_speed, env, viewer)
        if success:
            successful_replays += 1
        
        time.sleep(2.0)
    
    print("All demos finished, entering interactive mode")
    
    while viewer.is_running():
        mujoco.mj_step(env.sim.model._model, env.sim.data._data)
        viewer.sync()
        time.sleep(0.05)
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Interactive replay of robosuite demonstrations")
    parser.add_argument("dataset_path", type=str, nargs='?', 
                    #    default="/home/tianchongj/workspace/script_robosuite_demos/demos/square/ph/low_dim_v141.hdf5",
                        default="/home/tianchongj/workspace/script_robosuite_demos/demos/lift/mp_liftrand_eef_vel.hdf5",
                        # default="/home/tianchongj/workspace/script_robosuite_demos/demos/can/ph/low_dim_v141.hdf5",
                       help="Path to the HDF5 dataset file")
    parser.add_argument("--demos", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       help="Specific demo indices to replay (0-based)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    
    args = parser.parse_args()
    
    replay_dataset_interactive(
        args.dataset_path, 
        args.demos, 
        args.speed
    )

if __name__ == "__main__":
    main()
