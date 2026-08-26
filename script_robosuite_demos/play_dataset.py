import numpy as np
import robosuite as suite
import h5py
import imageio
import os
import sys
import json
import cv2
from robosuite.wrappers.action_wrapper import wrap_env_action_space


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            return (frame * 255).astype(np.uint8)
        return frame.astype(np.uint8)
    return frame


def _render_with_status(env, success: bool) -> np.ndarray:
    frame = env.sim.render(camera_name="frontview", height=480, width=640)
    frame = np.flipud(frame)
    frame = _ensure_uint8(frame)
    h, w = frame.shape[:2]
    text = "SUCCESS" if success else "FAILURE"
    bgr_color = (0, 255, 0) if success else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    ((text_w, text_h), baseline) = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - text_w - 10
    y = 10 + text_h
    frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    cv2.putText(frame_bgr, text, (x, y), font, font_scale, bgr_color, thickness, lineType=cv2.LINE_AA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def create_replay_env_from_dataset(dataset_path):
    """Create environment that matches the dataset configuration (env_name + env_kwargs).

    For eef_abs and eef_delta datasets, force absolute world-frame OSC_POSE.
    """
    with h5py.File(dataset_path, 'r') as f:
        env_args_str = f['data'].attrs['env_args']
        action_space = f['data'].attrs['action_space']
        env_config = json.loads(env_args_str)
        env_name = env_config.get('env_name', 'LiftRand')
        env_kwargs = env_config.get('env_kwargs', {})

        controller_type = env_kwargs['controller_configs']['body_parts']['right']['type']

        # Override for replay rendering
        env_kwargs = dict(env_kwargs)
        env_kwargs.update({
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'use_camera_obs': False,
            'camera_heights': 480,
            'camera_widths': 640,
        })

        env = suite.make(env_name=env_name, **env_kwargs)
        return env, controller_type, action_space

def replay_demo(demo_name, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load demo data
    with h5py.File(dataset_path, 'r') as f:
        demo_group = f['data'][demo_name]
        actions = np.array(demo_group['actions'])
        states = np.array(demo_group['states'])
    
    # Create environment that matches the dataset
    env, controller_type, action_space = create_replay_env_from_dataset(dataset_path)
    env.reset()
    
    # Set initial state and capture frames
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    
    frames = []

    frames.append(_render_with_status(env, bool(env._check_success())))
    
    success_achieved = False
    success_step = -1

    # Wrap env to handle delta action spaces directly
    if action_space in ('eef_delta', 'joint_delta'):
        env = wrap_env_action_space(env, action_space)
        env.set_init_action()

    for step, action in enumerate(actions):
        env.step(action)
        frames.append(_render_with_status(env, bool(env._check_success())))
        if env._check_success() and not success_achieved:
            success_achieved = True
            success_step = step
    
    env.close()
    
    status = "SUCCESS" if success_achieved else "FAILED"
    success_info = f" at step {success_step}" if success_achieved else ""
    print(f"{demo_name} ({controller_type}): {status}{success_info}")
    
    # Save video with success/failure in filename
    status_suffix = "success" if success_achieved else "failed"
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]  # Extract filename without extension
    output_path = os.path.join(output_dir, f"{demo_name}_replay_{dataset_name}_{status_suffix}.mp4")
    # Write H.264 video with stderr silenced
    processed_frames = [_ensure_uint8(f) for f in frames]
    with open(os.devnull, 'w') as devnull:
        sys.stderr = devnull
        with imageio.get_writer(output_path, fps=20, codec='h264', ffmpeg_params=['-crf', '23', '-preset', 'medium']) as writer:
            for f in processed_frames:
                writer.append_data(f)
        sys.stderr = sys.__stderr__
    
    return success_achieved

def replay_dataset(dataset_path, num_demos=10, output_dir=None):
    """Replay demos from any dataset by auto-detecting controller configuration"""
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset_path), "replayed_demos")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    successful_replays = 0
    dataset_name = os.path.basename(dataset_path)

    
    
    print(f"\nReplaying {dataset_name}:")

    # Collect existing demo keys in lexicographic order
    with h5py.File(dataset_path, 'r') as fh:
        demo_keys = sorted([k for k in fh['data'].keys() if k.startswith('demo')])

    if num_demos is not None:
        demo_keys = demo_keys[:num_demos]

    for key in demo_keys:
        success = replay_demo(key, dataset_path, output_dir)
        if success:
            successful_replays += 1
         
 
    print(f"Dataset {dataset_name}: {successful_replays}/{num_demos} successful replays")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination directory. Defaults to <dataset-dir>/replayed_demos.",
    )
    args = parser.parse_args()

    replay_dataset(
        dataset_path=args.dataset_path,
        num_demos=args.num_demos,
        output_dir=args.output_dir,
    )
