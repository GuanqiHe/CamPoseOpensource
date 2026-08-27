import os, random, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import einops
import torchvision.transforms.functional as TF
from cam_embedding import PluckerEmbedder
import gymnasium as gym

def to_mp4(save_path, image_list, reward_list=None, success_list=None, info_list=None):
    """
    Save a list of images as an MP4 video with reward and success overlaid using imageio with H264 encoding.
    """
    import imageio
    import os
    
    # Convert images to list of numpy arrays if needed
    if isinstance(image_list, torch.Tensor):
        image_list = image_list.cpu().numpy()
    
    # Ensure the save path has .mp4 extension
    if not save_path.endswith('.mp4'):
        save_path = save_path.replace('.avi', '.mp4')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare frames with overlays
    frames = []
    for i, img in enumerate(image_list):
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Convert to PIL for overlay drawing
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()

        # Upper-right corner: current step index
        step_text = f"Step {i}"
        draw.text((img_pil.width - 10, 10), step_text, fill=(255, 255, 255), font=font, anchor='rt')

        # Upper-left overlays (optional): reward and success
        y_offset = 10
        if reward_list is not None and i < len(reward_list):
            draw.text((10, y_offset), f"Reward: {reward_list[i]:.3f}", fill=(255, 255, 255), font=font)
            y_offset += 30
        if success_list is not None and i < len(success_list):
            success_text = "SUCCESS" if success_list[i] else "FAILURE"
            color = (0, 255, 0) if success_list[i] else (255, 0, 0)
            draw.text((10, y_offset), success_text, fill=color, font=font)

        img = np.array(img_pil)
        
        frames.append(img)
    
    # Save video with H264 encoding, suppress warnings
    import sys
    
    # Redirect stderr to suppress libx264 warnings
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        
        with imageio.get_writer(save_path, fps=10, codec='h264', ffmpeg_params=['-crf', '23', '-preset', 'medium']) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        sys.stderr = old_stderr

class Evaluator:
    """
    Class to evaluate policies on ManiSkill environments
    """
    def __init__(self, env, norm_stats, args):

        self.env = env
        self.args = args
        self.norm_stats = {k: torch.tensor(v).float() for k, v in norm_stats.items()}
        self.chunk_size = args.chunk_size
        self.max_steps = args.eval_max_steps
        self.eval_save_n_video = args.eval_save_n_video
        
        self.H = 256
        self.W = 256

        self.success_by_seed = {}
        
        self.plucker_embedder = PluckerEmbedder(img_size=256, device='cuda')
        self.num_side_cam = args.num_side_cam
        
    
    def _get_camera_intrinsics(self, cam_name):
        camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
        return camera.get_params()["intrinsic_cv"].cpu()
    
    def _get_cam2world(self, cam_name):
        camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
        return camera.get_params()["cam2world_gl"].cpu()
    
    def evaluate(self, policy, save_path, video_prefix, pose_name, episode_num=0):
        np.random.seed(episode_num)
        self.env.reset(seed=episode_num)

        if self.args.default_cam:
            cam_names = ["render_camera"] * self.num_side_cam
        elif self.num_side_cam == 1:
            idx = episode_num if pose_name == 'train' else 500 + episode_num
            cam_names = [f'cam_{idx}']
        else:
            base = 0 if pose_name == 'train' else 500
            cam_names = [f'cam_{base + 2 * episode_num}', f'cam_{base + 2 * episode_num + 1}']
        print(f"Episode {episode_num}: Using cameras {cam_names}")

        camera_frames, success_labels, rewards, success = [], [], [], []
        done = False
        step = 0
        has_succeeded = False
        
        while not done and step < self.max_steps:
            per_cam_images = [self.env.unwrapped.render_rgb_array(n).cpu().numpy()[0] for n in cam_names]
            combined = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
            camera_frames.append(combined)
            success_labels.append(has_succeeded)

            per_cam_tensors = []
            for n, img in zip(cam_names, per_cam_images):
                rgb_tensor = einops.rearrange(torch.from_numpy(img).float() / 255.0, 'h w c -> c h w')
                if self.args.use_plucker:
                    camera = self.env.unwrapped.scene.human_render_cameras[n]
                    intrinsics_tensor = camera.get_params()["intrinsic_cv"]
                    cam_to_world_tensor = camera.get_params()["cam2world_gl"]
                    with torch.no_grad():
                        plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                        plucker_tensor = einops.rearrange(plucker_data['plucker'][0].cpu(), 'h w c -> c h w')
                else:
                    plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2])
                per_cam_tensors.append(torch.cat([rgb_tensor, plucker_tensor], dim=0))

            image_tensor = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).cuda()
            
            st = self.env.unwrapped.get_state_dict()
            for key in ("panda", "panda_wristcam", "panda_stick"):
                if key in st['articulations']:
                    qpos = st['articulations'][key][0, 13:22]
                    break
            state_vector = qpos.cpu().numpy()
            
            if np.random.rand() < self.args.prob_drop_proprio:
                state_vector = np.zeros_like(state_vector)
            normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
            state_tensor = einops.rearrange(torch.tensor(normalized_state, device="cuda").float(), 'd -> 1 d')
            
            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if self.args.use_fp16 else nullcontext()):
                action_chunk = policy({'qpos': state_tensor, 'image': image_tensor})
            action_chunk = action_chunk[0].float().cpu().numpy() * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()
            
            # Execute action chunk
            for i in range(action_chunk.shape[0]):                
                if done or step >= self.max_steps:
                    break

                res = self.env.step(torch.tensor(action_chunk[i], device='cuda'))
                obs, reward, terminated, truncated, info = res
                done = bool(terminated or truncated)
                current_success = info['success'][0].item() if isinstance(info['success'], torch.Tensor) else bool(info['success'])
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1
                
                if episode_num < self.eval_save_n_video and i < action_chunk.shape[0] - 1:
                    per_cam_images = [self.env.unwrapped.render_rgb_array(n).cpu().numpy()[0] for n in cam_names]
                    combined = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                    camera_frames.append(combined)
                    success_labels.append(has_succeeded)
        
        final_success = any(success)
        print(f"Episode {episode_num}: Success = {final_success}")
        
        self.success_by_seed[episode_num] = bool(final_success)
        
        if episode_num < self.eval_save_n_video:
            camera_video_path = os.path.join(save_path, f"{video_prefix}_{pose_name}_success_{final_success}.mp4")
            to_mp4(camera_video_path, camera_frames, success_list=success_labels)
        
        results = {
            "success_rate": float(final_success),
            "mean_episode_length": float(step),
            "max_rewards": rewards
        }
        
        return results, float(final_success), step

def main():
    """Main function to run dataset replay or policy evaluation"""
    dataset_path = '/share/data/ripl/tianchong/projects/CamPoseRobosuite/demos/lift/ph/low_dim_v141.hdf5'
    output_dir = '/share/data/ripl/tianchong/projects/CamPoseRobosuite/evaluation_results'
    # camera_name will be retrieved from args.camera_names[0]
    


if __name__ == '__main__':
    main()
