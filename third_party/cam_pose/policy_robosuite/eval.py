import os, random, math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import einops
import json
from contextlib import nullcontext
import h5py
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF
from cam_embedding import PluckerEmbedder

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
        
        # Add text overlays if provided
        if reward_list is not None or success_list is not None:
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()
            
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
    Class to evaluate policies on robosuite environments
    """
    def __init__(self, env, norm_stats, dataset_path, args):

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
        self.intrinsics = self._get_camera_intrinsics()
        
        
        camera_poses_dir = args.camera_poses_dir
        self.num_side_cam = int(args.num_side_cam)
        if not args.default_cam:
            self.camera_poses_by_name = {}
            for filename in args.pose_files:
                poses_path = os.path.join(camera_poses_dir, filename)
                with open(poses_path, 'r') as f:
                    raw = json.load(f)
                pose_name = os.path.splitext(filename)[0]
                self.camera_poses_by_name[pose_name] = raw['poses']
                print(f"Loaded {len(raw['poses'])} camera poses (old format) from {poses_path}; key={pose_name}; num_side_cam={self.num_side_cam}")
        else:
            print("Evaluator: default_cam=True; using agentview pose duplicated if needed")

        # Detect action space from dataset metadata
        with h5py.File(dataset_path, 'r') as f:
            action_space_attr = f['data'].attrs['action_space']
        self.action_space = action_space_attr.decode('utf-8') if isinstance(action_space_attr, bytes) else action_space_attr
    
    def _get_camera_intrinsics(self):
        """Extract camera intrinsics from robosuite environment."""
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        # Get field of view and image dimensions
        fovy = self.env.sim.model.cam_fovy[cam_id] * np.pi / 180.0
        width, height = 256, 256
        
        # Compute focal length
        focal_length = height / (2 * np.tan(fovy / 2))
        
        # Create intrinsics matrix
        intrinsics = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics
    
    def _set_camera_pose(self, cam_to_world):
        """Set camera pose in robosuite environment."""
        
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        # Set camera position
        self.env.sim.model.cam_pos[cam_id] = cam_to_world[:3, 3]
        
        # Set camera orientation (convert rotation matrix to quaternion)
        rotation = Rotation.from_matrix(cam_to_world[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        # MuJoCo uses [w, x, y, z] format
        self.env.sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]
    
    def _render_cam_image(self, cam_pose_raw):
        if cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            self._set_camera_pose(cam_pose)
        self.env.sim.forward()
        img = self.env.sim.render(camera_name="agentview", height=256, width=256, depth=False)
        return np.flipud(img).copy()

    def _image_to_tensor(self, cam_img, cam_pose_raw):
        rgb_tensor = einops.rearrange(torch.from_numpy(cam_img).float() / 255.0, 'h w c -> c h w')
        if self.args.use_plucker and cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            intrinsics_tensor = torch.from_numpy(self.intrinsics).unsqueeze(0).float().cuda()
            cam_to_world_tensor = torch.from_numpy(cam_pose).unsqueeze(0).float().cuda()
            with torch.no_grad():
                plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                plucker_tensor = einops.rearrange(plucker_data['plucker'][0].cpu(), 'h w c -> c h w')
        else:
            plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2])
        return torch.cat([rgb_tensor, plucker_tensor], dim=0)
    
    def evaluate(self, policy, save_path, video_prefix, pose_name, init_state=None, episode_num=0):
        np.random.seed(episode_num)
        
        # Initialize environment
        if init_state is not None:
            self.env.reset()
            self.env.sim.set_state_from_flattened(init_state)
        else:
            self.env.reset()

        if self.action_space in ('eef_delta', 'joint_delta'):
            self.env.set_init_action()

        camera_frames, success_labels, rewards, success = [], [], [], []
        done = False
        step = 0
        has_succeeded = False

        if self.args.default_cam:
            pose_set = [None] * self.num_side_cam
        else:
            poses_list = self.camera_poses_by_name[pose_name]
            if self.num_side_cam == 1:
                pose_set = [poses_list[episode_num]]
            else:
                pose_set = [poses_list[2 * episode_num], poses_list[2 * episode_num + 1]]
        
        while not done and step < self.max_steps:
            per_cam_images = [self._render_cam_image(p) for p in pose_set]
            camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
            camera_frames.append(camera_frame)
            success_labels.append(has_succeeded)

            # Build model input [1, num_cameras, C, H, W]
            per_cam_tensors = [self._image_to_tensor(img, p) for img, p in zip(per_cam_images, pose_set)]
            image_tensor = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).cuda()

            # Camera extrinsics: always two entries, shape [1, 2, 4, 4]
            if self.args.use_cam_pose and not self.args.default_cam:
                per_cam_T = []
                for i in range(2):
                    if i < len(pose_set) and pose_set[i] is not None:
                        per_cam_T.append(torch.from_numpy(np.array(pose_set[i], dtype=np.float32)).float().cuda())
                    else:
                        per_cam_T.append(torch.zeros(4, 4, device='cuda'))
                cam_extrinsics = torch.stack(per_cam_T, dim=0).unsqueeze(0)
            else:
                cam_extrinsics = torch.zeros(1, 2, 4, 4, device='cuda')
            
            state_vector = self.env.sim.data.qpos[:7]
            if np.random.rand() < self.args.prob_drop_proprio:
                state_vector = np.zeros_like(state_vector)
            normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
            state_tensor = einops.rearrange(torch.tensor(normalized_state, device="cuda").float(), 'd -> 1 d')
            
            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if self.args.use_fp16 else nullcontext()):
                action_chunk = policy({'qpos': state_tensor, 'image': image_tensor, 'cam_extrinsics': cam_extrinsics})
            action_chunk = action_chunk[0].float().cpu().numpy() * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()
            
            # Execute action chunk
            for i in range(action_chunk.shape[0]):                
                if done or step >= self.max_steps:
                    break

                next_obs, reward, done, info = self.env.step(action_chunk[i])
                
                current_success = (reward == 1)
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1
                
                if episode_num < self.eval_save_n_video and i < action_chunk.shape[0] - 1:
                    per_cam_images = [self._render_cam_image(p) for p in pose_set]
                    camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                    camera_frames.append(camera_frame)
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