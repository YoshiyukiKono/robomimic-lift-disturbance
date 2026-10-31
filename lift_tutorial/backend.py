"""A small explicit adapter: robomimic policy + robosuite's modern MuJoCo backend.

The original v0.1 policy weights are unchanged. This is not an exact reproduction
of the original offline_study simulator or its reported benchmark success rate.
"""

from copy import deepcopy
import json
import os

# Set before importing mujoco / OpenGL. Override in the shell to try egl/glfw.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from .checkpoint import verify_checkpoint


class LiftBackend:
    def __init__(self, checkpoint, device="cuda", camera="agentview", width=640, height=480):
        # Never unpickle an unidentified file.
        self.checkpoint_metadata = verify_checkpoint(checkpoint)
        import torch
        import robosuite
        import robomimic.utils.file_utils as file_utils

        torch.set_num_threads(1)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Run doctor; or explicitly use --device cpu.")
        self.device = torch.device(device)
        checkpoint_dict = torch.load(str(checkpoint), map_location="cpu")
        config = json.loads(checkpoint_dict["config"])
        # v0.3 expects this flag, absent in v0.1. This RNN does not use a transformer.
        config["algo"].setdefault("transformer", {"enabled": False})
        checkpoint_dict["config"] = json.dumps(config)
        self.policy, checkpoint_dict = file_utils.policy_from_checkpoint(
            ckpt_dict=checkpoint_dict, device=self.device, verbose=False
        )
        if checkpoint_dict["env_metadata"]["env_name"] != "Lift":
            raise ValueError("Only the verified Lift checkpoint is supported")
        if checkpoint_dict["shape_metadata"]["use_images"]:
            raise ValueError("This tutorial implements the low-dimensional policy only")
        self.observation_shapes = dict(checkpoint_dict["shape_metadata"]["all_shapes"])
        self.action_dimension = checkpoint_dict["shape_metadata"]["ac_dim"]
        self.camera, self.width, self.height = camera, width, height
        self.original_environment = deepcopy(checkpoint_dict["env_metadata"])
        kwargs = deepcopy(checkpoint_dict["env_metadata"]["env_kwargs"])
        kwargs.update(
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=False, use_object_obs=True, ignore_done=True,
            hard_reset=True, camera_names=[camera], camera_widths=width, camera_heights=height,
        )
        self.environment_kwargs = kwargs
        self.env = robosuite.make("Lift", **kwargs)
        self.control_freq = float(self.env.control_freq)

    def _policy_observation(self, raw):
        mapped = dict(raw)
        mapped["object"] = raw["object-state"]
        result = {}
        for key, shape in self.observation_shapes.items():
            if key not in mapped:
                raise ValueError(f"Missing checkpoint observation {key}")
            array = np.asarray(mapped[key])
            if tuple(array.shape) != tuple(shape) or not np.isfinite(array).all():
                raise ValueError(f"Invalid observation {key}: {array.shape}, expected {shape}")
            result[key] = array
        return result

    def reset(self, seed, x_m, y_m):
        import random
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.env.reset()
        self.set_cube_force([0, 0, 0])
        joint = self.env.cube.joints[0]
        qpos = self.env.sim.data.get_joint_qpos(joint).copy()
        # Preserve sampled orientation, geometry, and height; change only XY.
        qpos[0:2] = np.asarray(self.env.table_offset[:2]) + [x_m, y_m]
        self.env.sim.data.set_joint_qpos(joint, qpos)
        self.env.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.env.sim.forward()
        self.policy.start_episode()  # reset RNN memory for EVERY episode
        raw = self.env._get_observations(force_update=True)
        observation = self._policy_observation(raw)
        cube = self.cube_position()
        if not np.allclose(cube[:2], qpos[:2], atol=1e-8):
            raise RuntimeError("Cube reset position was not applied")
        # Persist sufficient original state and geometry for auditing the reset.
        initial = {
            "cube_position_world_m": cube.tolist(),
            "cube_joint_qpos": qpos.tolist(),
            "cube_half_size_m": np.asarray(self.env.cube.size).tolist(),
            "table_offset_m": np.asarray(self.env.table_offset).tolist(),
            "qpos": self.env.sim.data.qpos.tolist(),
            "qvel": self.env.sim.data.qvel.tolist(),
        }
        return observation, initial

    def step(self, observation):
        import torch

        with torch.inference_mode():
            action = np.asarray(self.policy(ob=observation))
        if action.shape != (self.action_dimension,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid policy action: {action}")
        raw, reward, done, info = self.env.step(action)
        return self._policy_observation(raw), action, bool(self.env._check_success())

    def set_cube_force(self, force_n):
        """Apply world-frame force at cube COM; no torque or position teleport."""
        force = np.asarray(force_n, dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force_n must contain three finite values")
        applied = self.env.sim.data.xfrc_applied[self.env.cube_body_id]
        applied[:] = 0.0
        applied[:3] = force

    def cube_position(self):
        return np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_id]).copy()

    def frame(self):
        # MuJoCo's render image is bottom-up. Convert to normal top-down RGB.
        frame = self.env.sim.render(height=self.height, width=self.width, camera_name=self.camera)[::-1].copy()
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Unexpected frame: {frame.shape} {frame.dtype}")
        return frame

    def close(self):
        self.env.close()
