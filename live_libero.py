"""
Run SmolVLA (smolvla_libero) closed-loop in LIBERO with the same
observation / environment / policy processor flow used by lerobot-eval,
while showing the agentview camera live.

Run:
    conda activate vla
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python infer.py

Requires a display for the OpenCV live window.
"""

import os

# Use the rendering backend configured by the shell; default to EGL for
# NVIDIA hardware / headless-compatible rendering.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch

from libero.libero import benchmark

from lerobot.envs import preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


TASK_SUITE = "libero_spatial"
TASK_ID = 0
N_STEPS = TASK_SUITE_MAX_STEPS[TASK_SUITE]


# ------------------------------------------------------------------
# 1. Load SmolVLA + official policy processors
# ------------------------------------------------------------------
print("=" * 60)
print("Loading smolvla_libero + official processors...")
print("=" * 60)

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="lerobot/smolvla_libero",
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
    },
)


# ------------------------------------------------------------------
# 2. Build the LIBERO environment
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Loading LIBERO task suite: {TASK_SUITE} (task {TASK_ID})")
print("=" * 60)

task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()

env = LiberoEnv(
    task_suite=task_suite,
    task_id=TASK_ID,
    task_suite_name=TASK_SUITE,
    obs_type="pixels_agent_pos",
)

# IMPORTANT:
# LiberoEnv (the actual Gym environment) does NOT expose
# get_env_processors(). The official processor factory expects the
# EnvConfig object instead.
env_cfg = LiberoEnvConfig(
    task=TASK_SUITE,
)

env_preprocessor, env_postprocessor = make_env_pre_post_processors(
    env_cfg=env_cfg,
    policy_cfg=policy.config,
)

obs, info = env.reset()
task_description = env.task_description

print(f"Task: '{task_description}'")


# ------------------------------------------------------------------
# 3. Live camera view
# ------------------------------------------------------------------
WINDOW_NAME = "SmolVLA-LIBERO live rollout (press q to quit)"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 512, 512)


    
    
def show_frame(obs, step, reward):
    """Display the first LIBERO camera from the current observation."""
    pixels = obs.get("pixels", obs)

    if isinstance(pixels, dict):
        img = next(iter(pixels.values()))
    else:
        img = pixels

    # LIBERO image is RGB; flip vertically for display.
    img = img[::-1].copy()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(
        img_bgr,
        (512, 512),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.putText(
        img_bgr,
        f"step {step}  reward {reward:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.imshow(WINDOW_NAME, img_bgr)

    return cv2.waitKey(1) & 0xFF

def add_batch_dim(x):
    if isinstance(x, dict):
        return {k: add_batch_dim(v) for k, v in x.items()}

    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)

    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=0)

    return x
    
# ------------------------------------------------------------------
# 4. Closed-loop rollout
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Running closed-loop rollout ({N_STEPS} steps max). Press 'q' in")
print("the video window at any time to stop early.")
print("=" * 60)

reward = 0.0
show_frame(obs, 0, reward)

with torch.no_grad():
    for step in range(N_STEPS):

        # ----------------------------------------------------------
        # IMPORTANT:
        # Do NOT manually create:
        #   {"observation": obs, "task": ...}
        #
        # preprocess_observation() converts the raw LiberoEnv
        # observation into the standard LeRobot transition expected
        # by the environment and policy processors.
        # ----------------------------------------------------------
       	transition = preprocess_observation(obs)
        transition = add_batch_dim(transition)
        # Add the language instruction ex	actly after observation
        # preprocessing, matching the official evaluation flow.
        transition["task"] = [task_description]

        # LIBERO-specific observation preprocessing.
        transition = env_preprocessor(transition)

        # SmolVLA policy preprocessing.
        transition = preprocessor(transition)

        # Policy inference.
        action = policy.select_action(transition)

        # Policy action postprocessing.
        action = postprocessor(action)

        # LIBERO-specific action postprocessing.
        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        # Convert the policy output to the 1-D NumPy action expected by
        # LiberoEnv.step().
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        action_np = np.squeeze(action_np)

        # LIBERO expects exactly a 1-D action vector.
        if action_np.ndim != 1:
            raise RuntimeError(
                f"Expected 1-D action before env.step(), got "
                f"shape={action_np.shape}"
            )

        obs, reward, terminated, truncated, info = env.step(action_np)

        key = show_frame(obs, step + 1, reward)

        print(
            f"step={step + 1:03d} | "
            f"reward={reward:.1f} | "
            f"terminated={terminated} | "
            f"truncated={truncated}"
        )

        if key == ord("q"):
            print(f"Stopped by user at step {step + 1}.")
            break

        if terminated or truncated:
            print(
                f"Episode ended at step {step + 1} "
                f"(terminated={terminated}, truncated={truncated})."
            )
            break


print("\n" + "=" * 60)
print("Rollout finished. Close the window or press any key to exit.")
print("=" * 60)

cv2.waitKey(0)
cv2.destroyAllWindows()
env.close()