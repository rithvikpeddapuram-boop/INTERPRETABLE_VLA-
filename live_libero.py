"""
Run SmolVLA (smolvla_libero) closed-loop in LIBERO, using the OFFICIAL
preprocessor/postprocessor pipeline (same one lerobot-eval uses), with a
LIVE window showing the agentview camera feed as it runs.

Run:
    conda activate vla-interp
    export MUJOCO_GL=osmesa
    cd ~/vla-interp-project
    python3 05_infer_libero_live_view.py

Requires a display. On WSL2 with WSLg (Windows 11 default), this should
just work. If cv2.imshow errors with a Qt/display message, tell me the
exact error -- we'll switch to a disk-based fallback viewer instead.

NOTE: this uses the exact same preprocessing path as `lerobot-eval`, which
we've been debugging (camera key renaming). If this crashes with a
'camera1/camera2/camera3 missing' error, that confirms it's a pipeline
issue, not something specific to our script -- run 04_debug_preprocessor.py
to pin down exactly where the rename step is failing.
"""

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

import cv2
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS
from libero.libero import benchmark

TASK_SUITE = "libero_spatial"
TASK_ID = 0
N_STEPS = TASK_SUITE_MAX_STEPS[TASK_SUITE]  # 280 for libero_spatial

# ------------------------------------------------------------------
# 1. Load policy + official pre/post processors for this checkpoint
# ------------------------------------------------------------------
print("=" * 60)
print("Loading smolvla_libero + official processors...")
print("=" * 60)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path="lerobot/smolvla_libero",
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)

# ------------------------------------------------------------------
# 2. Build the LIBERO env (official lerobot wrapper)
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

obs, info = env.reset()
task_description = env.task_description
print(f"Task: '{task_description}'")

# ------------------------------------------------------------------
# 3. Set up the live view window
# ------------------------------------------------------------------
WINDOW_NAME = "SmolVLA-LIBERO live rollout (press q to quit)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 512, 512)

def show_frame(obs, step, reward):
    """Grab the agentview image out of the obs dict and display it."""
    pixels = obs.get("pixels", obs)  # obs["pixels"] is a dict of camera_name -> image
    if isinstance(pixels, dict):
        img = next(iter(pixels.values()))
    else:
        img = pixels
    # LIBERO images come out vertically flipped and in RGB -- fix both for display
    img = img[::-1].copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img_bgr, f"step {step}  reward {reward:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, img_bgr)
    # waitKey(1) lets the window actually refresh; returns key pressed (if any)
    key = cv2.waitKey(1) & 0xFF
    return key

# ------------------------------------------------------------------
# 4. Closed-loop rollout with live display
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Running closed-loop rollout ({N_STEPS} steps max). Press 'q' in the")
print("video window at any time to stop early.")
print("=" * 60)

reward = 0.0
show_frame(obs, 0, reward)

with torch.no_grad():
    for step in range(N_STEPS):
        batch = dict(obs)
        batch["task"] = [task_description]

        processed = preprocessor(batch)
        action = policy.select_action(processed)
        action = postprocessor(action)
        action_np = action.squeeze(0).cpu().numpy() if hasattr(action, "cpu") else np.array(action)

        obs, reward, terminated, truncated, info = env.step(action_np)

        key = show_frame(obs, step, reward)
        if key == ord("q"):
            print(f"Stopped by user at step {step}.")
            break

        if terminated or truncated:
            print(f"Episode ended at step {step} (terminated={terminated}, truncated={truncated}).")
            break

print("\n" + "=" * 60)
print("Rollout finished. Close the window or press any key to exit.")
print("=" * 60)
cv2.waitKey(0)
cv2.destroyAllWindows()
env.close()
