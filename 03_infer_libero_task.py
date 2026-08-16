"""
Step 3: Run SmolVLA (smolvla_libero checkpoint) in closed-loop control
inside a real LIBERO environment, on a single task.

Run AFTER 02_test_libero.py has succeeded (confirms LIBERO works) and
AFTER you've verified smolvla_libero loads cleanly (00_sanity_check.py
or the notebook).

    conda activate vla-interp
    cd ~/vla-interp-project
    export MUJOCO_GL=osmesa
    python3 03_infer_libero_task.py

NOTE ON OBSERVATION FORMAT:
SmolVLA expects a specific dict of keys (image tensors, robot state,
language instruction) that must match how it was trained. The keys
below are my best-effort guess based on common LeRobot/SmolVLA
conventions (observation.images.<camera>, observation.state, task).
If you get a KeyError or shape mismatch when calling select_action,
paste the error -- we'll inspect policy.config.input_features (printed
below) to get the exact expected keys/shapes and fix this in one pass.
"""

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

import numpy as np
import torch

N_STEPS = 150  # increased from 20 -- LIBERO pick/place tasks typically need 100-300+ steps

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ------------------------------------------------------------------
# 1. Load the policy
# ------------------------------------------------------------------
print("=" * 60)
print("Loading smolvla_libero...")
print("=" * 60)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
policy.eval()

print("\nPolicy input_features (expected observation keys/shapes):")
try:
    for k, v in policy.config.input_features.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"  (couldn't introspect input_features automatically: {e})")

print("\nPolicy output_features (expected action keys/shapes):")
try:
    for k, v in policy.config.output_features.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"  (couldn't introspect output_features automatically: {e})")

# ------------------------------------------------------------------
# 2. Load LIBERO task + env
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Loading LIBERO benchmark suite: libero_spatial")
print("=" * 60)
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()

task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
task_description = task.language
print(f"Task instruction: '{task_description}'")

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
    # Policy expects 3 cameras (camera1/camera2/camera3). LIBERO's default env
    # only renders agentview + wrist. Requesting a 3rd view (frontview) here --
    # this mapping (which physical camera -> camera1 vs camera2 vs camera3) is
    # an educated guess, not confirmed against the smolvla_libero training
    # config. If rollouts behave oddly, this is the first thing to revisit.
    "camera_names": ["agentview", "robot0_eye_in_hand", "frontview"],
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

# Step a couple of times with a no-op action so the sim settles
# (matches common LIBERO eval convention)
for _ in range(5):
    obs, _, _, _ = env.step([0.0] * 7)

# ------------------------------------------------------------------
# 3. Build the observation batch for the policy
# ------------------------------------------------------------------
def quat_to_axis_angle(quat_xyzw):
    """Convert a quaternion (x,y,z,w) to a 3D axis-angle (rotation) vector,
    without needing scipy. Standard formula: angle = 2*acos(w),
    axis = (x,y,z) / sin(angle/2)."""
    x, y, z, w = quat_xyzw
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 1e-8))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z]) / sin_half
    return (axis * angle).astype(np.float32)


def build_batch(obs, task_description, device="cpu"):
    """
    Convert a raw LIBERO obs dict into the batch format SmolVLA (smolvla_libero)
    expects, per policy.config.input_features:
      observation.state: (6,)
      observation.images.camera1/camera2/camera3: (3, 256, 256) each

    Camera-name mapping (agentview->camera1, wrist->camera2, frontview->camera3)
    and the state composition (eef_pos[3] + eef axis-angle[3]) are best-effort
    guesses -- not confirmed against the exact training config. If behavior
    looks wrong once running, this function is the first place to revisit.
    """
    def to_chw_tensor(img):
        # LIBERO images are vertically flipped relative to standard convention
        img = img[::-1].copy()
        t = torch.from_numpy(img).float() / 255.0
        t = t.permute(2, 0, 1)  # HWC -> CHW
        return t.unsqueeze(0).to(device)  # add batch dim

    state = np.concatenate([
        obs["robot0_eef_pos"],                      # 3
        quat_to_axis_angle(obs["robot0_eef_quat"]),  # 3
    ]).astype(np.float32)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)

    batch = {
        "observation.images.camera1": to_chw_tensor(obs["agentview_image"]),
        "observation.images.camera2": to_chw_tensor(obs["robot0_eye_in_hand_image"]),
        "observation.images.camera3": to_chw_tensor(obs["frontview_image"]),
        "observation.state": state_t,
        "task": [task_description],
    }
    return batch


# ------------------------------------------------------------------
# 4. Closed-loop rollout
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Running closed-loop inference ({N_STEPS} steps)...")
print("=" * 60)

policy.reset()  # clears any internal action-chunk buffer, if applicable
frames = []

with torch.no_grad():
    for step in range(N_STEPS):
        batch = build_batch(obs, task_description)
        action = policy.select_action(batch)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, done, info = env.step(action_np.tolist())
        print(f"Step {step}: action={np.round(action_np, 3)} reward={reward} done={done}")

        # Save a frame every 10 steps so you can visually inspect the rollout
        if step % 10 == 0:
            frame = obs["agentview_image"][::-1]
            frames.append(frame)

        if done:
            print("Task marked done by environment.")
            break

print("\n" + "=" * 60)
print("SUCCESS -- ran SmolVLA (smolvla_libero) closed-loop in LIBERO.")
print("=" * 60)

# Save the collected frames as a GIF so you can visually check what the
# policy actually did, instead of just reading action numbers.
if frames:
    try:
        import imageio
        out_path = os.path.join("results", "rollout.gif")
        os.makedirs("results", exist_ok=True)
        imageio.mimsave(out_path, frames, duration=0.2)
        print(f"Saved rollout preview ({len(frames)} frames) to {out_path}")
    except ImportError:
        print("imageio not installed -- skipping GIF save. "
              "Run 'pip install imageio' to enable this next time.")

env.close()