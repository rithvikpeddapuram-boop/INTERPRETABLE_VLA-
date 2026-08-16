"""
Step 2: Confirm LIBERO actually works -- loads a task, resets the
simulated environment, and steps through a few dummy actions.

Run AFTER install_libero.sh completes:
    conda activate vla-interp
    cd ~/vla-interp-project
    python3 02_test_libero.py

This uses OffScreenRenderEnv (headless -- no display window needed,
correct choice since you're running inside WSL2 without a GUI).
"""

import os
# Headless rendering -- required since there's no display in WSL2 terminal
os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

print("=" * 60)
print("Loading LIBERO benchmark suite: libero_spatial")
print("=" * 60)
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()

print(f"Number of tasks in this suite: {task_suite.n_tasks}")

task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
print(f"Task {task_id} instruction: '{task.language}'")
print(f"BDDL file: {task_bddl_file}")

print("\n" + "=" * 60)
print("Creating simulation environment (headless)...")
print("=" * 60)
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

print("\n" + "=" * 60)
print("Stepping through 10 dummy actions...")
print("=" * 60)
dummy_action = [0.0] * 7  # 7-dim: matches standard robosuite action space
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
    print(f"Step {step}: reward={reward}, done={done}")

print("\n" + "=" * 60)
print("SUCCESS -- LIBERO environment is fully working.")
print("=" * 60)
print(f"Observation keys available: {list(obs.keys())}")

env.close()
