# Interpretable VLA — SmolVLA + LIBERO Setup and Run Guide

This repository contains an interpretability workflow for **SmolVLA** on the
**LIBERO** robot-manipulation benchmark. The workflow includes:

- environment / SmolVLA sanity checks
- Jacobian (`torch.func.jacrev`) sanity checks
- SmolVLA module discovery and forward hooks
- LIBERO installation and simulator validation
- a legacy direct SmolVLA rollout script
- the newer LIBERO wrapper + official LeRobot processor rollout with a live view

---

## 0. Repository layout

Expected project layout:

```text
vla-interp-project/
├── README.md
├── install_vla_env.sh              # prerequisite; not included in this upload
├── install_libero.sh
├── smolvla_sanity_check.ipynb
├── runs/
│   ├── 00_sanity_check.py
│   ├── 01_discover_and_hook.py
│   ├── 02_test_libero.py
│   └── 03_infer_libero_task.py
├── live_libero.py
├── logs/
├── results/
└── outputs/
```

The current uploaded `runs.zip` contains the four scripts under `runs/`.

---

# 1. Prerequisites

Recommended host:

- Linux / Ubuntu
- Conda
- Git
- NVIDIA GPU + working CUDA if running SmolVLA on GPU
- A working MuJoCo installation through the Python environment
- An OpenCV GUI/display if you want the live-window script

The project files assume a Conda environment named **`vla-interp`** in the
installation and diagnostic scripts.

> **Important environment-name mismatch:** the uploaded `live_libero.py`
> currently says `conda activate vla`, while `install_libero.sh` and the
> diagnostic scripts use `vla-interp`. Pick one environment name and use it
> consistently. The instructions below use **`vla-interp`**, because that is
> what `install_libero.sh` explicitly checks.

---

# 2. Create the base SmolVLA environment

The project references a prerequisite script:

```text
install_vla_env.sh
```

This file is **not included in the current uploaded file set**, so this README
does not invent its exact installation commands.

If you already have the environment:

```bash
conda activate vla-interp
```

Verify:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import lerobot; print(lerobot.__file__)"
```

The later `install_libero.sh` script must be run inside this same environment.

---

# 3. First sanity check: Jupyter notebook

Before installing LIBERO, run the SmolVLA sanity notebook.

The notebook itself instructs:

```bash
conda activate vla-interp
cd ~/vla-interp-project
jupyter lab
```

Open:

```text
smolvla_sanity_check.ipynb
```

Run all cells in order.

## What it checks

### Cell 1 — PyTorch

Checks:

- PyTorch version
- CUDA availability

### Cell 2 — SmolVLA loading

Loads:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
```

Save the printed architecture because the later interpretability work needs the
exact module names and dimensions.

### Cell 3 — Jacobian sanity check

Runs:

```python
from torch.func import jacrev
```

and verifies that Jacobian computation works.

Expected output is a Jacobian with shape similar to:

```text
torch.Size([4, 4, 8])
```

---

# 4. Run `00_sanity_check.py`

After the notebook succeeds:

```bash
conda activate vla-interp
cd ~/vla-interp-project

python runs/00_sanity_check.py
```

This repeats the basic PyTorch / SmolVLA / Jacobian checks from the notebook
in a normal Python script.

Expected checks:

```text
torch version: ...
CUDA available: ...
Jacobian shape: ...
```

The script also loads:

```text
lerobot/smolvla_libero
```

and prints the policy architecture.

---

# 5. Run `01_discover_and_hook.py`

Next:

```bash
python runs/01_discover_and_hook.py
```

This script does **not** require LIBERO.

It:

1. Loads `lerobot/smolvla_libero`.
2. Prints every named module.
3. Automatically identifies candidate action-expert projection layers.
4. Looks for selected Llama decoder layers.
5. Attaches forward hooks.
6. Saves all module names to:

```text
logs/named_modules.txt
```

The script is specifically intended to establish the **real module paths before
doing activation/Jacobian analysis**.

Do not guess hook names manually if the architecture printout gives a different
path.

---

# 6. Install LIBERO

Only after the base SmolVLA environment is working:

```bash
conda activate vla-interp
cd ~/vla-interp-project

bash install_libero.sh
```

The installation script:

1. Requires the `vla-interp` Conda environment.
2. Clones LIBERO into:

```text
~/LIBERO
```

3. Installs a modern NumPy first:

```text
numpy>=1.26
```

4. Installs LIBERO requirements while deliberately excluding:

```text
torch
torchvision
torchaudio
numpy
```

because the SmolVLA environment already provides the newer versions.
5. Installs LIBERO with:

```bash
pip install -e .
```

6. Downloads only the `libero_spatial` task suite initially.

The script explicitly says it is intended to run **after the base VLA
environment installation has succeeded**. fileciteturn1file0L3-L10

---

# 7. Test the LIBERO simulator

After `install_libero.sh` completes:

```bash
conda activate vla-interp
cd ~/vla-interp-project

python runs/02_test_libero.py
```

This test uses:

```text
OffScreenRenderEnv
```

and therefore does not require a GUI window.

It:

1. Loads `libero_spatial`.
2. Prints the number of tasks.
3. Loads task `0`.
4. Prints the task language instruction.
5. Loads the BDDL file.
6. Creates the simulation.
7. Loads the first initial state.
8. Executes 10 zero actions.

You want to see:

```text
SUCCESS -- LIBERO environment is fully working.
```

Do **not** proceed to policy inference if this test fails.

---

# 8. Legacy direct inference script

There is an older script:

```text
runs/03_infer_libero_task.py
```

Run it with:

```bash
conda activate vla-interp
cd ~/vla-interp-project

export MUJOCO_GL=osmesa
python runs/03_infer_libero_task.py
```

This script directly constructs a LIBERO `OffScreenRenderEnv` and manually builds
the SmolVLA input batch.

It was useful for early debugging, but its own comments describe the camera
mapping and state construction as **best-effort guesses**.

Therefore:

> **Use this script primarily as a diagnostic / historical baseline. It is
> not the preferred final inference path.**

---

# 9. Recommended inference: `live_libero.py`

The newer script is:

```text
live_libero.py
```

It uses the LeRobot LIBERO wrapper and the official processor chain.

The intended pipeline is:

```text
Raw LIBERO observation
        │
        ▼
preprocess_observation()
        │
        ▼
batch dimension
        │
        ▼
task instruction
        │
        ▼
LIBERO env_preprocessor
        │
        ▼
SmolVLA policy preprocessor
        │
        ▼
SmolVLA select_action()
        │
        ▼
policy postprocessor
        │
        ▼
LIBERO env_postprocessor
        │
        ▼
LIBERO env.step()
```

The script creates the LIBERO environment configuration and then obtains the
environment pre/post processors using:

```python
make_env_pre_post_processors(
    env_cfg=env_cfg,
    policy_cfg=policy.config,
)
```

This is the current intended processor-based path. fileciteturn1file1L76-L89

---

## 9.1 Fix the environment name first

If your environment is actually named `vla-interp`:

```bash
conda activate vla-interp
```

The current script header says `vla`; that is inconsistent with
`install_libero.sh`.

---

## 9.2 Set the MuJoCo rendering backend

For the live-window version:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Then:

```bash
python live_libero.py
```

The script loads:

```text
lerobot/smolvla_libero
```

and uses:

```python
preprocess_observation
make_env_pre_post_processors
make_pre_post_processors
```

before calling the policy. fileciteturn1file1L28-L55

---

# 10. Important: batch dimension

The standalone `LiberoEnv` returns one observation at a time, whereas the
processor expects batched state tensors.

The current `live_libero.py` therefore contains:

```python
def add_batch_dim(x):
    if isinstance(x, dict):
        return {k: add_batch_dim(v) for k, v in x.items()}

    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)

    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=0)

    return x
```

This is important because, for example:

```text
EEF quaternion: (4,)
```

must become:

```text
EEF quaternion: (1, 4)
```

before the environment processor runs.

The uploaded script already contains this helper. fileciteturn1file1L139-L149

---

# 11. Current `live_libero.py` indentation warning

The uploaded `live_libero.py` currently contains a tab before:

```python
transition = preprocess_observation(obs)
```

while surrounding lines use spaces.

If Python reports:

```text
TabError: inconsistent use of tabs and spaces in indentation
```

run:

```bash
sed -i 's/\t/    /g' live_libero.py
```

Then:

```bash
python live_libero.py
```

---

# 12. What the live rollout should print

At startup:

```text
Loading smolvla_libero + official processors...
```

Then something like:

```text
Loading LIBERO task suite: libero_spatial (task 0)
```

Then:

```text
Task: 'pick up the black bowl ...'
```

Then:

```text
Running closed-loop rollout (280 steps max)...
```

During the rollout:

```text
step=001 | reward=0.0 | terminated=False | truncated=False
step=002 | reward=0.0 | terminated=False | truncated=False
...
```

The OpenCV window should show the LIBERO agentview.

Press:

```text
q
```

inside the OpenCV window to stop early.

---

# 13. Complete recommended execution order

If starting from a fresh machine/project:

```bash
# --------------------------------------------------
# 1. Base VLA environment
# --------------------------------------------------
bash install_vla_env.sh

# --------------------------------------------------
# 2. Activate it
# --------------------------------------------------
conda activate vla-interp

# --------------------------------------------------
# 3. SmolVLA sanity notebook
# --------------------------------------------------
cd ~/vla-interp-project
jupyter lab
# Run smolvla_sanity_check.ipynb completely

# --------------------------------------------------
# 4. Script sanity check
# --------------------------------------------------
python runs/00_sanity_check.py

# --------------------------------------------------
# 5. Discover modules + attach hooks
# --------------------------------------------------
python runs/01_discover_and_hook.py

# --------------------------------------------------
# 6. Install LIBERO
# --------------------------------------------------
bash install_libero.sh

# --------------------------------------------------
# 7. Verify LIBERO
# --------------------------------------------------
export MUJOCO_GL=egl
python runs/02_test_libero.py

# --------------------------------------------------
# 8. Optional legacy inference
# --------------------------------------------------
export MUJOCO_GL=osmesa
python runs/03_infer_libero_task.py

# --------------------------------------------------
# 9. Recommended current inference
# --------------------------------------------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python live_libero.py
```

---

# 14. Recommended workflow for the actual research

Once the above pipeline works:

```text
                 ┌──────────────────────────┐
                 │ SmolVLA sanity check     │
                 │ notebook / 00_sanity     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Discover exact modules   │
                 │ 01_discover_and_hook     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ LIBERO simulator test    │
                 │ 02_test_libero           │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ SmolVLA closed-loop      │
                 │ live_libero.py           │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Activation collection    │
                 │ + Jacobian analysis      │
                 └──────────────────────────┘
```

The key research-facing point is that `01_discover_and_hook.py` should be used
to establish exact module names before collecting real rollout activations.

---

# 15. Useful environment checks

If something fails, run:

```bash
conda activate vla-interp

python --version

python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

python -c "import lerobot; print('LeRobot:', lerobot.__file__)"

python -c "import libero; print('LIBERO import: OK')"

python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"

python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

For SmolVLA:

```bash
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('SmolVLA import: OK')"
```

For the processor path:

```bash
python -c "from lerobot.envs import preprocess_observation; print('preprocess_observation: OK')"
```

---

# 16. Common errors

## `num2words is required`

Install:

```bash
pip install num2words
```

This is required by the SmolVLM processor.

## `ObservationProcessorStep requires an observation`

Do not manually pass the raw LIBERO observation as:

```python
{"observation": obs, "task": ...}
```

Use the standardized preprocessing flow in `live_libero.py`:

```python
transition = preprocess_observation(obs)
transition = add_batch_dim(transition)
transition["task"] = [task_description]
transition = env_preprocessor(transition)
transition = preprocessor(transition)
```

## `_quat2axisangle expected shape (B, 4), got (4,)`

The standalone environment produced an unbatched quaternion.

Make sure `add_batch_dim()` handles both:

```python
torch.Tensor
```

and:

```python
np.ndarray
```

as shown above.

## `TabError`

Convert tabs to spaces:

```bash
sed -i 's/\t/    /g' live_libero.py
```

## OpenGL / `glGetError` failure

Check:

```bash
echo $MUJOCO_GL
echo $PYOPENGL_PLATFORM
```

For NVIDIA/EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

For a headless OSMesa setup:

```bash
export MUJOCO_GL=osmesa
```

Use the backend appropriate to the machine.

## `QFontDatabase: Cannot find font directory`

This is an OpenCV/Qt font warning. If the actual OpenCV window opens and works,
it is not the main SmolVLA/LIBERO inference failure.

---

# 17. Output locations

The project `.gitignore` excludes:

```text
logs/
outputs/
results/
```

as well as videos and model/checkpoint files.

The hook-discovery script writes:

```text
logs/named_modules.txt
```

The legacy inference script can write:

```text
results/rollout.gif
```

The current live-view script displays frames directly rather than being a
training script.

---

# 18. Important distinction: evaluation vs interpretability

The main pipeline has two separate purposes:

### Policy validation

```text
LIBERO → SmolVLA → action → LIBERO
```

Use:

```text
02_test_libero.py
live_libero.py
```

to establish that the environment and policy work.

### Interpretability

```text
SmolVLA
   ├── VLM trunk activations
   ├── action expert activations
   └── Jacobian / sensitivity analysis
```

Use:

```text
01_discover_and_hook.py
```

as the starting point for this stage.

Do not begin the Jacobian interpretation experiments until the closed-loop
SmolVLA rollout has been verified.

---

## Short version

If everything is already installed, the practical sequence is:

```bash
conda activate vla-interp
cd ~/vla-interp-project

python runs/00_sanity_check.py
python runs/01_discover_and_hook.py

bash install_libero.sh

export MUJOCO_GL=egl
python runs/02_test_libero.py

# Optional legacy baseline:
export MUJOCO_GL=osmesa
python runs/03_infer_libero_task.py

# Recommended current rollout:
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
sed -i 's/\t/    /g' live_libero.py
python live_libero.py
```
