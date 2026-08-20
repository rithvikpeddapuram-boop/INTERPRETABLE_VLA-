"""
Stage 1 (broad capture): run SmolVLA (smolvla_libero) closed-loop in LIBERO,
same processor pipeline as live_libero.py, but with forward hooks attached
to EVERY VLM decoder layer plus the action-expert projection points. Saves
raw per-step hidden states to disk so a later, cheap layer-wise probe sweep
can decide which few layers are worth the expensive Jacobian/SVD analysis.

This script does NOT compute the Jacobian. That's a separate, targeted
script that should only run on the small set of layers this sweep points
to -- computing jacrev over every layer for every step is the expensive
step we're deliberately avoiding here.

Run:
    conda activate vla-interp
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    python runs/04_hook_rollout.py

Output:
    outputs/hidden_states_<task_suite>_task<task_id>.pt
        A dict:
          "meta": {task_suite, task_id, task_description, hidden_dim, ...}
          "steps": list of per-forward-call records, each:
              {
                "call_idx": int,          # index of this policy forward call
                "env_step": int,          # env.step() count at capture time
                "reward": float,
                "terminated": bool,
                "truncated": bool,
                "activations": {layer_name: Tensor [seq or 1, hidden]},
              }

NOTE on capture granularity:
    select_action() runs the flow-matching action-expert internally to
    produce a whole action CHUNK. The policy's forward pass (and these
    hooks) therefore fire once per CHUNK, not once per env.step(). Each
    record's "env_step" tells you which env step was current when that
    forward call happened -- use that to align activations with the
    rollout timeline, don't assume one record per env step.
"""

import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import re

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
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, f"hidden_states_{TASK_SUITE}_task{TASK_ID}.pt"
)


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
# 2. Discover hook targets -- EVERY VLM decoder layer + every
#    action-expert projection point. Broadened from
#    01_discover_and_hook.py's sampled (0/8/15) version.
# ------------------------------------------------------------------
module_names = [name for name, _ in policy.named_modules() if name]

os.makedirs("logs", exist_ok=True)
with open("logs/named_modules.txt", "w") as f:
    for name in module_names:
        f.write(name + "\n")

# Any module whose path ends in ".layers.<int>" -- this covers every
# decoder layer in both the VLM trunk and the action expert, whatever
# their exact attribute names turn out to be. Confirm against
# logs/named_modules.txt if this looks wrong for your checkpoint.
#
# IMPORTANT: for this checkpoint, hooking the bare ".layers.N" block
# only fires for the vision encoder. text_model and lm_expert are
# orchestrated by a custom joint-attention forward in vlm_with_expert
# that calls each layer's .self_attn / .mlp submodules directly rather
# than invoking the block's own forward() -- so a hook on ".layers.N"
# itself never fires for those two stacks (confirmed via the
# never-fired report). All three stacks do consistently expose a
# ".mlp" submodule at ".layers.N.mlp", so we hook that -- it fires for
# all three and gives a consistent per-layer capture point.
#
# We also hook the attention-OUTPUT projection at each layer
# (self_attn.o_proj for text_model/lm_expert, self_attn.out_proj for
# the vision encoder -- naming differs between the two). This matters
# specifically for lm_expert: per the architecture diagram, the expert
# does NOT self-attend over its own tokens -- it CROSS-attends into the
# VLM's prefix KV cache at every layer. The attention-out projection is
# therefore the exact point where cross-modal (image+instruction)
# content gets absorbed into the action stream, upstream of the
# expert's own (likely more embodiment-specific) MLP -- a stronger
# candidate location for a shared/intent-level representation than the
# MLP output alone.
mlp_pattern = re.compile(r"\.layers\.\d+\.mlp$")
attn_out_pattern = re.compile(r"\.layers\.\d+\.self_attn\.(o_proj|out_proj)$")
decoder_layers = [
    n for n in module_names
    if mlp_pattern.search(n) or attn_out_pattern.search(n)
]

# Action-expert projection points (same candidates 01_discover_and_hook.py
# identified) -- these sit right next to the actual action output, so
# they're cheap insurance even though they're not full decoder layers.
projection_keys = [
    "action_out_proj",
    "action_in_proj",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
]
projection_layers = [
    n for n in module_names
    if any(k in n for k in projection_keys) and n.count(".") <= 3
]

HOOK_TARGETS = sorted(set(decoder_layers) | set(projection_layers))

print(f"\nFound {len(decoder_layers)} decoder layers and "
      f"{len(projection_layers)} projection modules.")
print(f"Total hook targets: {len(HOOK_TARGETS)}")

if len(HOOK_TARGETS) == 0:
    raise RuntimeError(
        "No hook targets matched. Open logs/named_modules.txt, find the "
        "real layer/projection names for this checkpoint, and hardcode "
        "HOOK_TARGETS as a plain list instead of relying on the pattern "
        "match above."
    )


# ------------------------------------------------------------------
# 3. Hook manager -- captures output activations by layer name on
#    every forward call, tagged with a call index.
# ------------------------------------------------------------------
class ActivationCapture:
    def __init__(self):
        self.activations = {}
        self.handles = []
        self.attached_names = set()
        self.ever_fired = set()
        self._warned_types = set()  # avoid spamming the same warning every call

    def _make_hook(self, name):
        def hook(module, inputs, output):
            out = None
            if isinstance(output, torch.Tensor):
                out = output
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                out = output[0]
            elif hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
                # Handles HF-style ModelOutput dataclasses, which are
                # dict-like, not plain tuples -- isinstance(output, tuple)
                # is False for these even though they wrap a tensor.
                out = output.last_hidden_state
            elif hasattr(output, "hidden_states") and isinstance(output.hidden_states, torch.Tensor):
                out = output.hidden_states

            if out is not None:
                self.activations[name] = out.detach().to("cpu")
                self.ever_fired.add(name)
            elif name not in self._warned_types:
                print(f"[hook warning] {name}: forward fired but output type "
                      f"{type(output)} wasn't recognized -- no tensor captured. "
                      f"Inspect this module's actual return type and extend "
                      f"the hook if this layer matters.")
                self._warned_types.add(name)
        return hook

    def attach(self, model, layer_names):
        layer_set = set(layer_names)
        for name, module in model.named_modules():
            if name in layer_set:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                self.attached_names.add(name)

    def report_never_fired(self):
        never_fired = self.attached_names - self.ever_fired
        print(f"\nHooks attached: {len(self.attached_names)} | "
              f"fired at least once: {len(self.ever_fired)} | "
              f"never fired: {len(never_fired)}")
        if never_fired:
            print("These modules were hooked but never produced a captured "
                  "activation (either never called during select_action(), "
                  "or their output type isn't handled above):")
            for n in sorted(never_fired):
                print(f"  - {n}")

    def snapshot(self):
        """Return a copy of the current activations and clear the buffer."""
        snap = {k: v.clone() for k, v in self.activations.items()}
        self.activations = {}
        return snap

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


capture = ActivationCapture()
capture.attach(policy, HOOK_TARGETS)
print(f"Hooks attached: {len(capture.handles)}")


# ------------------------------------------------------------------
# 4. Build the LIBERO environment (same as live_libero.py)
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

env_cfg = LiberoEnvConfig(task=TASK_SUITE)

env_preprocessor, env_postprocessor = make_env_pre_post_processors(
    env_cfg=env_cfg,
    policy_cfg=policy.config,
)

obs, info = env.reset()
task_description = env.task_description
print(f"Task: '{task_description}'")


def add_batch_dim(x):
    if isinstance(x, dict):
        return {k: add_batch_dim(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=0)
    return x


# ------------------------------------------------------------------
# 5. Closed-loop rollout with activation capture (no live window --
#    this run is for data collection, not visualization)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Running closed-loop rollout ({N_STEPS} steps max) with hooks...")
print("=" * 60)

records = []
call_idx = 0
reward = 0.0

with torch.no_grad():
    for step in range(N_STEPS):
        transition = preprocess_observation(obs)
        transition = add_batch_dim(transition)
        transition["task"] = [task_description]

        transition = env_preprocessor(transition)
        transition = preprocessor(transition)

        # select_action() internally runs the flow-matching action expert
        # to produce a chunk. Hooks fire during this call whenever the
        # policy actually does a forward pass (not on every env step --
        # cached chunk steps won't refire the hooks).
        action = policy.select_action(transition)

        if capture.activations:
            snap = capture.snapshot()
            records.append({
                "call_idx": call_idx,
                "env_step": step,
                "reward": reward,
                "terminated": False,
                "truncated": False,
                "activations": snap,
            })
            call_idx += 1

        action = postprocessor(action)

        action_transition = {"action": action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]

        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        action_np = np.squeeze(action_np)

        if action_np.ndim != 1:
            raise RuntimeError(
                f"Expected 1-D action before env.step(), got shape={action_np.shape}"
            )

        obs, reward, terminated, truncated, info = env.step(action_np)

        if records and records[-1]["call_idx"] == call_idx - 1:
            records[-1]["terminated"] = bool(terminated)
            records[-1]["truncated"] = bool(truncated)

        print(f"step={step + 1:03d} | reward={reward:.1f} | "
              f"terminated={terminated} | truncated={truncated} | "
              f"forward_calls_so_far={call_idx}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1} "
                  f"(terminated={terminated}, truncated={truncated}).")
            break

capture.report_never_fired()
capture.remove()

# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

meta = {
    "task_suite": TASK_SUITE,
    "task_id": TASK_ID,
    "task_description": task_description,
    "hook_targets": HOOK_TARGETS,
    "n_forward_calls": call_idx,
}

torch.save({"meta": meta, "steps": records}, OUTPUT_PATH)

print("\n" + "=" * 60)
print(f"Saved {call_idx} forward-call activation records to {OUTPUT_PATH}")
print(f"Hooked layers per record: {len(HOOK_TARGETS)}")
print("=" * 60)

env.close()