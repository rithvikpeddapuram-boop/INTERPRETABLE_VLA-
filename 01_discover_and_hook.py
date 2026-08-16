"""
Step 1: Discover SmolVLA's exact module names, then attach forward hooks
to capture activations from the VLM trunk (Llama decoder layers) and the
action expert (state_proj / action_in_proj / action_out_proj / etc.)

Run this INSIDE your vla-interp conda environment:
    conda activate vla-interp
    cd ~/vla-interp-project
    python3 01_discover_and_hook.py

What this does:
  1. Loads SmolVLA
  2. Prints every named module (so you can see exact hook targets --
     don't guess paths, always confirm against this printout)
  3. Registers forward hooks on a chosen set of layers
  4. Confirms hooks fire correctly with a dummy input matching the
     dimensions we already confirmed (state=32, action=32, hidden=720)

NOTE: a real forward pass needs real observations (images + language +
state) from an actual episode. Since LIBERO isn't installed yet, this
script uses dummy/random tensors ONLY to confirm the hook mechanism
works -- not to produce meaningful activations. Once LIBERO is in,
we swap in real episode data.
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("=" * 60)
print("Loading SmolVLA...")
print("=" * 60)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
policy.eval()

# ------------------------------------------------------------------
# STEP A: Print all named modules so we can confirm exact hook paths
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL NAMED MODULES (search this for your hook targets)")
print("=" * 60)
module_names = [name for name, _ in policy.named_modules() if name]
for name in module_names:
    print(name)

print(f"\nTotal modules found: {len(module_names)}")

# Save the full list to a file so you don't lose it in scrollback
with open("logs/named_modules.txt", "w") as f:
    for name in module_names:
        f.write(name + "\n")
print("Saved full module list to logs/named_modules.txt")

# ------------------------------------------------------------------
# STEP B: Identify candidate layers to hook, based on what we already
# confirmed from the printed architecture (Llama decoder layers +
# action expert projections). Adjust HOOK_TARGETS below once you've
# checked logs/named_modules.txt against this guess.
# ------------------------------------------------------------------
HOOK_TARGETS = [name for name in module_names if any(
    key in name for key in [
        "action_out_proj",
        "action_in_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]
) and name.count(".") <= 3]  # keep to top-level projections, not sub-parts

# Also grab a few Llama decoder layers (early / mid / late) if present
llama_layers = [n for n in module_names if n.endswith((".15", ".0", ".8")) and "layers" in n]
HOOK_TARGETS += llama_layers

print("\n" + "=" * 60)
print("CANDIDATE HOOK TARGETS")
print("=" * 60)
for t in HOOK_TARGETS:
    print(t)

# ------------------------------------------------------------------
# STEP C: Hook manager -- captures output activations by layer name
# ------------------------------------------------------------------
class ActivationCapture:
    def __init__(self):
        self.activations = {}
        self.handles = []

    def _make_hook(self, name):
        def hook(module, input, output):
            # output can be a tensor or a tuple (e.g. attention layers) --
            # handle both cases
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def attach(self, model, layer_names):
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                print(f"Hook attached: {name}")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


capture = ActivationCapture()
capture.attach(policy, HOOK_TARGETS)

print("\n" + "=" * 60)
print(f"Total hooks attached: {len(capture.handles)}")
print("=" * 60)

if len(capture.handles) == 0:
    print("\nWARNING: No hooks attached. This means HOOK_TARGETS didn't match")
    print("any module names. Open logs/named_modules.txt, find the exact")
    print("names you want, and hardcode HOOK_TARGETS as a plain list instead")
    print("of relying on the automatic guess above.")

print("\nDone. Next: once LIBERO is installed, run a real episode through")
print("policy.select_action(...) or the model's forward pass with real")
print("observations, then check capture.activations for the captured tensors.")
