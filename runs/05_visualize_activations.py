"""
Stage 1b: visualize which layers / features were most activated at each
timestep, using the .pt file saved by 04_hook_rollout.py.

This is a raw-magnitude sanity check, NOT a behavioral-relevance check.
High activation magnitude can come from layer-norm scale or baseline
statistics that have nothing to do with grasp/place semantics -- it just
tells you where energy is concentrated and how it moves through the
network over the rollout. The actual behavioral-relevance question still
needs the phase-labeled probe sweep (and eventually Jacobian/SVD +
steering) discussed separately.

Caveat carried over from 04_hook_rollout.py: action-expert layer
activations currently only reflect the LAST flow-matching denoising step
per chunk (the hook overwrites on each call). VLM trunk layers are
unaffected since they only fire once per chunk-gen event.

Run:
    conda activate vla-interp
    python runs/05_visualize_activations.py

Output (all written to outputs/):
    layer_timestep_energy.png   -- layers (y) x timesteps (x) heatmap of
                                    mean |activation|
    top_dynamic_layers.txt      -- ranked list of layers by temporal
                                    variance of their energy
    feature_heatmap_<layer>.png -- for the N most temporally dynamic
                                    layers, a heatmap of their most
                                    dynamic individual features (y) over
                                    timesteps (x)
"""

import os
import re

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PT_PATH = "outputs/hidden_states_libero_spatial_task0.pt"
OUTPUT_DIR = "outputs"
N_DYNAMIC_LAYERS_TO_PLOT = 5   # how many layers get a detailed feature heatmap
TOP_K_FEATURES_PER_LAYER = 30  # how many features shown per detailed heatmap


def summarize_activation(tensor: torch.Tensor) -> np.ndarray:
    """Collapse a captured activation down to a (hidden_dim,) vector.

    Hooked tensors can be [1, hidden] (a single token / pooled projection)
    or [1, seq_len, hidden] (a sequence of tokens, e.g. VLM trunk layers
    over image + language tokens). We mean over every dim except the last
    (feature) dim, then take abs() so "activation magnitude" doesn't
    cancel out from sign.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.to(torch.float32).numpy()
    else:
        arr = np.asarray(tensor)
    arr = np.abs(arr)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr  # shape: (hidden_dim,)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main():
    if not os.path.exists(PT_PATH):
        raise FileNotFoundError(
            f"{PT_PATH} not found. Run runs/04_hook_rollout.py first."
        )

    data = torch.load(PT_PATH, map_location="cpu")
    meta = data["meta"]
    steps = data["steps"]

    if len(steps) == 0:
        raise RuntimeError("No records found in the .pt file -- nothing to visualize.")

    print(f"Loaded {len(steps)} chunk-gen records for task: "
          f"'{meta.get('task_description', '?')}'")

    layer_names = sorted(steps[0]["activations"].keys())
    print(f"Layers present: {len(layer_names)}")

    # ------------------------------------------------------------------
    # 1. Build per-layer, per-timestep feature matrices:
    #    feature_matrices[layer] -> shape (n_timesteps, hidden_dim)
    #    Not every record necessarily has every layer (a hook may not
    #    fire for a given call if that submodule wasn't reached), so we
    #    only use records where the layer is present and track the
    #    corresponding env_step/call_idx for the x-axis.
    # ------------------------------------------------------------------
    feature_matrices = {}   # layer -> (T, hidden_dim) array
    timestep_labels = {}    # layer -> list of env_step ints (x-axis ticks)

    for layer in layer_names:
        vecs = []
        xticks = []
        for rec in steps:
            if layer in rec["activations"]:
                vecs.append(summarize_activation(rec["activations"][layer]))
                xticks.append(rec["env_step"])
        if len(vecs) == 0:
            continue
        feature_matrices[layer] = np.stack(vecs, axis=0)  # (T, hidden_dim)
        timestep_labels[layer] = xticks

    # ------------------------------------------------------------------
    # 2. Layer x timestep energy heatmap (mean |activation| per layer
    #    per timestep, i.e. one scalar per (layer, timestep) cell).
    #    Layers can have different hidden_dim (VLM trunk vs action
    #    expert), so we align on timestep count, not on feature axis.
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_timesteps = len(steps)
    energy_matrix = np.full((len(layer_names), n_timesteps), np.nan)

    call_idx_all = [rec["call_idx"] for rec in steps]

    for i, layer in enumerate(layer_names):
        if layer not in feature_matrices:
            continue
        mat = feature_matrices[layer]                 # (T_layer, hidden_dim)
        energy = mat.mean(axis=1)                      # (T_layer,)
        xticks = timestep_labels[layer]
        # place each captured timestep's energy at its position in the
        # global call_idx timeline
        rec_call_idxs = [rec["call_idx"] for rec in steps if layer in rec["activations"]]
        for c_idx, e in zip(rec_call_idxs, energy):
            energy_matrix[i, c_idx] = e

    fig, ax = plt.subplots(figsize=(max(8, n_timesteps * 0.3), max(6, len(layer_names) * 0.25)))
    im = ax.imshow(energy_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=6)
    ax.set_xlabel("chunk-gen call index")
    ax.set_ylabel("layer")
    ax.set_title(f"Mean |activation| per layer over rollout\n"
                 f"task: {meta.get('task_description', '?')}")
    fig.colorbar(im, ax=ax, label="mean |activation|")
    fig.tight_layout()
    energy_path = os.path.join(OUTPUT_DIR, "layer_timestep_energy.png")
    fig.savefig(energy_path, dpi=150)
    plt.close(fig)
    print(f"Saved {energy_path}")

    # ------------------------------------------------------------------
    # 3. Rank layers by temporal variance of their energy -- the layers
    #    whose activation level changes the most over the rollout are
    #    the ones most likely to be tracking behavior/phase rather than
    #    being static/saturated.
    # ------------------------------------------------------------------
    layer_variance = []
    for i, layer in enumerate(layer_names):
        row = energy_matrix[i]
        valid = row[~np.isnan(row)]
        var = float(np.var(valid)) if len(valid) > 1 else 0.0
        layer_variance.append((layer, var))

    layer_variance.sort(key=lambda x: x[1], reverse=True)

    ranking_path = os.path.join(OUTPUT_DIR, "top_dynamic_layers.txt")
    with open(ranking_path, "w") as f:
        for layer, var in layer_variance:
            f.write(f"{var:.6f}\t{layer}\n")
    print(f"Saved layer ranking to {ranking_path}")

    # ------------------------------------------------------------------
    # 4. For the top-N most dynamic layers, plot a feature-level heatmap
    #    of their most dynamic individual features over time.
    # ------------------------------------------------------------------
    top_layers = [l for l, v in layer_variance if l in feature_matrices][:N_DYNAMIC_LAYERS_TO_PLOT]

    for layer in top_layers:
        mat = feature_matrices[layer]  # (T, hidden_dim)
        xticks = timestep_labels[layer]

        feature_var = mat.var(axis=0)  # (hidden_dim,)
        top_feature_idxs = np.argsort(feature_var)[::-1][:TOP_K_FEATURES_PER_LAYER]
        sub = mat[:, top_feature_idxs].T  # (top_k, T)

        fig, ax = plt.subplots(figsize=(max(8, len(xticks) * 0.3), max(4, TOP_K_FEATURES_PER_LAYER * 0.2)))
        im = ax.imshow(sub, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_yticks(range(len(top_feature_idxs)))
        ax.set_yticklabels([f"f{idx}" for idx in top_feature_idxs], fontsize=6)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, fontsize=6, rotation=90)
        ax.set_xlabel("env step at capture")
        ax.set_ylabel(f"top {TOP_K_FEATURES_PER_LAYER} most dynamic features")
        ax.set_title(f"Feature activation over time -- {layer}")
        fig.colorbar(im, ax=ax, label="|activation|")
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"feature_heatmap_{sanitize_filename(layer)}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("\nDone. Remember: this ranks layers/features by raw activation "
          "energy and temporal variance only -- it does not confirm any "
          "of them encode grasp/place semantics. Use this to narrow down "
          "candidates, then run the phase-labeled probe sweep before "
          "committing to layers for Jacobian/SVD analysis.")


if __name__ == "__main__":
    main()