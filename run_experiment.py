"""
End-to-end runner for the emergent communication experiment.

Usage:
    python run_experiment.py              # full run (5000 steps)
    python run_experiment.py --smoke      # quick smoke test (200 steps)
    python run_experiment.py --steps 2000
"""

import argparse
import os
import sys
import pickle

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from emergent_comm.config import ExperimentConfig, EnvConfig
from emergent_comm.environment.world import SignalingGameDataset, collate_trials
from emergent_comm.environment.entities import all_objects
from emergent_comm.agents.sender import Sender
from emergent_comm.agents.receiver import Receiver
from emergent_comm.training.trainer import EmergentCommTrainer
from emergent_comm.evaluation.compositionality import evaluate_compositionality
from emergent_comm.evaluation.vocabulary import collect_vocabulary, print_vocabulary_report
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, batch_size):
    train_ds = SignalingGameDataset(cfg.env)
    val_ds   = SignalingGameDataset(EnvConfig(seed=cfg.env.seed + 99))
    kw = dict(batch_size=batch_size, collate_fn=collate_trials, num_workers=0)
    return DataLoader(train_ds, **kw), DataLoader(val_ds, **kw)


def build_agents(cfg):
    sender = Sender(
        n_colors=cfg.env.n_colors,
        n_shapes=cfg.env.n_shapes,
        n_sizes=cfg.env.n_sizes,
        bb_cfg=cfg.backbone,
        dec_cfg=cfg.decoder,
        agent_cfg=cfg.agent,
    )
    receiver = Receiver(
        n_colors=cfg.env.n_colors,
        n_shapes=cfg.env.n_shapes,
        n_sizes=cfg.env.n_sizes,
        enc_cfg=cfg.encoder,
        bb_cfg=cfg.backbone,
    )
    return sender, receiver


def collect_all_messages(sender, cfg, device):
    """Run sender on every object in the world; return objects + messages."""
    pool = all_objects(cfg.env.n_colors, cfg.env.n_shapes, cfg.env.n_sizes)
    objects_tuples, messages_bytes = [], []
    sender.eval()
    with torch.no_grad():
        for obj in pool:
            attrs = torch.tensor([obj.to_vector()], dtype=torch.long, device=device)
            out = sender(attrs)
            msg = out["message_bytes"][0].cpu().tolist()
            eos = cfg.agent.eos_byte
            if eos in msg:
                msg = msg[: msg.index(eos)]
            objects_tuples.append(obj.to_vector())
            messages_bytes.append(msg)
    sender.train()
    return pool, objects_tuples, messages_bytes


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(history, out_dir, n_candidates):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.steps, history.train_acc, label="train")
    axes[0].plot(history.steps, history.val_acc,   label="val")
    axes[0].axhline(1 / n_candidates, color="gray", linestyle="--", label="random")
    axes[0].set_title("Communication Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.steps, history.sender_loss,   label="sender (REINFORCE)")
    axes[1].plot(history.steps, history.receiver_loss, label="receiver (NLL)")
    axes[1].set_title("Losses")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_zipf(vocab_results, out_dir):
    freqs = sorted(vocab_results["frequency_distribution"].values(), reverse=True)
    if not freqs:
        return
    ranks = list(range(1, len(freqs) + 1))
    plt.figure(figsize=(6, 4))
    plt.loglog(ranks, freqs, "o-", markersize=4)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    zipf = vocab_results.get("zipf_correlation", float("nan"))
    plt.title(f"Word Frequency Distribution (Zipf corr={zipf:.3f})" if zipf == zipf else
              "Word Frequency Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "zipf.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_disentanglement(comp_results, out_dir):
    disent = comp_results.get("positional_disentanglement", {})
    max_pos, n_attrs = 8, 3
    matrix = np.zeros((max_pos, n_attrs))
    for pos in range(max_pos):
        for attr in range(n_attrs):
            matrix[pos, attr] = disent.get(f"pos{pos}_attr{attr}", 0.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_attrs))
    ax.set_xticklabels(["color", "shape", "size"])
    ax.set_yticks(range(max_pos))
    ax.set_yticklabels([f"pos {i}" for i in range(max_pos)])
    ax.set_title("Positional Disentanglement\n(byte pos vs. attribute)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(out_dir, "disentanglement.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_topsim_summary(topsim, out_dir):
    fig, ax = plt.subplots(figsize=(4, 3))
    color = "green" if topsim > 0.3 else "orange" if topsim > 0.1 else "red"
    ax.barh(["topsim"], [topsim], color=color)
    ax.axvline(0.3, color="gray", linestyle="--", label="weak threshold")
    ax.axvline(0.6, color="black", linestyle="--", label="strong threshold")
    ax.set_xlim(0, 1)
    ax.set_title("Topographic Similarity")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "topsim.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_example_messages(pool, messages_bytes, n=20):
    print(f"\n{'Object':<35}  {'Message'}")
    print("-" * 70)
    for obj, msg in zip(pool[:n], messages_bytes[:n]):
        printable = "".join(
            chr(b) if 32 <= b < 127 else f"[{b:02x}]" for b in msg
        )
        print(f"{obj.to_label():<35}  {printable!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true", help="Quick 200-step smoke test")
    parser.add_argument("--steps",  type=int, default=5000)
    parser.add_argument("--out",    type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Config
    cfg = ExperimentConfig()
    cfg.training.checkpoint_dir = "checkpoints"

    if args.smoke:
        cfg.training.n_steps    = 200
        cfg.training.eval_every = 50
        cfg.training.batch_size = 64
        print("=== SMOKE TEST MODE (200 steps) ===")
    else:
        cfg.training.n_steps = args.steps

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device        : {device}")
    print(f"Object space  : {cfg.env.n_colors}x{cfg.env.n_shapes}x{cfg.env.n_sizes}"
          f" = {cfg.env.n_colors * cfg.env.n_shapes * cfg.env.n_sizes} objects")
    print(f"Candidates    : {cfg.env.n_distractors + 1}  "
          f"(random baseline = {1/(cfg.env.n_distractors+1):.3f})")
    print(f"Training steps: {cfg.training.n_steps}")

    # ---- Build ----
    print("\n[1/4] Building models...")
    train_loader, val_loader = build_dataloaders(cfg, cfg.training.batch_size)
    sender, receiver = build_agents(cfg)
    n_s = sum(p.numel() for p in sender.parameters())
    n_r = sum(p.numel() for p in receiver.parameters())
    print(f"  Sender params  : {n_s:,}")
    print(f"  Receiver params: {n_r:,}")

    # ---- Train ----
    print("\n[2/4] Training...")
    trainer = EmergentCommTrainer(sender, receiver, cfg, device)
    history = trainer.train(train_loader, val_loader)

    # ---- Load best checkpoint ----
    print("\n[3/4] Collecting messages from trained sender...")
    best_ckpt = os.path.join("checkpoints", "best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        sender.load_state_dict(ckpt["sender"])
        receiver.load_state_dict(ckpt["receiver"])
        print(f"  Loaded best checkpoint (step={ckpt['step']}, val_acc={ckpt['val_acc']:.3f})")

    pool, objects_tuples, messages_bytes = collect_all_messages(sender, cfg, device)
    lengths = [len(m) for m in messages_bytes]
    unique  = len(set(tuple(m) for m in messages_bytes))
    print(f"  Total objects   : {len(pool)}")
    print(f"  Unique messages : {unique}")
    print(f"  Mean msg length : {np.mean(lengths):.1f} bytes")

    # ---- Analyse ----
    print("\n[4/4] Analysis...")

    comp = evaluate_compositionality(
        objects_tuples, messages_bytes,
        n_topsim_samples=cfg.eval.n_topsim_samples,
        seed=cfg.env.seed,
    )
    print(f"\n  Topographic Similarity (topsim) : {comp['topsim']:.4f}")
    print(  "    > 0.3 = weak compositionality, > 0.6 = strong")
    for attr_name, attr_idx in [("color", 0), ("shape", 1), ("size", 2)]:
        key = f"attr{attr_idx}_max_disentanglement"
        print(f"  Max positional disentanglement [{attr_name}]: {comp[key]:.4f}")

    vocab = collect_vocabulary(objects_tuples, messages_bytes, seg_cfg=cfg.segmentation)
    print_vocabulary_report(vocab)
    print_example_messages(pool, messages_bytes, n=min(20, len(pool)))

    # ---- Save plots ----
    print(f"\nSaving plots to {args.out}/")
    plot_training_curves(history, args.out, cfg.env.n_distractors + 1)
    plot_zipf(vocab, args.out)
    plot_disentanglement(comp, args.out)
    plot_topsim_summary(comp["topsim"], args.out)

    # Save raw results
    results = {
        "history":  history,
        "topsim":   comp["topsim"],
        "vocab":    vocab,
        "comp":     comp,
        "objects":  objects_tuples,
        "messages": messages_bytes,
    }
    with open(os.path.join(args.out, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved: {args.out}/results.pkl")

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Best val accuracy : {history.val_acc[-1]:.3f}  (random = {1/(cfg.env.n_distractors+1):.3f})")
    print(f"  Topsim            : {comp['topsim']:.4f}")
    print(f"  Unique messages   : {unique} / {len(pool)}")
    print(f"  Vocab size        : {vocab['vocab_size']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
