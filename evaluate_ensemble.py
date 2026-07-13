#!/usr/bin/env python
"""Evaluate an ensemble of trained Saluki models on the held-out test fold.

Each ensemble member is a model trained with the *same* ``test_fold`` held out
(and a different ``valid_fold`` / ``seed``). Because that fold was never seen in
training or validation by any member, averaging their predictions on it gives an
honest estimate of the ensemble's generalization.

The script:
  1. builds the test-fold dataloader for each requested species,
  2. runs every checkpoint over it (predictions align because shuffle=False),
  3. averages predictions across members, and
  4. reports per-member and ensemble Pearson R, R^2 (coeff. of determination),
     and Spearman rho -- so the ensembling gain is visible.

Example:
    python evaluate_ensemble.py \
        --params outputs/run_valid1/params_<id>.json \
        --checkpoints outputs/run_valid*/model_best_*.pt \
        --test_fold 0 --species human mouse \
        --out ensemble_test_predictions.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from data import N_FOLDS, SalukiTsvDataset, build_split_to_folds
from model import SalukiModel


def r2_coeff_determination(preds: np.ndarray, targets: np.ndarray) -> float:
    """R^2 = 1 - SS_res/SS_tot, matching train.py's ``r2_score``."""
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "pearson_r": float(pearsonr(preds, targets)[0]),
        "r2": r2_coeff_determination(preds, targets),
        "spearman_r": float(spearmanr(preds, targets)[0]),
        "n": int(len(preds)),
    }


def build_test_loader(
    tsv_path: str,
    seq_length: int,
    test_fold: int,
    batch_size: int,
) -> DataLoader:
    """Test-fold loader for one species. ``valid_fold`` is irrelevant here (we only
    read the 'test' split) but must differ from ``test_fold``, so pick any other."""
    valid_fold = (test_fold + 1) % N_FOLDS
    split_to_folds = build_split_to_folds(test_fold, valid_fold)
    dataset = SalukiTsvDataset(
        tsv_path,
        split="test",
        seq_length=seq_length,
        split_to_folds=split_to_folds,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader, dataset


def load_state_dict(path: str, device: str) -> dict:
    """Accept either a bare state_dict (model_best_*.pt) or a full training
    checkpoint (checkpoint_*.pt, which nests it under 'model_state_dict')."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


@torch.no_grad()
def predict(
    model: SalukiModel,
    loader: DataLoader,
    species_index: int,
    device: str,
) -> np.ndarray:
    """Predictions over the loader in dataset order (shuffle=False)."""
    model.eval()
    preds: List[torch.Tensor] = []
    for x, _y in loader:
        x = x.to(device, dtype=torch.float32)
        out = model(x, species_index=species_index)
        preds.append(out.cpu().view(-1))
    return torch.cat(preds).numpy()


def evaluate_species(
    species: str,
    species_index: int,
    tsv_path: str,
    checkpoints: Sequence[str],
    params_model: dict,
    test_fold: int,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader, dataset = build_test_loader(
        tsv_path, params_model["seq_length"], test_fold, batch_size
    )
    genes = dataset.dataframe["gene"].to_numpy()
    targets = dataset.dataframe["half_life"].to_numpy(dtype=np.float64)

    print(f"\n=== species={species!r} (head {species_index}) "
          f"| test fold {test_fold} | {len(targets)} genes "
          f"| {len(checkpoints)} models ===")

    per_model_preds: List[np.ndarray] = []
    for ckpt in checkpoints:
        model = SalukiModel(**params_model).to(device)
        model.load_state_dict(load_state_dict(ckpt, device))
        p = predict(model, loader, species_index, device)
        per_model_preds.append(p)
        m = metrics(p, targets)
        print(f"  [member] {ckpt}\n"
              f"           pearson_r={m['pearson_r']:.4f}  "
              f"r2={m['r2']:.4f}  spearman={m['spearman_r']:.4f}")

    preds_stack = np.vstack(per_model_preds)          # (n_models, n_genes)
    ensemble_pred = preds_stack.mean(axis=0)          # average predictions, not metrics
    em = metrics(ensemble_pred, targets)
    mean_member = {
        k: float(np.mean([metrics(p, targets)[k] for p in per_model_preds]))
        for k in ("pearson_r", "r2", "spearman_r")
    }
    print(f"  [mean of members] pearson_r={mean_member['pearson_r']:.4f}  "
          f"r2={mean_member['r2']:.4f}  spearman={mean_member['spearman_r']:.4f}")
    print(f"  [ENSEMBLE]        pearson_r={em['pearson_r']:.4f}  "
          f"r2={em['r2']:.4f}  spearman={em['spearman_r']:.4f}  "
          f"(gain vs mean member: {em['pearson_r'] - mean_member['pearson_r']:+.4f} R)")

    out = pd.DataFrame({
        "gene": genes,
        "species": species,
        "half_life": targets,
        "ensemble_pred": ensemble_pred,
    })
    for i, p in enumerate(per_model_preds):
        out[f"pred_model_{i}"] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True,
                        help="A params.json from one of the runs (model arch + data_dirs).")
    parser.add_argument("--checkpoints", required=True, nargs="+",
                        help="Model checkpoints (model_best_*.pt). Globs are expanded.")
    parser.add_argument("--test_fold", type=int, default=0,
                        help="Held-out fold the ensemble is evaluated on [default: 0].")
    parser.add_argument("--species", nargs="+", default=["human", "mouse"],
                        help="Species to evaluate [default: human mouse].")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional CSV path to write per-gene predictions.")
    args = parser.parse_args()

    # Expand any globs the shell didn't (and de-duplicate, keeping order).
    checkpoints: List[str] = []
    for pattern in args.checkpoints:
        matched = sorted(glob.glob(pattern)) or [pattern]
        for c in matched:
            if c not in checkpoints:
                checkpoints.append(c)
    if not checkpoints:
        parser.error("No checkpoints found.")

    with open(args.params) as f:
        params = json.load(f)
    params_model = params["model"]
    data_dirs = params["data_dirs"]
    species_order = list(data_dirs.keys())  # head index == position in data_dirs

    frames = []
    for species in args.species:
        if species not in data_dirs:
            raise ValueError(
                f"Species {species!r} not in params data_dirs {species_order}"
            )
        frames.append(evaluate_species(
            species=species,
            species_index=species_order.index(species),
            tsv_path=data_dirs[species],
            checkpoints=checkpoints,
            params_model=params_model,
            test_fold=args.test_fold,
            batch_size=args.batch_size,
            device=args.device,
        ))

    if args.out:
        pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nWrote per-gene predictions to {args.out}")


if __name__ == "__main__":
    main()
