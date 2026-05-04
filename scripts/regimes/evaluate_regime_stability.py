"""Measure cross-seed stability of regime prototypes.

Supports two alignment strategies:
  - 'greedy' (original): per-prototype best-match cosine similarity
  - 'hungarian': optimal 1-to-1 matching via linear_sum_assignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment


def load_prototypes(
    summary_path: Path, residual_target: torch.Tensor, topk: int
) -> tuple[torch.Tensor, list[int]]:
    """Load prototype vectors for top-used regimes.

    Parameters
    ----------
    summary_path : Path
        Path to regime_prototypes_summary.json.
    residual_target : torch.Tensor
        Full residual tensor of shape (N, H, C).
    topk : int
        Number of top regimes to consider.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        Prototype matrix (topk, D) and regime IDs.
    """
    summary = json.loads(summary_path.read_text())
    regime_ids = summary["summary"]["most_used_regimes"][:topk]
    prototypes = []
    for regime_idx in regime_ids:
        indices = summary["regimes"][str(regime_idx)]["top_indices"][:topk]
        chosen = residual_target[indices]
        prototype = chosen.mean(dim=0).flatten()
        prototypes.append(prototype)
    stacked = torch.stack(prototypes, dim=0)
    return stacked, regime_ids


def greedy_alignment(
    protos_i: torch.Tensor, protos_j: torch.Tensor
) -> dict[str, float]:
    """Original greedy best-match alignment (non-unique matching).

    Parameters
    ----------
    protos_i, protos_j : torch.Tensor
        Prototype matrices of shape (K, D).

    Returns
    -------
    dict
        mean_best_match_a_to_b, mean_best_match_b_to_a.
    """
    norm_i = torch.nn.functional.normalize(protos_i, dim=-1)
    norm_j = torch.nn.functional.normalize(protos_j, dim=-1)
    sim = norm_i @ norm_j.T
    best_i = float(sim.max(dim=1).values.mean().item())
    best_j = float(sim.max(dim=0).values.mean().item())
    return {
        "mean_best_match_a_to_b": best_i,
        "mean_best_match_b_to_a": best_j,
        "mean_greedy": (best_i + best_j) / 2.0,
    }


def hungarian_alignment(
    protos_i: torch.Tensor, protos_j: torch.Tensor
) -> dict[str, float | list[list[int]]]:
    """Optimal 1-to-1 prototype matching via Hungarian algorithm.

    Maximizes total cosine similarity under unique assignment constraint.

    Parameters
    ----------
    protos_i, protos_j : torch.Tensor
        Prototype matrices of shape (K, D).

    Returns
    -------
    dict
        mean_aligned_similarity, matched_pairs, per_pair_similarities.
    """
    norm_i = torch.nn.functional.normalize(protos_i, dim=-1)
    norm_j = torch.nn.functional.normalize(protos_j, dim=-1)
    sim = (norm_i @ norm_j.T).numpy()  # (K, K)

    # Hungarian maximizes by minimizing the negative
    row_ind, col_ind = linear_sum_assignment(-sim)

    pair_sims = [float(sim[r, c]) for r, c in zip(row_ind, col_ind, strict=True)]
    matched_pairs = [[int(r), int(c)] for r, c in zip(row_ind, col_ind, strict=True)]
    mean_sim = sum(pair_sims) / max(len(pair_sims), 1)

    return {
        "mean_aligned_similarity": mean_sim,
        "matched_pairs": matched_pairs,
        "per_pair_similarities": pair_sims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-seed regime stability analysis")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to residual dataset .pt")
    parser.add_argument("--summaries", nargs="+", required=True,
                        help="Paths to regime_prototypes_summary.json from different seeds")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alignment", type=str, default="both",
                        choices=["greedy", "hungarian", "both"],
                        help="Alignment strategy for cross-seed comparison")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    residual_target = payload["residual_target"]

    seed_prototypes: list[tuple[str, torch.Tensor, list[int]]] = []
    for summary_file in args.summaries:
        path = Path(summary_file)
        protos, regime_ids = load_prototypes(path, residual_target, args.topk)
        seed_prototypes.append((path.parent.name, protos, regime_ids))

    pairwise = []
    for i in range(len(seed_prototypes)):
        for j in range(i + 1, len(seed_prototypes)):
            name_i, protos_i, ids_i = seed_prototypes[i]
            name_j, protos_j, ids_j = seed_prototypes[j]
            overlap = len(set(ids_i).intersection(ids_j))

            entry: dict[str, object] = {
                "seed_a": name_i,
                "seed_b": name_j,
                "top_regime_id_overlap": overlap,
            }

            if args.alignment in ("greedy", "both"):
                greedy = greedy_alignment(protos_i, protos_j)
                entry["greedy"] = greedy

            if args.alignment in ("hungarian", "both"):
                hung = hungarian_alignment(protos_i, protos_j)
                entry["hungarian"] = hung

            pairwise.append(entry)

    # Aggregate summaries
    result: dict[str, object] = {"pairwise": pairwise}

    avg_overlap = sum(p["top_regime_id_overlap"] for p in pairwise) / max(len(pairwise), 1)
    result["average_top_regime_id_overlap"] = avg_overlap

    if args.alignment in ("greedy", "both"):
        avg_greedy = sum(p["greedy"]["mean_greedy"] for p in pairwise) / max(len(pairwise), 1)
        result["average_greedy_similarity"] = avg_greedy

    if args.alignment in ("hungarian", "both"):
        avg_hung = sum(
            p["hungarian"]["mean_aligned_similarity"] for p in pairwise
        ) / max(len(pairwise), 1)
        result["average_hungarian_similarity"] = avg_hung

    # Backward compatibility: keep average_best_match_similarity for greedy
    if args.alignment in ("greedy", "both"):
        result["average_best_match_similarity"] = avg_greedy

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_dir = Path(args.summaries[0]).parent.parent
        suffix = f"_stability{'_hungarian' if args.alignment == 'hungarian' else ''}.json"
        out_path = out_dir / f"regime{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved stability report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
