#!/usr/bin/env python3
"""Run trust policy evaluation with mechanistic plausibility."""

from __future__ import annotations

import json

from arie.trust_policy_mech import CURVE_PATH, SCORES_PATH, SUMMARY_PATH, run_trust_policy_mech


def main() -> None:
    merged, summary, curve_df = run_trust_policy_mech(eps=0.05)

    # Write artifacts
    CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(CURVE_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scores_cols = [
        "row_id",
        "seed",
        "fold",
        "y_true",
        "y_pred",
        "error",
        "mech_support",
        "mech_missing_frac",
        "mech_plausibility",
        "score_combined",
        "score_combined_plus_mech",
        "score_combined_plus_mech_eps",
    ]
    merged[scores_cols].to_csv(SCORES_PATH, index=False)

    alignment = summary.get("alignment", {})
    pred_mech = alignment.get("predictions_vs_mech", {})
    print(
        "Alignment check (predictions vs mech scores): "
        f"pred n={pred_mech.get('left_rows')} | "
        f"mech n={pred_mech.get('right_rows')} | "
        f"sym_diff={pred_mech.get('sym_diff')}"
    )

    print("Coverage comparison (balanced accuracy mean±std):")
    for cov_key in ["0.90", "0.70", "0.50"]:
        variants = summary["summary"].get(cov_key, {})
        line = [f"coverage {cov_key}:"]
        for variant in ["confidence", "ood", "combined", "combined_plus_mech", "combined_plus_mech_eps"]:
            stats = variants.get(variant, {})
            mean = stats.get("balanced_accuracy_mean")
            std = stats.get("balanced_accuracy_std")
            if mean is None:
                entry = f"{variant} n/a"
            else:
                entry = f"{variant} {mean:.4f}±{std:.4f}"
            line.append(entry)
        deltas = summary["deltas_vs_combined"].get(cov_key, {})
        delta_mech = deltas.get("combined_plus_mech", {}).get("delta_vs_combined")
        delta_mech_eps = deltas.get("combined_plus_mech_eps", {}).get("delta_vs_combined")
        line.append(f"delta_vs_combined mech={delta_mech:+.4f}" if delta_mech is not None else "delta_vs_combined mech=n/a")
        line.append(f"mech_eps={delta_mech_eps:+.4f}" if delta_mech_eps is not None else "mech_eps=n/a")
        print("  " + " | ".join(line))

    print("Best variant per coverage:")
    for cov_key, winner in summary["best_variants"].items():
        print(
            f"- coverage {cov_key}: {winner.get('variant')} "
            f"({winner.get('balanced_accuracy_mean')})"
        )

    print("Outputs written:")
    print(f"- {CURVE_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {SCORES_PATH}")


if __name__ == "__main__":
    main()
