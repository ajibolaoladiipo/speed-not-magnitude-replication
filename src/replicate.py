#!/usr/bin/env python3
"""Methodological Monte Carlo for a single-vector AECM speed test.

The generated series are used only to examine estimator behavior. They are not
empirical observations or substitutes for external microdata. The simulation
compares a common long-run slope with sign-dependent adjustment against
recurring regime-specific long-run slopes with common adjustment. The latter
violates the fitted single-vector specification. Its rejection frequency is
not a valid test or general misclassification probability for slope asymmetry.
"""
import json
import math
import argparse
from statistics import NormalDist
from pathlib import Path

import numpy as np

SEED = 20261119
REPS = 1000
ALPHA = 0.05


def hac_speed_test(dp, q, p, max_lag=4):
    """Fit one-vector Engle-Granger residual AECM and HAC Wald test alpha+=alpha-."""
    beta = np.polyfit(q, p, 1)[0]
    z = p - beta * q
    lagz = z[1:-1]
    y = dp[1:]
    dq = np.diff(q)
    x = np.column_stack((np.ones(len(y)), np.maximum(lagz, 0),
                         np.minimum(lagz, 0), dq[1:], dp[:-1], dq[:-1]))
    n, k = x.shape
    inv = np.linalg.pinv(x.T @ x)
    b = inv @ x.T @ y
    e = y - x @ b
    scores = x * e[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    cov = inv @ meat @ inv
    contrast = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    se = math.sqrt(max(float(contrast @ cov @ contrast), 1e-20))
    stat = float(contrast @ b / se)
    pval = float(2 * (1.0 - NormalDist().cdf(abs(stat))))
    return pval, beta


def draw_speed_only(rng, n, equal_speed=False):
    dq = rng.normal(0, 0.124, n)
    q = np.r_[0.0, np.cumsum(dq)]
    u = np.zeros(n + 1)
    eps = rng.normal(0, 0.098, n)
    for j in range(1, n + 1):
        a = -0.29 if equal_speed else (-0.13 if u[j - 1] > 0 else -0.40)
        u[j] = (1 + a) * u[j - 1] + eps[j - 1]
    p = 0.956 * q + u
    return np.diff(p), q, p


def draw_magnitude_only(rng, n):
    dq = rng.normal(0, 0.124, n)
    q = np.r_[0.0, np.cumsum(dq)]
    # Regime-specific long-run slope with common adjustment speed.
    # Regime follows the sign of the current upstream price change.
    p = np.zeros(n + 1)
    eps = rng.normal(0, 0.098, n)
    for j in range(1, n + 1):
        beta_s = 1.05 if dq[j - 1] >= 0 else 0.85
        z_prev = p[j - 1] - beta_s * q[j - 1]
        p[j] = p[j - 1] + beta_s * dq[j - 1] - 0.29 * z_prev + eps[j - 1]
    dp = np.diff(p)
    return dp, q, p


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the published-parameter benchmark and estimator simulation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="directory for generated JSON outputs",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    out = []
    for n in (120, 240):
        for name, draw in (("null_symmetric", lambda r, n: draw_speed_only(r, n, True)),
                           ("speed_only", draw_speed_only),
                           ("magnitude_only", draw_magnitude_only)):
            rejects = 0
            beta_hats = []
            for _ in range(REPS):
                dp, q, p = draw(rng, n)
                pval, beta = hac_speed_test(dp, q, p)
                rejects += pval < ALPHA
                beta_hats.append(beta)
            out.append({
                "T": n, "dgp": name, "replications": REPS,
                "speed_symmetry_rejections": rejects,
                "rejection_rate": rejects / REPS,
                "mean_single_vector_beta": float(np.mean(beta_hats)),
                "seed": SEED,
                "test": "two-sided Newey-West HAC Wald, lag 4, normal reference, alpha_plus=alpha_minus",
                "dgp_note": (
                    "True regime-specific long-run slopes 1.05/0.85; common alpha=-0.29. Single-vector null is misspecified; rejection rate is not a valid test of slope asymmetry."
                    if name == "magnitude_only" else ("Common beta=0.956 and alpha_plus=alpha_minus=-0.29." if name == "null_symmetric" else "Common beta=0.956; true alpha_plus=-0.13, alpha_minus=-0.40.")
                ),
            })
    target = args.output_dir.resolve()
    target.mkdir(exist_ok=True)
    (target / "single_vector_monte_carlo.json").write_text(json.dumps(out, indent=2) + "\n")
    def exposure(alpha, horizon):
        residual = math.exp(alpha * horizon)
        return {
            "alpha": alpha,
            "horizon_months": horizon,
            "residual_share": residual,
            "completion_percent": 100.0 * (1.0 - residual),
            "finite_per_V_abs_Z0": (1.0 - residual) / abs(alpha),
            "lifetime_per_V_abs_Z0": 1.0 / abs(alpha),
        }
    plus, minus = exposure(-0.13, 30.0), exposure(-0.40, 10.0)
    calibration = {
        "source": "Abdulai (2000), Accra maize; coefficients and horizons as reported",
        "units": "volume-weighted log-wedge months; normalized by V*|Z0|",
        "buyer_adverse_Z_positive": plus,
        "other_Z_negative": minus,
        "gap_plus_minus": plus["finite_per_V_abs_Z0"] - minus["finite_per_V_abs_Z0"],
        "realized_ratio": plus["finite_per_V_abs_Z0"] / minus["finite_per_V_abs_Z0"],
        "limiting_speed_ratio": 0.40 / 0.13,
    }
    (target / "abdulai_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({"monte_carlo": out, "abdulai_calibration": calibration}, indent=2))


if __name__ == "__main__":
    main()
