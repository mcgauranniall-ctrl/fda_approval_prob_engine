"""Statistical transforms that put trial results on comparable scales.

References:
- Newcombe (1998) Stat Med 17:873 — hybrid score interval for a difference of proportions (method 10).
- Chinn (2000) Stat Med 19:3127 — converting standardised mean differences to log odds ratios.
- O'Hagan, Stevens & Campbell (2005) Pharm Stat 4:187 — assurance (predictive probability of success).
"""

from __future__ import annotations

import math
import re

from scipy import stats

Z975 = stats.norm.ppf(0.975)


def parse_pvalue(text: str | float | None) -> tuple[float | None, str]:
    """'<0.001' -> (0.001, '<'); '0.0002' -> (0.0002, '='). Returns (None, '') if unparseable."""
    if text is None:
        return None, ""
    if isinstance(text, (int, float)):
        return float(text), "="
    m = re.match(r"\s*([<>≤≥]=?)?\s*(\d*\.?\d+(?:[eE]-?\d+)?)", str(text))
    if not m:
        return None, ""
    op = (m.group(1) or "=").replace("≤", "<").replace("≥", ">")[0]
    return float(m.group(2)), op


def z_from_p(p: float, two_sided: bool = True) -> float:
    """Magnitude of the z-statistic implied by a p-value."""
    p = min(max(p, 1e-300), 1.0)
    return float(stats.norm.isf(p / 2 if two_sided else p))


def z_from_ratio_ci(estimate: float, lower: float, upper: float) -> float:
    """z for a ratio measure (HR/OR/RR) from its 95% CI. Positive = ratio below 1 (benefit for hazards)."""
    se = (math.log(upper) - math.log(lower)) / (2 * Z975)
    return -math.log(estimate) / se


def z_from_difference_ci(estimate: float, lower: float, upper: float) -> float:
    se = (upper - lower) / (2 * Z975)
    return estimate / se


def p_consistent_with_ci(p: float, z_from_ci: float, tolerance: float = 0.5) -> bool:
    """Do a reported p-value and CI tell the same story? Compares |z| on the log scale of p."""
    return abs(z_from_p(p) - abs(z_from_ci)) <= tolerance + 0.15 * abs(z_from_ci)


def wilson_ci(x: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = x / n
    denom = 1 + Z975**2 / n
    centre = (p + Z975**2 / (2 * n)) / denom
    half = Z975 * math.sqrt(p * (1 - p) / n + Z975**2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """Risk difference p1 - p2 with Newcombe's hybrid score 95% interval."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson_ci(x1, n1)
    l2, u2 = wilson_ci(x2, n2)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lower, upper


def risk_ratio_ci(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """Risk ratio with a Katz log interval; 0.5 continuity correction if either event count is zero."""
    a, c = x1, x2
    if 0 in (a, c):
        a, c, n1, n2 = a + 0.5, c + 0.5, n1 + 1, n2 + 1
    rr = (a / n1) / (c / n2)
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return rr, math.exp(math.log(rr) - Z975 * se), math.exp(math.log(rr) + Z975 * se)


def fisher_p(x1: int, n1: int, x2: int, n2: int) -> float:
    return float(stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1])


def smd_to_log_or(d: float) -> float:
    """Chinn (2000): log OR ≈ d · π / √3."""
    return d * math.pi / math.sqrt(3)


def assurance(z_phase2: float, n_phase2: int, n_phase3: int, shrinkage: float = 0.75,
              alpha_z: float = Z975) -> float:
    """Predictive probability that a Phase 3 of size n3 reaches significance, given the Phase 2 z-statistic.

    With δ̂ ~ N(δ, se2²) and se3 = se2·√(n2/n3), the Phase 3 statistic is predictively
    z3 ~ N(k·z2·r, 1 + r²) where r = √(n3/n2) and k shrinks the Phase 2 effect for the winner's curse.
    `shrinkage` is a provisional value until it is estimated from historical P2→P3 pairs (hypothesis H2).
    """
    r = math.sqrt(n_phase3 / n_phase2)
    return float(stats.norm.cdf((shrinkage * z_phase2 * r - alpha_z) / math.sqrt(1 + r * r)))
