"""Acquisition functions used by the approximate planner."""

from __future__ import annotations

import math
import re
from typing import Dict, Tuple


def _normal_pdf(z_value: float) -> float:
    return math.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z_value: float) -> float:
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


class AcquisitionManager:
    """Compute acquisition scores to *minimize* during candidate selection."""

    def parse_name(self, name: str) -> Tuple[str, Dict[str, float]]:
        normalized = (name or "expected").strip().lower()
        if normalized in {"expected", "expected_cost"}:
            return "expected", {}
        if normalized == "variance":
            return "variance", {}
        if normalized in {"optimistic", "lcb"}:
            return "optimistic", {"beta": 1.0}
        if normalized in {"pessimistic", "ucb"}:
            return "pessimistic", {"beta": 1.0}
        if normalized in {"expected_improvement", "ei"}:
            return "expected_improvement", {"xi": 0.01}
        if normalized in {"probability_improvement", "pi"}:
            return "probability_improvement", {"xi": 0.1}

        match = re.fullmatch(r"(lcb|ucb)_beta_([0-9]+(?:\.[0-9]+)?)", normalized)
        if match:
            kind, beta_value = match.groups()
            return ("optimistic" if kind == "lcb" else "pessimistic", {"beta": float(beta_value)})

        match = re.fullmatch(r"(ei|pi)_xi_([0-9]+(?:\.[0-9]+)?)", normalized)
        if match:
            kind, xi_value = match.groups()
            return (
                "expected_improvement" if kind == "ei" else "probability_improvement",
                {"xi": float(xi_value)},
            )

        return "expected", {}

    def score(
        self,
        mean: float,
        std: float,
        acquisition: str,
        *,
        beta: float = 1.0,
        best_cost: float | None = None,
        xi: float = 0.01,
    ) -> float:
        """Return an acquisition score where lower is better."""
        acq_type, parsed = self.parse_name(acquisition)
        beta = float(parsed.get("beta", beta))
        xi = float(parsed.get("xi", xi))
        std = max(0.0, float(std))

        if acq_type == "expected":
            return float(mean)
        if acq_type == "variance":
            return -(std ** 2)
        if acq_type == "optimistic":
            return float(mean) - beta * std
        if acq_type == "pessimistic":
            return float(mean) + beta * std

        if best_cost is None:
            best_cost = float(mean)

        if std <= 1e-12:
            improvement = max(float(best_cost) - float(mean) - xi, 0.0)
            if acq_type == "expected_improvement":
                return -improvement
            if acq_type == "probability_improvement":
                return -float(improvement > 0.0)
            return float(mean)

        z_value = (float(best_cost) - float(mean) - xi) / std
        if acq_type == "expected_improvement":
            improvement = (float(best_cost) - float(mean) - xi) * _normal_cdf(z_value) + std * _normal_pdf(z_value)
            return -max(improvement, 0.0)
        if acq_type == "probability_improvement":
            return -_normal_cdf(z_value)
        return float(mean)

