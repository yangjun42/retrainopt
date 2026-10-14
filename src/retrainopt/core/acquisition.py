"""Acquisition functions used by the approximate planner."""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Dict, Tuple


def _normal_pdf(z_value: float) -> float:
    return math.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z_value: float) -> float:
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


class AcquisitionManager:
    """Compute acquisition scores to *minimize* during candidate selection."""

    def parse_name(self, name: str) -> Tuple[str, Dict[str, float]]:
        if not isinstance(name, str):
            raise ValueError("acquisition must be a supported name.")
        normalized = name.strip().lower()
        if normalized in {"expected", "expected_cost"}:
            return "expected", {}
        if normalized == "variance":
            return "variance", {}
        if normalized in {"optimistic", "lcb"}:
            return "optimistic", {}
        if normalized in {"pessimistic", "ucb"}:
            return "pessimistic", {}
        if normalized in {"expected_improvement", "ei"}:
            return "expected_improvement", {}
        if normalized in {"probability_improvement", "pi"}:
            return "probability_improvement", {}

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

        raise ValueError(f"Unknown acquisition: {name}")

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
        """Score a complete stage cost; EI/PI are zero at zero variance.

        ``best_cost`` and ``xi`` use the same weighted cost units as ``mean``.
        Only explicit parameters embedded in a name override keyword values.
        """
        acq_type, parsed = self.parse_name(acquisition)
        beta = parsed.get("beta", beta)
        xi = parsed.get("xi", xi)
        for label, value in (("mean", mean), ("std", std), ("beta", beta), ("xi", xi)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite real number.")
        if std < 0 or beta < 0 or xi < 0:
            raise ValueError("std, beta and xi must be nonnegative.")
        if best_cost is not None and (
            isinstance(best_cost, bool) or not isinstance(best_cost, Real) or not math.isfinite(best_cost)
        ):
            raise ValueError("best_cost must be finite when provided.")

        if acq_type == "expected":
            return float(mean)
        if acq_type == "variance":
            result = -(std * std)
        elif acq_type == "optimistic":
            result = float(mean) - beta * std
        elif acq_type == "pessimistic":
            result = float(mean) + beta * std
        else:
            if best_cost is None:
                raise ValueError("EI/PI require a best observed complete stage cost.")
            if std == 0.0:
                return 0.0
            improvement = float(best_cost) - float(mean) - xi
            if not math.isfinite(improvement):
                raise ValueError("Acquisition improvement overflowed; rescale the cost units.")
            z_value = improvement / std
            if acq_type == "expected_improvement":
                result = -max(improvement * _normal_cdf(z_value) + std * _normal_pdf(z_value), 0.0)
            else:
                result = -_normal_cdf(z_value)
        if not math.isfinite(result):
            raise ValueError("Acquisition score overflowed; rescale the cost units.")
        return float(result)

