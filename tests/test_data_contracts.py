"""Pin the design contracts of the data mixer.

These run without GPU and without HF auth. They protect the pieces that are
easy to silently break across ablation cells.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from babylm.config import BYTE_PREMIUM, DataConfig
from babylm.data import MixtureBudget, schedule_weights, _normalize


def test_byte_premium_table_unchanged():
    # If these change, every prior result needs re-byte-premium-adjusting.
    assert BYTE_PREMIUM["eng"] == 1.000000
    assert abs(BYTE_PREMIUM["nld"] - 1.051606) < 1e-9
    assert abs(BYTE_PREMIUM["zho"] - 0.935966) < 1e-9


def test_normalize_drops_zeros_and_renormalizes():
    w = _normalize({"eng": 1.0, "nld": 0.0, "zho": 1.0})
    assert "nld" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_static_mixture_returns_constant():
    cfg = DataConfig(mixture={"eng": 0.5, "nld": 0.3, "zho": 0.2}, schedule=[])
    for f in [0.0, 0.25, 0.5, 0.99]:
        w = schedule_weights(cfg, f)
        assert abs(w["eng"] - 0.5) < 1e-9
        assert abs(w["nld"] - 0.3) < 1e-9
        assert abs(w["zho"] - 0.2) < 1e-9


def test_curriculum_piecewise_constant():
    cfg = DataConfig(
        mixture={"eng": 1.0},  # ignored when schedule is set
        schedule=[
            (0.0, {"eng": 1.0}),
            (0.5, {"eng": 0.6, "nld": 0.4}),
            (0.9, {"eng": 0.34, "nld": 0.33, "zho": 0.33}),
        ],
    )
    # Stage 1: monolingual EN
    w = schedule_weights(cfg, 0.1)
    assert w == {"eng": 1.0}
    # Stage 2: EN+NL
    w = schedule_weights(cfg, 0.6)
    assert set(w.keys()) == {"eng", "nld"}
    assert abs(w["eng"] - 0.6) < 1e-9
    # Stage 3: trilingual
    w = schedule_weights(cfg, 0.95)
    assert set(w.keys()) == {"eng", "nld", "zho"}


def test_budget_caps_total_and_epochs():
    avail = {"eng": 100, "nld": 100, "zho": 100}
    # corpus budget 200, max_epochs 2 -> total effective budget 400.
    b = MixtureBudget(
        budget_reference_tokens=200,
        max_epochs=2,
        available_reference_tokens=avail,
        consumed_reference_tokens={k: 0 for k in avail},
    )
    assert b.total_token_budget == 400
    assert b.can_continue()
    b.add("eng", 100)
    assert b.can_continue()           # 100 of 400 total
    b.add("nld", 100)
    assert b.can_continue()           # 200 of 400 total
    b.add("zho", 200)
    # 400/400 — total effective cap hit (and zho has done 2 epochs)
    assert not b.can_continue()


def test_budget_per_language_epoch_cap():
    avail = {"eng": 100, "nld": 100, "zho": 100}
    # corpus 10000, max_epochs 2 -> 20000 effective budget; per-lang cap still 200.
    b = MixtureBudget(
        budget_reference_tokens=10_000,
        max_epochs=2,
        available_reference_tokens=avail,
        consumed_reference_tokens={k: 0 for k in avail},
    )
    b.add("eng", 250)  # eng has gone 2.5 epochs of its 100-token corpus
    assert not b.can_continue()


if __name__ == "__main__":
    test_byte_premium_table_unchanged()
    test_normalize_drops_zeros_and_renormalizes()
    test_static_mixture_returns_constant()
    test_curriculum_piecewise_constant()
    test_budget_caps_total_and_epochs()
    test_budget_per_language_epoch_cap()
    print("ok")
