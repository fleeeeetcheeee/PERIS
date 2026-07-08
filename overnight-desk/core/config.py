"""Pydantic-validated YAML configs. No magic numbers in code — everything lives here."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class CostConfig(BaseModel):
    stock_bps_per_side: float = 7.5
    etf_bps_per_side: float = 3.0


class PortfolioConfig(BaseModel):
    top_k: int = 12
    max_weight: float = 0.10
    vol_target_annual: float = 0.12
    vol_lookback_days: int = 63
    # Skip a rebalance trade when the weight change is below this threshold —
    # the turnover penalty expressed as a minimum-trade band.
    min_trade_weight: float = 0.01
    # Recompute targets every N sessions; between rebalances weights drift with
    # prices and no trades are emitted. 1 = daily (legacy behavior).
    rebalance_every: int = 1
    # EMA halflife (sessions) applied to model scores before selection; 0 = off.
    # Smooths pick churn without touching the model itself.
    score_smoothing_halflife: int = 0
    # Weighting of the selected top-K: "inverse_vol" (legacy), "hrp" (hierarchical
    # risk parity), or "rmt_minvar" (min-variance on Marchenko-Pastur-cleaned corr).
    weighting: str = "inverse_vol"
    # Meta-labeling layer (models/meta.py) applied to the picks at each rebalance:
    #   "off"   - no meta model (legacy)
    #   "tilt"  - weights tilted by P(payoff), renormalized (exposure-neutral)
    #   "gate"  - picks with P(payoff) below meta_gate_threshold are passed on
    #             (their weight stays in cash, no replacement pick)
    #   "sized" - absolute bet sizing from the trailing-calibrated probability;
    #             unallocated weight stays in cash
    # A missing/uncalibrated probability is always neutral (multiplier 1, no veto).
    meta_mode: str = "off"
    meta_gate_threshold: float = 0.5
    capital: float = 10_000.0


class GatesConfig(BaseModel):
    earnings_exclusion_sessions: int = 2
    vix_term_structure: bool = True
    drawdown_halt_pct: float = 0.10
    # Statistical jump-model regime gate (models/regime.py): in the stress state,
    # exposure is halved and new entries blocked — same semantics as the VIX gate.
    jump_model: bool = False
    jump_penalty: float = 50.0


class CVConfig(BaseModel):
    label_horizon_days: int = 5
    purge_days: int = 5  # must be >= label horizon
    embargo_days: int = 2
    min_train_days: int = 504
    test_window_days: int = 63

    @field_validator("purge_days")
    @classmethod
    def purge_covers_horizon(cls, v: int, info) -> int:
        horizon = info.data.get("label_horizon_days", 5)
        if v < horizon:
            raise ValueError(f"purge_days ({v}) must be >= label_horizon_days ({horizon})")
        return v


class ModelConfig(BaseModel):
    seed: int = 42
    objective: str = "regression"  # regression-on-ranks
    num_boost_round: int = 400
    early_stopping_rounds: int = 50
    params: dict[str, float | int | str] = Field(default_factory=dict)


class Config(BaseModel):
    universe_file: str = "data/reference/universe.csv"
    start: date = date(2018, 1, 2)
    end: date | None = None
    benchmark: str = "SPY"
    # How many strategy variants were tried before settling on this config — feeds the
    # deflated-Sharpe multiple-testing penalty in headline reports. Increment when a
    # config change was chosen from an experiment matrix.
    selection_trials: int = 1
    costs: CostConfig = Field(default_factory=CostConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(raw)
