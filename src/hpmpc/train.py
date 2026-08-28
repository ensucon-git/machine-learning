"""End-to-end training: history in, model on disk out.

    hpmpc collect     # pull Home Assistant history into data/history.csv.gz
    hpmpc train       # fit everything and write models/thermal_model.json

The saved model file carries the building parameters, any heat-pump settings
that were learned rather than configured, and the full metrics of the run, so a
model can always be traced back to the data it came from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .identify import fit_cop, fit_heating_curve, fit_thermal
from .model.thermal import ThermalParams, load_params, save_params
from .residual import ResidualModel, fit_residual

log = logging.getLogger(__name__)

# Heat-pump settings that identification is allowed to override.
PUMP_OVERRIDE_KEYS = (
    "curve_slope",
    "curve_offset",
    "curve_ref",
    "outdoor_filter_hours",
    "carnot_efficiency",
    "efficiency_scale",
    "capacity_scale",
    "standby_power_w",
)


def apply_pump_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a config whose heat-pump section reflects the learned values."""
    usable = {k: float(v) for k, v in (overrides or {}).items() if k in PUMP_OVERRIDE_KEYS}
    if not usable:
        return cfg
    return replace(cfg, heat_pump=replace(cfg.heat_pump, **usable))


def train(cfg: Config, frame: pd.DataFrame, fit_curve: bool = True) -> dict[str, Any]:
    """Fit the heating curve, the building and the COP, then the residual model."""
    report: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "rows": int(len(frame)),
            "start": str(frame.index[0]),
            "end": str(frame.index[-1]),
            "span_days": round((frame.index[-1] - frame.index[0]).total_seconds() / 86400.0, 2),
            "resample_minutes": cfg.training.resample_minutes,
        },
    }
    overrides: dict[str, Any] = {}

    # 1. Heating curve -----------------------------------------------------
    if fit_curve:
        curve = fit_heating_curve(frame, cfg)
        report["heating_curve"] = curve
        if curve and curve.get("accepted"):
            overrides.update(
                {
                    "curve_slope": curve["curve_slope"],
                    "curve_offset": curve["curve_offset"],
                    "outdoor_filter_hours": curve["outdoor_filter_hours"],
                }
            )
            log.info(
                "Learned heating curve: supply = %.2f + %.3f * (%.0f - T_out_filtered), tau = %.1f h (R2 %.2f)",
                curve["curve_offset"], curve["curve_slope"], curve["curve_ref"],
                curve["outdoor_filter_hours"], curve["r2"],
            )
    working = apply_pump_overrides(cfg, overrides)

    # 2. Building ----------------------------------------------------------
    initial = _previous_params(cfg)
    params, thermal_metrics = fit_thermal(frame, working, initial=initial)
    report["thermal"] = thermal_metrics
    log.info(
        "Building fit: validation RMSE %.3f C (persistence baseline %.3f C), UA %.0f W/K, slab tau %.1f h",
        thermal_metrics["validation"].get("rmse_c", float("nan")),
        thermal_metrics["validation"].get("baseline_rmse_c", float("nan")),
        params.heat_loss_w_per_k(),
        params.time_constants_hours()["slab"],
    )

    # 3. COP ---------------------------------------------------------------
    cop = fit_cop(frame, working, params)
    report["cop"] = cop
    if cop:
        overrides.update(
            {k: cop[k] for k in ("carnot_efficiency", "efficiency_scale", "standby_power_w") if k in cop}
        )
        correction = cop.get("efficiency_scale", cop.get("carnot_efficiency"))
        log.info(
            "Efficiency fit: correction %.3f, standby %.0f W, observed seasonal COP %.2f, "
            "power RMSE %.0f W, backup heater ran %.1f h",
            correction, cop["standby_power_w"], cop["seasonal_cop_observed"],
            cop["power_rmse_w"], cop.get("backup_heater_hours", 0.0),
        )
        drift = cop.get("residual_by_ambient", {})
        if drift and max(abs(v) for v in drift.values()) > 250:
            log.warning(
                "Power error varies with outdoor temperature (%s W by bin) - the shape of the "
                "performance map looks wrong. Replace it with the databook table.",
                drift,
            )
    working = apply_pump_overrides(cfg, overrides)

    # 4. Residual ----------------------------------------------------------
    residual: ResidualModel | None = None
    if cfg.training.use_residual_model:
        residual = fit_residual(frame, working, params)
        report["residual"] = residual.metrics if residual else {"accepted": False}

    report["pump_overrides"] = overrides
    report["parameters"] = params.to_dict()

    save_params(cfg.model_path, params, metadata=report)
    if residual is not None:
        residual.save(cfg.residual_path)
    elif Path(cfg.residual_path).exists():
        Path(cfg.residual_path).unlink()
    log.info("Saved model to %s", cfg.model_path)
    return report


def _previous_params(cfg: Config) -> ThermalParams | None:
    """Warm-start from the last fit if one exists - the house does not change
    much between retrainings, and it keeps successive fits stable."""
    try:
        params, _ = load_params(cfg.model_path)
        log.info("Warm-starting identification from the previous model")
        return params
    except (OSError, ValueError, KeyError):
        return None


class ModelNotTrained(FileNotFoundError):
    """No model on disk yet. Expected on a fresh install, not a fault."""


def load_model(cfg: Config) -> tuple[Config, ThermalParams, ResidualModel | None, dict[str, Any]]:
    """Load a trained model and return a config with the learned pump settings."""
    if not Path(cfg.model_path).exists():
        raise ModelNotTrained(
            f"No trained model at {cfg.model_path}. A model is built from your own house's "
            "history, so a fresh install has none yet:\n"
            "  hpmpc collect --days 45   # pull history out of Home Assistant\n"
            "  hpmpc train               # fit the house\n"
            "Until then 'hpmpc serve' and 'hpmpc run' still start, hold the offset at "
            "control.fallback_offset and keep collecting history."
        )
    params, metadata = load_params(cfg.model_path)
    working = apply_pump_overrides(cfg, metadata.get("pump_overrides", {}))
    residual = None
    if cfg.training.use_residual_model and Path(cfg.residual_path).exists():
        try:
            residual = ResidualModel.load(cfg.residual_path)
        except Exception as exc:  # pragma: no cover - never block control on this
            log.warning("Could not load the residual model (%s); continuing with physics only", exc)
    return working, params, residual, metadata


def load_model_if_trained(
    cfg: Config,
) -> tuple[Config, ThermalParams | None, ResidualModel | None, dict[str, Any]]:
    """Like :func:`load_model`, but an untrained install is a state, not an error.

    The controller has real work to do before it can optimise anything - keeping
    the pump supplied with a sensor reading and building the history the fit
    needs - so refusing to start until a model exists would be exactly backwards.
    """
    try:
        return load_model(cfg)
    except ModelNotTrained:
        return cfg, None, None, {}


def summarise(report: dict[str, Any]) -> str:
    """Compact human-readable training summary for the CLI."""
    lines: list[str] = []
    data = report.get("data", {})
    lines.append(f"Data:      {data.get('rows')} rows, {data.get('span_days')} days ({data.get('start')} -> {data.get('end')})")
    curve = report.get("heating_curve")
    if curve:
        status = "applied" if curve.get("accepted") else "rejected (keeping configured curve)"
        lines.append(
            f"Curve:     slope {curve['curve_slope']}, offset {curve['curve_offset']} C, "
            f"filter {curve['outdoor_filter_hours']} h, R2 {curve['r2']} - {status}"
        )
    thermal = report.get("thermal", {})
    val = thermal.get("validation", {})
    lines.append(
        f"Building:  validation RMSE {val.get('rmse_c')} C over {report.get('thermal',{}).get('n_windows_validation')} windows "
        f"(persistence {val.get('baseline_rmse_c')} C)"
    )
    long_val = thermal.get("validation_long_horizon")
    if long_val:
        lines.append(f"           48 h horizon RMSE {long_val.get('rmse_c')} C")
    lines.append(
        f"           UA {thermal.get('heat_loss_w_per_k')} W/K, time constants {thermal.get('time_constants_hours')}"
    )
    ident = thermal.get("identifiability", {})
    if ident:
        lines.append(
            f"           identifiability: condition number {ident.get('condition_number')}, "
            f"weakest direction {list((ident.get('weakest_direction') or {}).keys())}"
        )
    cop = report.get("cop")
    if cop:
        correction = cop.get("efficiency_scale", cop.get("carnot_efficiency"))
        label = "efficiency scale" if "efficiency_scale" in cop else "Carnot efficiency"
        lines.append(
            f"Pump:      {label} {correction}, standby {cop['standby_power_w']} W, "
            f"observed SCOP {cop['seasonal_cop_observed']}, power RMSE {cop['power_rmse_w']} W"
        )
        lines.append(f"           measured via {cop.get('method', 'unknown')}")
        if cop.get("backup_heater_hours"):
            lines.append(f"           backup heater active {cop['backup_heater_hours']} h in this data")
        if cop.get("residual_by_ambient"):
            lines.append(f"           power error by outdoor bin (W): {cop['residual_by_ambient']}")
        disaggregation = cop.get("disaggregation")
        if disaggregation:
            uncertainty = disaggregation.get("efficiency_scale_uncertainty")
            lines.append(
                f"           heat pump is {100 * (disaggregation.get('heatpump_share_of_measured') or 0):.0f}% "
                f"of the measured power over {disaggregation.get('span_hours', 0) / 24:.0f} days"
                + (f", scale precise to +/-{100 * uncertainty:.1f}%" if uncertainty else "")
            )
            ev = disaggregation.get("ev_power_estimated_w")
            if ev:
                lines.append(
                    f"           car charger inferred at {ev / 1000:.1f} kW "
                    f"(ratio to nominal {disaggregation.get('ev_power_ratio')})"
                )
        for warning in cop.get("warnings", []):
            lines.append(f"           WARNING: {warning}")
    residual = report.get("residual")
    if residual and residual.get("accepted"):
        lines.append(
            f"Residual:  {residual['uncorrected_rmse_k_per_h']} -> {residual['residual_rmse_k_per_h']} K/h "
            f"({100 * residual['improvement']:.0f}% better)"
        )
    elif residual is not None:
        lines.append("Residual:  not used (no improvement over the physical model)")
    return "\n".join(lines)


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
