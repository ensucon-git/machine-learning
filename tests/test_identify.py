"""Identification tests on synthetic data with a known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from hpmpc.dataset import add_derived
from hpmpc.identify import fit_cop, fit_heating_curve, fit_thermal, make_windows, predict
from hpmpc.model.thermal import ThermalParams
from hpmpc.residual import build_features, fit_residual, one_step_residuals
from hpmpc.simulator import make_demo_dataset, perturbed


@pytest.fixture(scope="module")
def synthetic():
    from hpmpc.config import Config

    cfg = Config()
    cfg.entities.indoor_temp = "sensor.indoor"
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.entities.price = "sensor.price"
    cfg.training.max_windows = 120
    cfg.training.restarts = 1
    cfg.training.long_window_hours = 36.0
    cfg.validate()
    frame, truth = make_demo_dataset(cfg, days=12, seed=7)
    return cfg, frame, truth


def test_windows_have_the_expected_shape_and_scoring_mask(synthetic):
    cfg, frame, _ = synthetic
    ws = make_windows(cfg=cfg, frame=frame)
    steps_per_hour = 60 / cfg.training.resample_minutes
    expected = int((cfg.training.burn_in_hours + cfg.training.window_hours) * steps_per_hour)
    assert ws.offset.shape[1] == expected
    scored = int(cfg.training.window_hours * steps_per_hour)
    assert ws.eval_mask[0].sum() == scored
    assert not ws.eval_mask[0, 0]  # the burn-in is never scored


def test_windows_use_the_measured_supply_temperature_when_available(synthetic):
    cfg, frame, _ = synthetic
    assert make_windows(cfg=cfg, frame=frame).supply is not None
    assert make_windows(cfg=cfg, frame=frame.drop(columns=["t_supply"])).supply is None


def test_heating_curve_is_recovered(synthetic):
    cfg, frame, _ = synthetic
    curve = fit_heating_curve(frame, cfg)
    assert curve is not None and curve["accepted"]
    assert curve["curve_slope"] == pytest.approx(cfg.heat_pump.curve_slope, abs=0.05)
    assert curve["curve_offset"] == pytest.approx(cfg.heat_pump.curve_offset, abs=1.0)
    assert curve["outdoor_filter_hours"] == pytest.approx(cfg.heat_pump.outdoor_filter_hours, abs=1.0)
    assert curve["r2"] > 0.9


def test_heating_curve_needs_a_supply_sensor(synthetic):
    cfg, frame, _ = synthetic
    assert fit_heating_curve(frame.drop(columns=["t_supply"]), cfg) is None


def test_identification_recovers_the_building(synthetic):
    cfg, frame, truth = synthetic
    start = perturbed(truth)
    params, metrics = fit_thermal(frame, cfg, initial=start)

    # Prediction quality is the acceptance criterion.
    assert metrics["validation"]["rmse_c"] < 0.3
    assert metrics["validation"]["rmse_c"] < 0.5 * metrics["validation"]["baseline_rmse_c"]
    assert metrics["validation"]["rmse_c"] < metrics["prior"]["validation"]["rmse_c"]

    # The physically meaningful aggregates should also land close.
    assert params.heat_loss_w_per_k() == pytest.approx(truth.heat_loss_w_per_k(), rel=0.35)
    assert params.Hfloor == pytest.approx(truth.Hfloor, rel=0.5)
    assert 2.0 < params.time_constants_hours()["slab"] < 40.0


def test_identifiability_diagnostics_are_reported(synthetic):
    cfg, frame, truth = synthetic
    _, metrics = fit_thermal(frame, cfg, initial=perturbed(truth))
    ident = metrics["identifiability"]
    assert ident["condition_number"] > 1.0
    assert len(ident["weakest_direction"]) > 0


def test_fitted_parameters_predict_better_than_the_starting_guess(synthetic):
    cfg, frame, truth = synthetic
    ws = make_windows(cfg=cfg, frame=frame)
    start = perturbed(truth)
    fitted, _ = fit_thermal(frame, cfg, initial=start)
    err_start = np.abs(predict(start, cfg, ws) - ws.target)[ws.eval_mask].mean()
    err_fitted = np.abs(predict(fitted, cfg, ws) - ws.target)[ws.eval_mask].mean()
    assert err_fitted < err_start


def test_cop_fit_recovers_the_carnot_efficiency(synthetic):
    cfg, frame, truth = synthetic
    result = fit_cop(frame, cfg, truth)
    assert result is not None
    assert result["carnot_efficiency"] == pytest.approx(cfg.heat_pump.carnot_efficiency, abs=0.08)
    assert 1.5 < result["seasonal_cop_observed"] < 6.0


def test_cop_fit_needs_a_power_sensor(synthetic):
    cfg, frame, truth = synthetic
    assert fit_cop(frame.drop(columns=["power"]), cfg, truth) is None


def test_residual_features_are_exogenous_only(synthetic):
    cfg, frame, _ = synthetic
    features = build_features(frame.index, frame, cfg.site.timezone)
    assert "t_indoor" not in features.columns
    assert len(features) == len(frame)
    assert features.notna().all().all()


def test_residuals_are_small_when_the_model_is_the_truth(synthetic):
    cfg, frame, truth = synthetic
    residual = one_step_residuals(frame, cfg, truth)
    assert not residual.empty
    # Only sensor noise is left, so the leftover rate error must be tiny.
    assert float(np.abs(residual).median()) < 0.35


def test_residual_model_is_discarded_when_it_adds_nothing(synthetic):
    cfg, frame, truth = synthetic
    assert fit_residual(frame, cfg, truth) is None


def test_identification_reports_missing_history_clearly(synthetic):
    cfg, frame, _ = synthetic
    with pytest.raises(ValueError, match="contiguous history"):
        make_windows(cfg=cfg, frame=frame.iloc[:5])


def test_bad_parameters_are_clipped_into_the_feasible_set():
    low, high = ThermalParams.bounds()
    clipped = ThermalParams(Ci=1e9, Hfloor=-5.0).clipped().to_vector()
    assert np.all(clipped >= low) and np.all(clipped <= high)


def test_identifiability_separates_strong_from_weak_parameters(synthetic):
    cfg, frame, truth = synthetic
    _, metrics = fit_thermal(frame, cfg, initial=perturbed(truth))
    ident = metrics["identifiability"]
    names = set(ThermalParams.names())
    assert set(ident["well_determined"]) <= names
    assert set(ident["weakest_direction"]) <= names
    assert len(ident["singular_values"]) == len(names)
    assert ident["singular_values"][0] == pytest.approx(1.0)


def test_the_fit_reports_what_it_learned_about_the_weather():
    """Sun and wind are two of the things this exists to learn, so the summary
    should say what it concluded rather than leaving it inside the vector."""
    from hpmpc.model.thermal import ThermalParams

    params = ThermalParams(A_sol=5.5, f_sol_i=0.45, k_wind=0.045, Hie=170.0, Hme=35.0)
    calm = params.heat_loss_w_per_k()
    windy = params.heat_loss_w_per_k(wind=5.0)
    assert windy > calm
    # Only the indoor-to-exterior path feels the wind; the slab's does not.
    assert windy == pytest.approx(170.0 * (1 + 0.045 * 5) + 35.0)


def test_wind_and_sun_are_recovered_from_data(synthetic):
    """Two of the things this exists to learn. A house identified as indifferent
    to wind under-heats on the windy evening it was supposed to see coming."""
    cfg, frame, truth = synthetic
    params, metrics = fit_thermal(frame, cfg, initial=perturbed(truth))

    assert params.k_wind == pytest.approx(truth.k_wind, rel=0.4)
    assert params.A_sol == pytest.approx(truth.A_sol, rel=0.4)

    weather = metrics["weather_sensitivity"]
    assert weather["wind_pct_per_m_s"] == pytest.approx(100 * params.k_wind, abs=0.1)
    assert weather["solar_aperture_m2"] == pytest.approx(params.A_sol, abs=0.01)
    assert weather["ua_at_5_m_s"] > metrics["heat_loss_w_per_k"]


# ---------------------------------------------------------------------------
# The wood stove: an uncontrolled heat input that identification must not be
# allowed to blame on the house.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stoked():
    """A house with a stove fired twice a day, and the truth behind it."""
    from hpmpc.config import Config
    from hpmpc.simulator import true_params

    cfg = Config()
    cfg.entities.indoor_temp = "sensor.indoor"
    cfg.entities.outdoor_temp = "sensor.outdoor"
    cfg.entities.price = "sensor.price"
    cfg.entities.stove_temp = "sensor.stove"
    cfg.training.max_windows = 120
    cfg.training.restarts = 1
    cfg.training.long_window_hours = 36.0
    # No deadband, so the derived input is exactly what the true house burned
    # and the test measures identification rather than threshold arithmetic.
    cfg.training.stove_deadband_k = 0.0
    cfg.validate()

    truth = true_params()
    truth.k_stove = 800.0        # 4 kW at the 5 K peak of a fire
    frame, _ = make_demo_dataset(cfg, days=14, seed=11, params=truth, stove_per_day=2.0)
    # The real pipeline derives 'stove' from the sensor in add_derived; the
    # simulator stops at the raw sensor, so do what build_dataset would.
    frame = add_derived(frame, cfg)
    return cfg, frame, truth


def test_the_synthetic_stove_actually_burns(stoked):
    _, frame, _ = stoked
    assert "stove" in frame or "stove_temp" in frame
    excess = frame["stove_temp"] - frame["t_indoor"]
    assert excess.max() > 3.0           # fires are visible
    assert (excess > 1.0).mean() < 0.5  # and are not the whole dataset


def test_the_stove_gain_is_recovered(stoked):
    cfg, frame, truth = stoked
    fitted, _ = fit_thermal(frame, cfg, initial=perturbed(truth))
    assert fitted.k_stove == pytest.approx(truth.k_stove, rel=0.45)


def test_ignoring_the_stove_biases_the_house(stoked):
    """The point of the whole feature.

    Fitting the same data with the stove hidden forces the optimiser to blame
    the warmth on the building, and it does: the envelope, the sun or the slab
    moves. Seeing the stove puts them back.
    """
    cfg, frame, truth = stoked

    seen, _ = fit_thermal(frame, cfg, initial=perturbed(truth))
    # Hiding the column is exactly what an unconfigured entities.stove_temp
    # produces: make_windows then feeds the model an all-zero stove.
    hidden, _ = fit_thermal(frame.drop(columns=["stove", "stove_temp"], errors="ignore"),
                            cfg, initial=perturbed(truth))

    def err(p):
        """Worst relative error across the parameters a fire can masquerade as."""
        return max(
            abs(p.Hie - truth.Hie) / truth.Hie,
            abs(p.A_sol - truth.A_sol) / max(truth.A_sol, 1e-6),
            abs(p.Cm - truth.Cm) / truth.Cm,
        )

    assert err(seen) < err(hidden), (
        f"stove-aware fit should be closer to the truth: {err(seen):.3f} vs {err(hidden):.3f}"
    )

    # And it predicts better, which is what the controller actually consumes.
    # No absolute parameter tolerance here: this fixture is 14 days with one
    # restart, and the sub-percent recoveries quoted elsewhere come from much
    # longer runs. The comparison is the claim; the magnitude is not.
    ws = make_windows(cfg=cfg, frame=frame)
    rmse = {
        name: float(np.sqrt(np.mean((predict(p, cfg, ws) - ws.target)[ws.eval_mask] ** 2)))
        for name, p in (("seen", seen), ("hidden", hidden))
    }
    assert rmse["seen"] < rmse["hidden"], rmse
