"""Command line interface.

    hpmpc demo                  run the whole pipeline on a synthetic house
    hpmpc init-config           write a starting config.yaml
    hpmpc check                 verify config and Home Assistant connectivity
    hpmpc collect               pull recorder history into a dataset
    hpmpc excite                run the identification experiment
    hpmpc train                 fit the model
    hpmpc plan                  solve once and print the plan (writes nothing)
    hpmpc backtest              replay history, MPC vs constant offset
    hpmpc run                   the control loop
    hpmpc serve                 control loop + local HTTP API
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import Config, load_config
from .controller import Controller
from .dataset import build_dataset, describe, load_dataset, save_dataset
from .evaluate import backtest, format_backtest
from .ha import HomeAssistant, HomeAssistantError
from .train import load_model, summarise, train, write_report

log = logging.getLogger("hpmpc")


def example_config_text() -> str:
    """The annotated starting config, shipped as package data so it is
    available from an installed wheel and not only from a source checkout."""
    from importlib.resources import files

    return (files("hpmpc") / "resources" / "config.example.yaml").read_text(encoding="utf-8")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _connect(cfg: Config) -> HomeAssistant:
    ha = HomeAssistant(cfg.home_assistant)
    if not ha.ping():
        raise SystemExit(f"Could not reach Home Assistant at {cfg.home_assistant.base_url}")
    return ha


# --------------------------------------------------------------- commands


def cmd_init_config(args: argparse.Namespace) -> int:
    target = Path(args.config)
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_config_text(), encoding="utf-8")
    print(f"Wrote {target}. Fill in your entity ids and the HA_TOKEN environment variable, then run 'hpmpc check'.")
    return 0


def cmd_ntc_table(args: argparse.Namespace) -> int:
    """Print the temperature/resistance table so it can be checked against the
    pump's service manual, plus the resolution a given resistance step buys."""
    from .ntc import resolution_check, temperature_to_resistance

    cfg = _load(args)
    temps = np.arange(args.low, args.high + 0.001, args.step)
    print(f"NTC model: {cfg.ntc.model}", end="")
    if cfg.ntc.model == "beta":
        print(f" (R25 = {cfg.ntc.r25:.0f} ohm, B = {cfg.ntc.beta:.0f})")
    else:
        print(f" ({len(cfg.ntc.table_temp_c)} table points)")
    print(f"\n{'temp (C)':>10}{'ohm':>12}{'K per ' + str(args.step_ohm) + ' ohm':>18}")
    for temp in temps:
        ohm = float(temperature_to_resistance(float(temp), cfg.ntc))
        resolution = resolution_check(cfg.ntc, float(temp), args.step_ohm)
        flag = "  <- coarse" if resolution > 0.2 else ""
        print(f"{temp:>10.1f}{ohm:>12.0f}{resolution:>18.3f}{flag}")
    print(
        "\nA step worth more than about 0.2 K makes the controller quantise noticeably.\n"
        "Size the digital potentiometer (or its series resistor) around the outdoor\n"
        "temperatures you actually see, not the whole curve."
    )
    return 0


def cmd_mode(args: argparse.Namespace) -> int:
    """Show or switch the comfort mode."""
    from .comfort import apply_mode, resolve_mode, set_mode

    cfg = _load(args)
    with _connect(cfg) as ha:
        if args.name:
            try:
                actions = set_mode(cfg, ha, args.name)
            except ValueError as exc:
                print(exc)
                return 1
            for action in actions:
                print(action)
            print(f"\nMode set to '{args.name}'. The controller applies it on its next cycle.")

        active, notes = resolve_mode(cfg, ha)
        control = apply_mode(cfg, active).control
        print(f"\nActive mode: {active}")
        print(f"  setpoint      {control.setpoint:.1f} C")
        print(f"  comfort band  {control.comfort_min:.1f} - {control.comfort_max:.1f} C")
        print(f"  hard band     {control.hard_min:.1f} - {control.hard_max:.1f} C")
        print(f"  offset range  {control.offset_min:+.1f} .. {control.offset_max:+.1f} K")
        for note in notes:
            print(f"  note: {note}")

        print(f"\n{'profile':12}{'setpoint':>10}{'comfort':>16}{'offset max':>12}")
        for name in cfg.modes.names():
            profile = apply_mode(cfg, name).control
            marker = " *" if name == active else "  "
            print(
                f"{marker}{name:10}{profile.setpoint:>10.1f}"
                f"{f'{profile.comfort_min:.1f} - {profile.comfort_max:.1f}':>16}{profile.offset_max:>12.1f}"
            )
        if cfg.modes.return_entity:
            from .comfort import read_return_time

            back = read_return_time(cfg, ha)
            print(f"\nBack home at: {back if back else 'not set'} ({cfg.modes.return_entity})")
            print(
                "The comfort band returns to normal at that moment; the optimiser decides on its\n"
                "own when to start reheating the slab, which for a deep setback is many hours."
            )
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    """Show how whole-house power is being split between the heat pump and the rest."""
    from .disaggregate import disaggregate, house_power, quality_warnings
    from .identify import _simulate_period
    from .train import load_model

    cfg = _load(args)
    path = Path(args.dataset or cfg.dataset_path)
    if not path.exists():
        print(f"No dataset at {path} - run 'hpmpc collect' first")
        return 1
    frame = load_dataset(path)
    try:
        working, params, _, _ = load_model(cfg)
    except (OSError, ValueError, KeyError):
        print("No trained model yet - run 'hpmpc train' first (the split needs the thermal model)")
        return 1

    context = _simulate_period(frame, working, params)
    if context is None:
        print("Not enough data to simulate the period")
        return 1
    result = disaggregate(
        context["data"], working, context["q_compressor"], context["q_backup"], context["cop_base"]
    )
    if result is None:
        print(
            "No whole-house power data usable for the split. Configure entities.house_power_l1/l2/l3\n"
            "(or house_power_total) and collect some history."
        )
        return 1

    metrics = result.metrics
    step_hours = (result.base_load_w.index[1] - result.base_load_w.index[0]).total_seconds() / 3600.0
    total, _ = house_power(context["data"], working)
    measured_kwh = float(total.sum() * step_hours / 1000.0)

    print(f"Split over {result.span_hours / 24:.1f} days, fitted against {result.target}\n")
    print(f"{'':28}{'kWh':>10}{'share':>9}")
    print(f"{'heat pump':28}{metrics['heatpump_kwh']:>10.1f}{metrics['heatpump_kwh'] / measured_kwh:>9.1%}")
    print(f"{'balanced other load':28}{metrics['base_load_kwh']:>10.1f}{metrics['base_load_kwh'] / measured_kwh:>9.1%}")
    print(f"{'whole house measured':28}{measured_kwh:>10.1f}")
    print()
    print(f"efficiency scale         {result.efficiency_scale:.4f}"
          + (f" +/- {100 * metrics['efficiency_scale_uncertainty']:.1f}%"
             if metrics.get("efficiency_scale_uncertainty") else ""))
    print(f"   relative to the {working.heat_pump.efficiency_scale:.4f} already in the model, so a value")
    print("   near 1.0 here means the last calibration is still right.")
    print(f"car charger inferred     {result.ev_power_w / 1000:.1f} kW "
          f"(configured nominal {working.power.ev_nominal_kw:.1f} kW)")
    print(f"charging fraction        {metrics['charging_fraction']:.1%} of samples, excluded from the fit")
    print(f"validation R2            {metrics['validation_r2']}")
    print(f"clock confounding        {metrics['clock_confounding']:.2f}  (1.0 would mean indistinguishable)")

    warnings = quality_warnings(result)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nNo warnings. The car charger figure is the check worth trusting: if an 11 kW")
        print("charger comes out near 11 kW, the phase entities and the split are both sane.")
    return 0 if not warnings else 0


def cmd_settings(args: argparse.Namespace) -> int:
    """Show what can be changed at runtime, and what it is set to now."""
    from . import settings

    cfg = _load(args)
    mapping = cfg.runtime_overrides or {}
    print(f"{'field':38}{'value':>12}{'allowed':>18}   entity")
    for path, low, high in settings.describe():
        value = settings.get_value(cfg, path)
        entity = mapping.get(path, "")
        rendered = f"{value:.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        print(f"{path:38}{rendered:>12}{f'[{low:g}, {high:g}]':>18}   {entity}")

    problems = settings.validate_mapping(cfg)
    if problems:
        print("\nProblems with runtime_overrides:")
        for problem in problems:
            print(f"  {problem}")
    print(
        "\nChange one permanently:   hpmpc set control.price_addition 0.8855\n"
        "Or map it to a helper entity under runtime_overrides: in config.yaml and\n"
        "change it from a Home Assistant dashboard; the controller picks it up on\n"
        "the next cycle."
    )
    return 1 if problems else 0


def cmd_set(args: argparse.Namespace) -> int:
    """Change one setting in config.yaml, keeping the comments intact."""
    from . import settings

    try:
        previous, new = settings.set_in_file(args.config, args.field, args.value)
    except settings.SettingError as exc:
        print(exc)
        return 1
    print(f"{args.field}: {previous} -> {new}")
    print(f"Written to {args.config}. A running controller picks it up at the next cycle.")
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    """Convert a two-point heating curve into the config's slope/offset form.

    Daikin (and most Nordic pumps) express the weather-dependent curve as two
    endpoints - "40 C leaving water at -15 C outdoor, 25 C at +15 C". This
    turns that into the linear form the model uses, and shows the result, so a
    transcription error does not silently become the single number that decides
    every heating decision.
    """
    points: list[tuple[float, float]] = []
    for raw in args.point:
        try:
            outdoor, supply = raw.split(":")
            points.append((float(outdoor), float(supply)))
        except ValueError:
            print(f"Could not parse '{raw}'; expected OUTDOOR:SUPPLY, for example --point=-15:40")
            return 1
    if len(points) != 2:
        print("Give exactly two points, the cold end and the warm end of the curve.")
        return 1
    (t1, s1), (t2, s2) = sorted(points)
    if t1 == t2:
        print("The two points must be at different outdoor temperatures.")
        return 1

    reference = args.reference
    slope = (s1 - s2) / (t2 - t1)
    offset = s1 - slope * (reference - t1)
    print(f"curve_slope: {slope:.3f}")
    print(f"curve_offset: {offset:.2f}")
    print(f"curve_ref: {reference:.1f}")
    print(f"\n  supply = {offset:.2f} + {slope:.3f} * ({reference:.0f} - outdoor_filtered)\n")
    print(f"{'outdoor':>9}{'supply':>9}")
    for outdoor in range(int(min(t1, -20)), int(max(t2, 15)) + 1, 5):
        print(f"{outdoor:>9}{offset + slope * (reference - outdoor):>9.1f}")
    print(
        "\nCheck the two endpoints against your pump's display before pasting this in.\n"
        "Cap heat_pump.supply_max at what your floor loop is designed for."
    )
    return 0


def cmd_geocode(args: argparse.Namespace) -> int:
    """Look up coordinates for an address, once, so you can paste them in."""
    from .providers import geocode

    results = geocode(args.address, limit=args.limit)
    if not results:
        print(f"No match for '{args.address}'. Try adding the municipality and country.")
        return 1
    print(f"Matches for '{args.address}':\n")
    for i, hit in enumerate(results, 1):
        print(f"  {i}. {hit['display_name']}")
        print(f"     latitude: {hit['latitude']}   longitude: {hit['longitude']}   ({hit['type']})")
    best = results[0]
    print("\nPaste into config.yaml:\n")
    print("site:")
    print(f"  address: {args.address}")
    print(f"  latitude: {best['latitude']}")
    print(f"  longitude: {best['longitude']}")
    print(
        "\nSMHI's forecast grid is about 2.5 km, so anywhere in the same town gives\n"
        "the same numbers - do not agonise over the exact house."
    )
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    """Verify that the weather and price sources actually answer from here."""
    from .providers import fetch_forecast, fetch_prices
    from .providers._http import ProviderError

    cfg = _load(args)
    ok = True

    print(f"Weather source: {cfg.forecast.weather_source}")
    if cfg.forecast.weather_source == "smhi":
        try:
            frame, meta = fetch_forecast(
                cfg.site.latitude, cfg.site.longitude,
                cache_dir=None if args.no_cache else cfg.cache_dir,
                cache_minutes=cfg.forecast.weather_cache_minutes,
                timeout=cfg.forecast.timeout,
            )
            span = (frame.index[-1] - frame.index[0]).total_seconds() / 3600.0
            print(f"  ok       {len(frame)} points over {span:.0f} h, cache: {meta.get('cache')}")
            print(f"           reference time {meta.get('reference_time')}")
            head = frame.head(6)
            print(f"\n{'time (UTC)':22}{'temp':>7}{'wind':>7}{'cloud':>8}{'humidity':>10}")
            for stamp, row in head.iterrows():
                print(
                    f"{stamp.strftime('%Y-%m-%d %H:%M'):22}{row['t_outdoor']:>7.1f}{row['wind']:>7.1f}"
                    f"{row['cloud']:>8.0f}{row['humidity']:>10.0f}"
                )
        except (ProviderError, ValueError) as exc:
            print(f"  FAILED   {exc}")
            ok = False
    else:
        print("  (using the Home Assistant weather entity; run 'hpmpc check' instead)")

    print(f"\nPrice source: {cfg.forecast.price_source} ({cfg.forecast.price_area})")
    if cfg.forecast.price_source == "elprisetjustnu":
        try:
            points, meta = fetch_prices(
                area=cfg.forecast.price_area,
                timezone_name=cfg.site.timezone,
                cache_dir=None if args.no_cache else cfg.cache_dir,
                cache_minutes=cfg.forecast.price_cache_minutes,
                timeout=cfg.forecast.timeout,
            )
            print(f"  ok       {len(points)} hourly prices, {meta['first']} .. {meta['last']}")
            for day in meta.get("days", []):
                print(f"           {day['date']}: {day['hours']} h{' (cached)' if day['cached'] else ''}")
            if meta.get("tomorrow_published") is False:
                print("           tomorrow not published yet (Nord Pool publishes just after 13:00)")
            if meta.get("warning"):
                print(f"  WARNING  {meta['warning']}")
            values = np.array([v for _, v in points])
            print(
                f"           spot excl. VAT/fees: min {values.min():.3f}  max {values.max():.3f}  "
                f"mean {values.mean():.3f} SEK/kWh"
            )
            marginal_min = (values.min() * cfg.control.price_scale + cfg.control.price_addition) * (
                1 + cfg.control.price_vat_pct / 100
            )
            marginal_max = (values.max() * cfg.control.price_scale + cfg.control.price_addition) * (
                1 + cfg.control.price_vat_pct / 100
            )
            print(f"           your marginal cost:  min {marginal_min:.3f}  max {marginal_max:.3f} SEK/kWh")
            if cfg.control.price_addition == 0 and cfg.control.price_vat_pct == 0:
                print(
                    "  WARNING  price_addition and price_vat_pct are both zero, so the optimiser is\n"
                    "           planning against bare spot. Add your grid transfer, energy tax and VAT,\n"
                    "           or it will overestimate what load shifting is worth."
                )
            print(f"           {meta['attribution']}")
        except (ProviderError, ValueError) as exc:
            print(f"  FAILED   {exc}")
            ok = False
    else:
        print("  (using the Home Assistant price entity; run 'hpmpc check' instead)")

    print("\nAll providers reachable." if ok else "\nSome providers failed; see above.")
    return 0 if ok else 1


def cmd_pump_table(args: argparse.Namespace) -> int:
    """Print what the loaded performance map implies, to check against the databook."""
    from .model import build_pump

    cfg = _load(args)
    pump = build_pump(cfg)
    if pump.performance is None:
        print(
            "No performance map loaded (heat_pump.model is empty), so the generic Carnot model is in\n"
            f"use with efficiency {cfg.heat_pump.carnot_efficiency}. That model cannot see the electric\n"
            "backup heater. Set heat_pump.model to your unit to fix that."
        )
        return 1
    performance = pump.performance
    print(f"{performance.name}")
    print(f"source: {performance.source}")
    print(f"efficiency scale {performance.efficiency_scale:.4f} (fitted against your meter by 'hpmpc train')")
    print(f"backup heater: {'enabled' if performance.backup_enabled else 'disabled'}, "
          f"{performance.backup_max_kw} kW at COP {performance.backup_cop}")
    print(f"compressor stops below {performance.min_ambient_c} degC ambient\n")

    supply = args.supply or [30.0, 35.0, 40.0, 45.0, 50.0]
    ambient = args.ambient or [-20.0, -15.0, -10.0, -7.0, -2.0, 2.0, 7.0, 12.0]

    print("COP")
    print("  ambient  " + "".join(f"{f'W{int(t)}':>8}" for t in supply))
    for ta in ambient:
        row = "".join(f"{float(pump.cop(np.array(ts), np.array(ta))):>8.2f}" for ts in supply)
        print(f"  {ta:>7.0f}  {row}")

    print("\nCompressor capacity (kW)")
    print("  ambient  " + "".join(f"{f'W{int(t)}':>8}" for t in supply))
    for ta in ambient:
        row = "".join(f"{float(pump.capacity_w(np.array(ta), np.array(ts))) / 1000:>8.1f}" for ts in supply)
        print(f"  {ta:>7.0f}  {row}")

    print(
        "\nCompare against the 'heating capacity tables' in your unit's databook.\n"
        "  efficiency = COP * (T_supply + 273.15) / (T_supply - T_ambient)\n"
        "Edit the table with heat_pump.model pointing at your own YAML file if the shape is off;\n"
        "the level is corrected automatically from your power sensor during training."
    )
    return 0


def cmd_calibrate_ntc(args: argparse.Namespace) -> int:
    """Fit an NTC model to resistances you measured on your own sensor."""
    from .ntc import resistance_to_temperature, temperature_to_resistance

    points: list[tuple[float, float]] = []
    for raw in args.point:
        try:
            temp, ohm = raw.split(":")
            points.append((float(temp), float(ohm)))
        except ValueError:
            print(f"Could not parse '{raw}'; expected TEMP:OHM, for example 0:66800")
            return 1
    if len(points) < 2:
        print("Need at least two measurements at different temperatures.")
        return 1

    temps = np.array([t for t, _ in points], dtype=float)
    ohms = np.array([r for _, r in points], dtype=float)
    # ln R = ln R25 + B (1/T - 1/298.15): linear in (1/T), so a plain least squares.
    x = 1.0 / (temps + 273.15) - 1.0 / 298.15
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, np.log(ohms), rcond=None)
    r25 = float(np.exp(coefficients[0]))
    beta = float(coefficients[1])

    from .config import NTCConfig

    fitted = NTCConfig(model="beta", r25=r25, beta=beta)
    print(f"Fitted: R25 = {r25:.0f} ohm, B = {beta:.0f}\n")
    print(f"{'measured T':>12}{'measured R':>12}{'model R':>12}{'error':>10}")
    for temp, ohm in points:
        modelled = float(temperature_to_resistance(temp, fitted))
        print(f"{temp:>12.1f}{ohm:>12.0f}{modelled:>12.0f}{100 * (modelled / ohm - 1):>9.1f}%")
    worst = max(
        abs(float(resistance_to_temperature(r, fitted)) - t) for t, r in points
    )
    print(f"\nWorst temperature error at the measured points: {worst:.2f} K")
    if len(points) < 4 or worst > 0.5:
        print(
            "A two-parameter beta model drifts at the ends of the range. With four or more\n"
            "measurements, or the table from the service manual, prefer ntc.model: table."
        )
    print("\nPaste into config.yaml:\n")
    print("ntc:")
    print("  model: beta")
    print(f"  r25: {r25:.0f}")
    print(f"  beta: {beta:.0f}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(f"Config OK: {args.config}")
    ha = HomeAssistant(cfg.home_assistant)
    if not ha.ping():
        print(f"FAIL: no response from {cfg.home_assistant.base_url}")
        return 1
    print(f"Home Assistant reachable at {cfg.home_assistant.base_url}")

    ok = True
    for name, entity_id in vars(cfg.entities).items():
        if not entity_id or name == "extra":
            continue
        state = ha.get_state(entity_id)
        if state is None:
            print(f"  MISSING  {name:16} {entity_id}")
            ok = False
            continue
        age = state.age()
        age_txt = f"{age.total_seconds() / 60:.0f} min ago" if age else "unknown age"
        print(f"  ok       {name:16} {entity_id} = {state.state} ({age_txt})")

    if cfg.entities.offset_output:
        domain = cfg.entities.offset_output.split(".", 1)[0]
        if domain not in {"number", "input_number"}:
            print(f"  WARNING  offset_output must be a number/input_number entity, got '{domain}'")
            ok = False
    else:
        print("  WARNING  no offset_output entity configured - the controller will run read-only")

    if cfg.entities.weather:
        forecast = ha.weather_forecast(cfg.entities.weather)
        print(f"  {'ok' if forecast else 'WARNING'}       weather forecast: {len(forecast)} hourly entries")

    try:
        _, params, residual, metadata = load_model(cfg)
        print(f"Model:     trained {metadata.get('trained_at', 'unknown')}, "
              f"UA {params.heat_loss_w_per_k():.0f} W/K, residual model: {'yes' if residual else 'no'}")
    except (OSError, ValueError, KeyError):
        print("Model:     not trained yet - run 'hpmpc collect' then 'hpmpc train'")

    print("\nAll checks passed." if ok else "\nSome checks failed; see above.")
    return 0 if ok else 1


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = _load(args)
    with _connect(cfg) as ha:
        frame = build_dataset(cfg, ha, args.days)
    path = Path(args.output or cfg.dataset_path)
    save_dataset(frame, path)
    info = describe(frame)
    print(f"Saved {len(frame)} rows to {path}")
    print(json.dumps(info, indent=2))
    excitation = info.get("offset_excitation", {})
    if excitation and excitation.get("std", 0.0) < 0.5:
        print(
            "\nWARNING: the commanded offset barely varies in this data. The fit will struggle to\n"
            "learn how strongly the offset moves the house. Run 'hpmpc excite' for about a week\n"
            "during the heating season, then collect and train again."
        )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load(args)
    path = Path(args.dataset or cfg.dataset_path)
    if not path.exists():
        print(f"No dataset at {path} - run 'hpmpc collect' first")
        return 1
    frame = load_dataset(path)
    if args.days:
        frame = frame.iloc[-int(args.days * 24 * 60 / cfg.training.resample_minutes) :]
    report = train(cfg, frame, fit_curve=not args.no_curve_fit)
    write_report(Path(cfg.paths.model_dir) / "training_report.json", report)
    print()
    print(summarise(report))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = _load(args)
    working, params, residual, _ = load_model(cfg)
    with _connect(working) as ha:
        controller = Controller(working, params, ha, residual)
        report = controller.step(apply=False)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "plan"}, indent=2, default=str))
    _print_plan(report)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    working, params, residual, _ = load_model(cfg)
    apply = not (args.dry_run or cfg.control.dry_run)
    with _connect(working) as ha:
        controller = Controller(working, params, ha, residual)
        controller.reload_config(args.config)   # record the current mtime
        while True:
            started = datetime.now(timezone.utc)
            try:
                controller.reload_config(args.config)
                if args.excite:
                    report = controller.excite_step(apply=apply, hold_hours=args.hold_hours)
                else:
                    report = controller.step(apply=apply)
                _print_cycle(report)
            except (HomeAssistantError, ValueError) as exc:
                log.error("Control cycle failed: %s", exc)
            if args.once:
                return 0
            _sleep_until_next_cycle(started, cfg.control.cycle_minutes)


def cmd_excite(args: argparse.Namespace) -> int:
    args.excite = True
    return cmd_run(args)


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load(args)
    working, params, residual, _ = load_model(cfg)
    path = Path(args.dataset or cfg.dataset_path)
    if not path.exists():
        print(f"No dataset at {path} - run 'hpmpc collect' first")
        return 1
    frame = load_dataset(path)
    result = backtest(working, params, frame, days=args.days, residual=residual)
    print(format_backtest(result))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    cfg = _load(args)
    app = create_app(args.config, run_scheduler=not args.no_scheduler)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the entire pipeline against a synthetic house - no Home Assistant."""
    from .simulator import make_demo_dataset

    cfg = Config()
    cfg.entities.indoor_temp = "sensor.demo_indoor"
    cfg.entities.outdoor_temp = "sensor.demo_outdoor"
    cfg.entities.price = "sensor.demo_price"
    cfg.heat_pump.model = args.pump
    cfg.paths.data_dir = args.workdir
    cfg.paths.model_dir = args.workdir
    cfg.paths.state_file = str(Path(args.workdir) / "controller_state.json")
    cfg.validate()

    print(f"1/4  Generating {args.days} days of synthetic history ...")
    frame, truth = make_demo_dataset(cfg, days=args.days, seed=args.seed)
    save_dataset(frame, cfg.dataset_path)

    print("2/4  Identifying the building from that history ...")
    report = train(cfg, frame)
    print()
    print(summarise(report))

    print("\n3/4  Recovered parameters vs ground truth")
    fitted = report["parameters"]
    print(f"     {'parameter':10}{'true':>12}{'fitted':>12}{'error':>10}")
    for name, true_value in truth.to_dict().items():
        got = fitted[name]
        err = 100.0 * (got - true_value) / true_value if true_value else 0.0
        print(f"     {name:10}{true_value:>12.1f}{got:>12.1f}{err:>9.0f}%")
    print(f"     {'UA (W/K)':10}{truth.heat_loss_w_per_k():>12.1f}{report['thermal']['heat_loss_w_per_k']:>12.1f}")

    print("\n4/4  Backtesting the controller ...")
    working, params, residual, _ = load_model(cfg)
    result = backtest(working, params, frame, days=args.backtest_days, residual=residual)
    print()
    print(format_backtest(result))
    print(f"\nArtifacts written to {args.workdir}/")
    return 0


# ---------------------------------------------------------------- helpers


def _print_plan(report: dict[str, Any]) -> None:
    mpc = report.get("mpc")
    if not mpc:
        print(json.dumps(report, indent=2, default=str))
        return
    out = report.get("output", {})
    print(
        f"\nOffset now: {report['offset']:+.2f} K  ->  {out.get('entity_id', '(no entity)')} "
        f"= {out.get('value')} {out.get('unit')}   [{report.get('mode')}]"
    )
    print(
        f"Horizon: {mpc['horizon_kwh']} kWh / {mpc['horizon_cost_sek']} SEK; "
        f"indoor {mpc['predicted_indoor_min']}-{mpc['predicted_indoor_max']} C "
        f"(mean {mpc['predicted_indoor_mean']})"
    )
    print(
        f"Saving vs constant offset {mpc['baseline_matched']['offset']:+.2f} K at the same average "
        f"indoor temperature: {mpc['predicted_saving_sek']} SEK ({mpc['predicted_saving_pct']} %)"
    )
    for note in report.get("notes", []):
        print(f"  note: {note}")
    rows = report.get("plan", [])
    if rows:
        print(f"\n{'time':17}{'price':>7}{'out':>7}{'offset':>8}{'supply':>8}{'indoor':>8}{'kW':>7}")
        for row in rows[:36]:
            stamp = row["time"][:16].replace("T", " ")
            print(
                f"{stamp:17}{row['price']:>7.2f}{row['t_outdoor']:>7.1f}{row['offset']:>8.2f}"
                f"{row['t_supply']:>8.1f}{row['t_indoor']:>8.2f}{row['kw']:>7.2f}"
            )


def _print_cycle(report: dict[str, Any]) -> None:
    mpc = report.get("mpc", {})
    out = report.get("output", {})
    stamp = report["timestamp"][11:16]
    extra = ""
    if mpc:
        extra = (
            f" | indoor now {report.get('readings', {}).get('t_indoor')} C"
            f" -> min {mpc.get('predicted_indoor_min')} C"
            f" | {mpc.get('horizon_cost_sek')} SEK/horizon"
            f" | saves {mpc.get('predicted_saving_pct')} %"
        )
    print(
        f"[{stamp}] {report.get('mode'):18} offset {report.get('offset', 0.0):+5.2f} K"
        f" -> {out.get('value')} {out.get('unit')}"
        f" {'(written)' if report.get('applied') else '(not written)'}{extra}"
    )
    for note in report.get("notes", []):
        print(f"          note: {note}")


def _sleep_until_next_cycle(started: datetime, cycle_minutes: int) -> None:
    period = timedelta(minutes=cycle_minutes)
    epoch = started.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (started - epoch) % period
    target = started + (period - elapsed)
    delay = max(1.0, (target - datetime.now(timezone.utc)).total_seconds())
    time.sleep(delay)


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpmpc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default=os.environ.get("HPMPC_CONFIG", "config/config.yaml"),
        help="path to config.yaml (default: $HPMPC_CONFIG, then config/config.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"hpmpc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-config", help="write a starting config file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("check", help="validate config and Home Assistant connectivity")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("mode", help="show or switch the comfort mode (normal / away / holiday)")
    p.add_argument("name", nargs="?", default=None)
    p.set_defaults(func=cmd_mode)

    p = sub.add_parser("power", help="show how whole-house power is split between the pump and the rest")
    p.add_argument("--dataset", default=None)
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("settings", help="show the settings you can change at runtime")
    p.set_defaults(func=cmd_settings)

    p = sub.add_parser("set", help="change one setting in config.yaml")
    p.add_argument("field", help="for example control.price_addition")
    p.add_argument("value")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("curve", help="convert a two-point heating curve to slope/offset")
    p.add_argument("--point", action="append", required=True, metavar="OUTDOOR:SUPPLY",
                   help="a curve endpoint. Negative temperatures need the equals form: --point=-15:40")
    p.add_argument("--reference", type=float, default=20.0)
    p.set_defaults(func=cmd_curve)

    p = sub.add_parser("geocode", help="look up coordinates for an address")
    p.add_argument("address")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_geocode)

    p = sub.add_parser("providers", help="verify the weather and price sources from this machine")
    p.add_argument("--no-cache", action="store_true", help="bypass the local cache and go to the network")
    p.set_defaults(func=cmd_providers)

    p = sub.add_parser("pump-table", help="print the loaded COP and capacity tables")
    p.add_argument("--ambient", type=float, nargs="*", default=None)
    p.add_argument("--supply", type=float, nargs="*", default=None)
    p.set_defaults(func=cmd_pump_table)

    p = sub.add_parser("calibrate-ntc", help="fit an NTC model to your own measurements")
    p.add_argument("--point", action="append", required=True, metavar="TEMP:OHM",
                   help="a measured pair, e.g. --point 0:66800. Negative temperatures need "
                        "the equals form: --point=-20:197000")
    p.set_defaults(func=cmd_calibrate_ntc)

    p = sub.add_parser("ntc-table", help="print the NTC curve and the resolution of a resistance step")
    p.add_argument("--low", type=float, default=-25.0)
    p.add_argument("--high", type=float, default=20.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--step-ohm", type=float, default=78.0, help="one step of your digital potentiometer")
    p.set_defaults(func=cmd_ntc_table)

    p = sub.add_parser("collect", help="download recorder history into a dataset")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("train", help="fit the model from a dataset")
    p.add_argument("--dataset", default=None)
    p.add_argument("--days", type=float, default=None, help="use only the most recent N days")
    p.add_argument("--no-curve-fit", action="store_true", help="keep the configured heating curve")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("plan", help="solve once and print the plan without writing anything")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run the control loop")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--excite", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--hold-hours", type=float, default=6.0, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("excite", help="run the identification experiment instead of the controller")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hold-hours", type=float, default=6.0)
    p.set_defaults(func=cmd_excite)

    p = sub.add_parser("backtest", help="replay history: MPC vs constant offset")
    p.add_argument("--dataset", default=None)
    p.add_argument("--days", type=float, default=7.0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("serve", help="run the control loop with a local HTTP API")
    p.add_argument("--no-scheduler", action="store_true", help="serve the API only, do not control")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="run the whole pipeline on a synthetic house (no Home Assistant)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--backtest-days", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pump", default="daikin_erlq016caw1",
                   help="bundled performance map to use; empty string for the generic Carnot model")
    p.add_argument("--workdir", default="demo_output")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    np.seterr(all="ignore")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (HomeAssistantError, ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
