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
        while True:
            started = datetime.now(timezone.utc)
            try:
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
    parser.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"hpmpc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-config", help="write a starting config file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("check", help="validate config and Home Assistant connectivity")
    p.set_defaults(func=cmd_check)

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
