"""Local HTTP API.

Runs the control loop in a background thread and exposes what it is doing, so
Home Assistant (or a browser, or curl) can see the current plan, the predicted
saving and the model's health. Binds to the LAN only; nothing leaves the house.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from . import __version__
from .config import Config, load_config
from .controller import Controller
from .archive import build_training_frame
from .dataset import save_dataset
from .ha import HomeAssistant, HomeAssistantError
from .train import load_model, train

log = logging.getLogger(__name__)


class ControllerService:
    """Owns the controller and serialises access to it."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = str(config_path)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_report: dict[str, Any] = {}
        self.last_error: str | None = None
        self.last_training: dict[str, Any] | None = None
        self.started_at = datetime.now(timezone.utc)
        self.cycles = 0
        self.cfg: Config = load_config(self.config_path)
        working, params, residual, self.model_metadata = load_model(self.cfg)
        self.cfg = working
        self.ha = HomeAssistant(self.cfg.home_assistant)
        self.controller = Controller(self.cfg, params, self.ha, residual)
        self.controller.reload_config(self.config_path)

    def step(self, apply: bool | None = None) -> dict[str, Any]:
        with self.lock:
            if self.controller.reload_config(self.config_path):
                self.cfg = self.controller.cfg
            try:
                report = self.controller.step(apply=apply)
                self.last_error = None
            except (HomeAssistantError, ValueError) as exc:
                self.last_error = str(exc)
                log.error("Control cycle failed: %s", exc)
                raise
            if apply is not False:
                self.last_report = report
                self.cycles += 1
            return report

    def model_age_days(self) -> float | None:
        try:
            stamp = Path(self.cfg.model_path).stat().st_mtime
        except OSError:
            return None
        return (datetime.now(timezone.utc).timestamp() - stamp) / 86400.0

    def maybe_retrain(self) -> None:
        """Refit the model when it gets stale, without needing a cron job.

        A house is not the same house in March as it was in November - leaves,
        snow cover, how it is actually lived in. Retraining takes a minute or
        two of one core, so it waits for a quiet hour rather than interrupting
        an evening.
        """
        days = self.cfg.training.retrain_days
        if days <= 0:
            return
        age = self.model_age_days()
        if age is None or age < days:
            return
        local_hour = datetime.now(ZoneInfo(self.cfg.site.timezone)).hour
        if local_hour != self.cfg.training.retrain_hour:
            return
        log.info("Model is %.0f days old; retraining", age)
        try:
            with self.lock:
                frame, _ = build_training_frame(self.cfg, self.ha)
                save_dataset(frame, self.cfg.dataset_path)
                report = train(self.cfg, frame)
                working, params, residual, metadata = load_model(self.cfg)
                self.cfg = working
                self.controller.base_cfg = working
                self.controller.cfg = working
                self.controller.params = params
                self.controller.residual = residual
                self.controller.solver.params = params
                self.model_metadata = metadata
            self.last_training = {
                "at": datetime.now(timezone.utc).isoformat(),
                "validation_rmse_c": report.get("thermal", {}).get("validation", {}).get("rmse_c"),
            }
            log.info("Retrained: validation RMSE %s C", self.last_training["validation_rmse_c"])
        except Exception as exc:  # pragma: no cover - never take control down for this
            log.error("Automatic retraining failed (%s); continuing on the previous model", exc)
            self.last_training = {"at": datetime.now(timezone.utc).isoformat(), "error": str(exc)}

    def loop(self) -> None:
        cycle = timedelta(minutes=self.cfg.control.cycle_minutes)
        while not self.stop_event.is_set():
            try:
                self.step()
                self.maybe_retrain()
            except Exception as exc:  # pragma: no cover - keep the loop alive
                log.error("Scheduler cycle failed: %s", exc)
            now = datetime.now(timezone.utc)
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elapsed = (now - midnight) % cycle
            self.stop_event.wait(max(1.0, (cycle - elapsed).total_seconds()))

    def close(self) -> None:
        self.stop_event.set()
        self.ha.close()


def create_app(config_path: str | Path = "config/config.yaml", run_scheduler: bool = True) -> FastAPI:
    service = ControllerService(config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if run_scheduler:
            thread = threading.Thread(target=service.loop, name="hpmpc-control", daemon=True)
            thread.start()
            log.info("Control loop started (every %d min)", service.cfg.control.cycle_minutes)
        yield
        service.close()

    app = FastAPI(title="Heat pump MPC", version=__version__, lifespan=lifespan)
    app.state.service = service

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = service.cfg.server.api_key
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "error" if service.last_error else "ok",
            "version": __version__,
            "uptime_s": round((datetime.now(timezone.utc) - service.started_at).total_seconds()),
            "cycles": service.cycles,
            "last_error": service.last_error,
            "last_cycle": service.last_report.get("timestamp"),
            "scheduler": run_scheduler,
            "model_age_days": (
                round(service.model_age_days(), 1) if service.model_age_days() is not None else None
            ),
            "last_training": service.last_training,
        }

    @app.get("/status", dependencies=[Depends(require_key)])
    def status() -> dict[str, Any]:
        return {
            "report": {k: v for k, v in service.last_report.items() if k != "plan"},
            "state": service.controller.state.__dict__,
        }

    @app.get("/plan", dependencies=[Depends(require_key)])
    def plan() -> dict[str, Any]:
        return service.step(apply=False)

    @app.post("/step", dependencies=[Depends(require_key)])
    def step() -> dict[str, Any]:
        return service.step()

    @app.get("/model", dependencies=[Depends(require_key)])
    def model() -> dict[str, Any]:
        return {
            "parameters": service.controller.params.to_dict(),
            "time_constants_hours": service.controller.params.time_constants_hours(),
            "heat_loss_w_per_k": service.controller.params.heat_loss_w_per_k(),
            "residual_model": bool(service.controller.residual),
            "metadata": service.model_metadata,
        }

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_key)])
    def metrics() -> str:
        """Prometheus-style metrics, for anyone who already scrapes their house."""
        report = service.last_report
        mpc = report.get("mpc", {})
        rows = {
            "hpmpc_offset_kelvin": report.get("offset", 0.0),
            "hpmpc_applied": 1 if report.get("applied") else 0,
            "hpmpc_slab_temperature_celsius": service.controller.state.t_mass,
            "hpmpc_indoor_temperature_celsius": service.controller.state.t_indoor,
            "hpmpc_horizon_cost_sek": mpc.get("horizon_cost_sek", 0.0),
            "hpmpc_horizon_energy_kwh": mpc.get("horizon_kwh", 0.0),
            "hpmpc_predicted_saving_sek": mpc.get("predicted_saving_sek", 0.0),
            "hpmpc_predicted_saving_pct": mpc.get("predicted_saving_pct", 0.0),
            "hpmpc_cycles_total": service.cycles,
        }
        return "\n".join(f"{k} {float(v)}" for k, v in rows.items()) + "\n"

    return app
