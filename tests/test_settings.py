"""Runtime settings: Home Assistant helpers and the config-file editor."""

from __future__ import annotations

import pytest

from hpmpc import settings
from hpmpc.config import Config, load_config


def test_every_overridable_field_actually_exists(cfg):
    for path, _, _ in settings.describe():
        assert settings.get_value(cfg, path) is not None or path in settings.BOOLEAN_FIELDS


def test_values_within_range_are_applied(cfg):
    updated, notes = settings.apply(cfg, {"control.price_addition": 0.8855})
    assert updated.control.price_addition == pytest.approx(0.8855)
    assert notes == ["control.price_addition = 0.8855"]


def test_values_outside_range_are_rejected_individually(cfg):
    updated, notes = settings.apply(cfg, {"control.setpoint": 99.0, "control.price_addition": 0.9})
    assert updated.control.setpoint == cfg.control.setpoint     # unchanged
    assert updated.control.price_addition == pytest.approx(0.9)  # the good one still lands
    assert any("outside the allowed range" in n for n in notes)


def test_a_change_that_breaks_consistency_is_rolled_back_whole(cfg):
    before = cfg.control.comfort_below
    # A comfort band wider than the hard band is contradictory.
    updated, notes = settings.apply(cfg, {"control.comfort_below": 5.0})
    assert updated.control.comfort_below == before
    assert any("rejected" in n for n in notes)


def test_moving_the_setpoint_moves_the_whole_band(cfg):
    updated, _ = settings.apply(cfg, {"control.setpoint": 16.0})
    assert updated.control.comfort_min == pytest.approx(15.3)
    assert updated.control.comfort_max == pytest.approx(17.0)
    assert updated.control.hard_min == pytest.approx(14.0)


def test_unknown_fields_are_refused_with_the_valid_list(cfg):
    _, notes = settings.apply(cfg, {"control.horizon_hours": 48})
    assert any("not changeable at runtime" in n for n in notes)


def test_booleans_come_from_switch_states(cfg, fake_ha):
    cfg.runtime_overrides = {"control.dry_run": "input_boolean.hpmpc_dry_run"}
    fake_ha.set("input_boolean.hpmpc_dry_run", "on")
    values = settings.read_from_home_assistant(cfg, fake_ha)
    assert values == {"control.dry_run": True}
    updated, _ = settings.apply(cfg, values)
    assert updated.control.dry_run is True


def test_numbers_come_from_helper_entities(cfg, fake_ha):
    cfg.runtime_overrides = {"control.setpoint": "input_number.borvarde"}
    fake_ha.set("input_number.borvarde", 22.5)
    assert settings.read_from_home_assistant(cfg, fake_ha) == {"control.setpoint": 22.5}


def test_a_missing_or_junk_entity_is_skipped_not_fatal(cfg, fake_ha):
    cfg.runtime_overrides = {
        "control.setpoint": "input_number.absent",
        "control.price_addition": "input_number.junk",
    }
    fake_ha.set("input_number.junk", "unavailable")
    assert settings.read_from_home_assistant(cfg, fake_ha) == {}


def test_mapping_typos_are_reported(cfg):
    cfg.runtime_overrides = {"control.nonsense": "input_number.x", "control.setpoint": "oops"}
    problems = settings.validate_mapping(cfg)
    assert len(problems) == 2


# ------------------------------------------------------- editing the file


@pytest.fixture
def written_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "token")
    from hpmpc.cli import example_config_text

    path = tmp_path / "config.yaml"
    path.write_text(example_config_text(), encoding="utf-8")
    return path


def test_set_in_file_changes_the_value(written_config):
    previous, new = settings.set_in_file(written_config, "control.price_addition", "0.95")
    assert new == pytest.approx(0.95)
    assert load_config(written_config).control.price_addition == pytest.approx(0.95)
    assert previous == pytest.approx(0.7084)      # the shipped default, excluding VAT


def test_set_in_file_keeps_comments_and_the_rest_of_the_document(written_config):
    before = written_config.read_text(encoding="utf-8").splitlines()
    settings.set_in_file(written_config, "control.setpoint", "21.5")
    after = written_config.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert after[differing[0]].strip().startswith("setpoint: 21.5")
    assert "#" in "\n".join(after)          # the commentary survived


def test_set_in_file_preserves_a_trailing_comment(written_config):
    settings.set_in_file(written_config, "control.price_addition", "0.9")
    line = next(
        l for l in written_config.read_text(encoding="utf-8").splitlines()
        if l.strip().startswith("price_addition:")
    )
    assert "#" in line


def test_set_in_file_refuses_an_out_of_range_value(written_config):
    with pytest.raises(settings.SettingError, match="outside the allowed range"):
        settings.set_in_file(written_config, "control.setpoint", "99")


def test_set_in_file_refuses_a_field_it_cannot_find(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "token")
    path = tmp_path / "c.yaml"
    path.write_text("entities:\n  indoor_temp: sensor.a\n  outdoor_temp: sensor.b\n", encoding="utf-8")
    with pytest.raises(settings.SettingError, match="Could not find"):
        settings.set_in_file(path, "control.setpoint", "21")


def test_set_in_file_leaves_no_backup_behind(written_config):
    settings.set_in_file(written_config, "control.setpoint", "21.5")
    assert list(written_config.parent.glob("*.bak")) == []


# ----------------------------------------------------- through the controller


def test_controller_picks_up_changed_helper_entities(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.runtime_overrides = {"control.price_addition": "input_number.transfer"}
    fake_ha.set("input_number.transfer", 0.8855)
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.price_addition == pytest.approx(0.8855)
    assert any("price_addition" in note for note in report["settings"])
    # And the solver got the change, not just the controller.
    assert controller.solver.cfg.control.price_addition == pytest.approx(0.8855)


def test_controller_survives_a_helper_entity_going_rogue(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.runtime_overrides = {"control.setpoint": "input_number.borvarde"}
    fake_ha.set("input_number.borvarde", 500.0)
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.setpoint == pytest.approx(Config().control.setpoint)
    assert report["mode"] == "mpc"           # still controls, just ignores the bad value


def test_editing_the_config_file_is_picked_up_without_a_restart(
    cfg, fake_ha, written_config, tmp_path, monkeypatch
):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams, save_params

    # The shipped config reads its paths from the environment, which is how the
    # Docker image points them at its volume without editing the file.
    monkeypatch.setenv("HPMPC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HPMPC_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HPMPC_STATE_FILE", str(tmp_path / "state.json"))
    save_params(tmp_path / "thermal_model.json", ThermalParams(), {})

    controller = Controller(cfg, ThermalParams(), fake_ha)
    assert controller.reload_config(written_config) is False    # first call only records the time
    settings.set_in_file(written_config, "control.setpoint", "22.0")
    assert controller.reload_config(written_config) is True
    assert controller.cfg.control.setpoint == pytest.approx(22.0)


# ------------------------------------------------------------------ modes


def test_holiday_switch_wins_over_the_selector(cfg, fake_ha):
    from hpmpc.comfort import resolve_mode

    cfg.modes.entity = "input_select.lage"
    cfg.modes.holiday_entity = "input_boolean.semester"
    fake_ha.set("input_select.lage", "normal")
    fake_ha.set("input_boolean.semester", "on")
    mode, _ = resolve_mode(cfg, fake_ha)
    assert mode == "holiday"


def test_selector_chooses_a_profile(cfg, fake_ha):
    from hpmpc.comfort import resolve_mode

    cfg.modes.entity = "input_select.lage"
    fake_ha.set("input_select.lage", "Away")
    assert resolve_mode(cfg, fake_ha)[0] == "away"


def test_an_unknown_mode_falls_back_with_a_note(cfg, fake_ha):
    from hpmpc.comfort import resolve_mode

    cfg.modes.entity = "input_select.lage"
    fake_ha.set("input_select.lage", "party")
    mode, notes = resolve_mode(cfg, fake_ha)
    assert mode == cfg.modes.default
    assert any("not a defined profile" in n for n in notes)


def test_controller_enters_and_leaves_holiday_mode(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.modes.holiday_entity = "input_boolean.semester"
    fake_ha.set("input_boolean.semester", "off")
    controller = Controller(cfg, ThermalParams(), fake_ha)

    controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.setpoint == pytest.approx(21.0)

    fake_ha.set("input_boolean.semester", "on")
    report = controller.step(now=fake_ha.now, apply=False)
    assert controller.mode == "holiday"
    assert controller.cfg.control.setpoint == pytest.approx(16.0)
    assert report["comfort"]["comfort_band_now"] == [14.5, 21.0]

    # And back again - the mode must not be sticky.
    fake_ha.set("input_boolean.semester", "off")
    controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.setpoint == pytest.approx(21.0)


def test_holiday_mode_ignores_the_setpoint_helper(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.modes.holiday_entity = "input_boolean.semester"
    cfg.runtime_overrides = {"control.setpoint": "input_number.borvarde"}
    fake_ha.set("input_number.borvarde", 22.5)
    fake_ha.set("input_boolean.semester", "on")
    controller = Controller(cfg, ThermalParams(), fake_ha)
    report = controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.setpoint == pytest.approx(16.0)
    assert any("overrides control.setpoint" in note for note in report["settings"])


def test_the_setpoint_helper_still_works_in_the_default_mode(cfg, fake_ha):
    from hpmpc.controller import Controller
    from hpmpc.model.thermal import ThermalParams

    cfg.runtime_overrides = {"control.setpoint": "input_number.borvarde"}
    fake_ha.set("input_number.borvarde", 22.5)
    controller = Controller(cfg, ThermalParams(), fake_ha)
    controller.step(now=fake_ha.now, apply=False)
    assert controller.cfg.control.setpoint == pytest.approx(22.5)
    assert controller.solver.cfg.control.comfort_min == pytest.approx(21.8)


def test_setback_bands_are_lopsided_on_purpose(cfg):
    """Too cold is what matters when nobody is home. A tight upper bound would
    also forbid reheating the slab ahead of a return."""
    from hpmpc.comfort import apply_mode

    holiday = apply_mode(cfg, "holiday").control
    assert holiday.comfort_above > holiday.comfort_below
    assert holiday.hard_above > holiday.hard_below
    # And a deep setback needs authority the normal mode does not.
    assert holiday.offset_max > apply_mode(cfg, "normal").control.offset_max


def test_a_profile_may_not_use_unknown_keys(cfg):
    cfg.modes.profiles["silly"] = {"setpoint": 20.0, "wibble": 1.0}
    with pytest.raises(ValueError, match="unknown keys"):
        cfg.validate()


def test_every_profile_must_set_a_setpoint(cfg):
    cfg.modes.profiles["vague"] = {"comfort_below": 1.0}
    with pytest.raises(ValueError, match="must set a setpoint"):
        cfg.validate()


def test_set_mode_writes_the_holiday_switch(cfg, fake_ha):
    from hpmpc.comfort import set_mode

    calls = []
    fake_ha.call_service = lambda d, s, data, **k: calls.append((d, s, data["entity_id"]))
    cfg.modes.holiday_entity = "input_boolean.semester"
    set_mode(cfg, fake_ha, "holiday")
    assert calls == [("input_boolean", "turn_on", "input_boolean.semester")]
    calls.clear()
    set_mode(cfg, fake_ha, "normal")
    assert calls[0] == ("input_boolean", "turn_off", "input_boolean.semester")


def test_set_mode_refuses_an_unknown_profile(cfg, fake_ha):
    from hpmpc.comfort import set_mode

    cfg.modes.holiday_entity = "input_boolean.semester"
    with pytest.raises(ValueError, match="not a defined profile"):
        set_mode(cfg, fake_ha, "party")


def test_set_mode_needs_somewhere_to_write(cfg, fake_ha):
    from hpmpc.comfort import set_mode

    with pytest.raises(ValueError, match="No mode entity"):
        set_mode(cfg, fake_ha, "holiday")
