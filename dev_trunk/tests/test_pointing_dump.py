"""
Tests for pointing_dump.py's turret correction (pointing_corr) and validity flags (flag_maker, data_valid).

pointing_dump connects to Redis and reads config/conf.yaml when imported, so the tests take only these
functions out of its source.

Run from dev_trunk:  python -m pytest tests
"""

import ast
import logging
import os

import numpy as np
import pandas as pd
import pytest

DEV_TRUNK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "turret_pulses.csv")


def functions(path, names):
    tree = ast.parse(open(path).read())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "logging": logging}
    exec(compile(tree, path, "exec"), namespace)
    return [namespace[name] for name in names]


pointing_corr, flag_maker, data_valid = functions(os.path.join(DEV_TRUNK, "pointing_dump.py"),
                                                  ["pointing_corr", "flag_maker", "data_valid"])


def beam_offset(turret, el, az=200.0):
    """The beam's offset from the pointing in arcmin: (cross-elevation, elevation)."""
    beam_el, beam_az = pointing_corr(turret, az, el)
    return (beam_az - az) * np.cos(np.radians(el)) * 60, (beam_el - el) * 60


# ---------------------------------------------------------------- pointing_corr

@pytest.mark.parametrize("turret", [0.0, 360.0])
def test_l_band_unchanged(turret):
    assert pointing_corr(turret, 123.4, 45.6) == (pytest.approx(45.6, abs=1e-12), pytest.approx(123.4, abs=1e-12))


def test_continuous_through_zero():
    a, b = np.array(beam_offset(359.7, 40)), np.array(beam_offset(0.3, 40))
    assert np.hypot(*(a - b)) < 0.3   # arcmin


@pytest.mark.parametrize("turret, expected", [
    # Surnis et al. 2019, Table 1, negated: where the beam is relative to the pointing (cross-el', el')
    (60, (22.1, 12.5)), (100, (25.4, 29.8)), (260, (-25.3, 30.2)), (300, (-22.5, 13.2)),
])
def test_surnis_offsets(turret, expected):
    assert beam_offset(turret, 40) == (pytest.approx(expected[0], abs=3), pytest.approx(expected[1], abs=3))


@pytest.mark.parametrize("el", [6, 30, 60, 85])
def test_offset_on_sky_independent_of_elevation(el):
    assert beam_offset(100, el) == (pytest.approx(25.18 * np.sin(np.radians(100))),
                                    pytest.approx(23.30 * (1 - np.cos(np.radians(100)))))


def test_zenith_guard():
    beam_el, beam_az = pointing_corr(100, 200.0, 89.99)
    assert np.isfinite(beam_az)
    assert abs(beam_az - 200) <= 25.18 / 60 / np.cos(np.radians(89.5)) + 1e-9


def test_azimuth_keeps_gbt_wrap():
    # The GBT's azimuth runs past 360 (to about 450); the corrected value stays in the same range for the
    # AltAz conversion, rather than wrapping to [0, 360)
    beam_el, beam_az = pointing_corr(60, 406.6, 43.5)
    assert 406.6 < beam_az < 407.2


def test_arrays():
    el, az = pointing_corr(np.array([0.0, 60, 260]), np.array([10.0, 200, 359.9]), np.array([30.0, 45, 89.9]))
    assert el.shape == az.shape == (3,)
    assert el[0] == 30 and az[0] == 10


def test_known_pulsar_detections():
    # FETCH-positive pulses from bright pulsars with another receiver in focus (greenburst.db, 2019-2026):
    # telescope az/el and turret angle at the pulse, and the pulsar's az/el then. The pipeline's old correction
    # put the beam 31-82' from them; the corrected beam is within 17', median 9'.
    d = pd.read_csv(FIXTURE)
    beam_el, beam_az = pointing_corr(d.turret.values, d.tel_az.values, d.tel_el.values)
    dist = np.hypot(((d.psr_az - beam_az + 180) % 360 - 180) * np.cos(np.radians(d.psr_el)), d.psr_el - beam_el) * 60
    assert len(d) == 19 and set(np.round(d.turret, -1)) == {60, 100, 260, 300}
    assert dist.max() < 17
    assert np.median(dist) < 10


# ---------------------------------------------------------------- validity flags

def status(**change):
    s = dict(SCPROJID="AGBT26A_001", ATMTLS="TURRET_LOCK_LOCKED", ATMFBS="BOOM_RETRACTED", ATRXOCTA=0.0, WEBCNTRL=1)
    s.update(change)
    return s


@pytest.mark.parametrize("turret", [0.0, 359.7, 59.666, 99.65, 259.69, 299.665])
def test_valid_turret_positions(turret):
    assert data_valid(status(ATRXOCTA=turret)) == 1


@pytest.mark.parametrize("change", [
    dict(SCPROJID="JMAINT"), dict(ATMTLS="TURRET_LOCK_UNLOCKED"), dict(ATMFBS="BOOM_EXTENDED"),
    dict(ATRXOCTA=180.0), dict(ATRXOCTA=140.0), dict(ATRXOCTA=220.0), dict(WEBCNTRL=0),
])
def test_invalid(change):
    assert not data_valid(status(**change))


@pytest.mark.parametrize("turret, valid", [(109.4, 1), (109.6, 0), (110.0, 0), (250.0, 0), (250.4, 0), (250.6, 1)])
def test_turret_angle_limits(turret, valid):
    # Valid below 110 deg or above 250 deg, after rounding to a whole degree
    assert flag_maker("ATRXOCTA", turret) == valid


def test_pointing_test_has_same_correction():
    # pointing_test.py keeps its own copy of the correction
    (copy,) = functions(os.path.join(DEV_TRUNK, "pointing_test.py"), ["pointing_corr"])
    turret, az, el = np.array([0.0, 60, 100, 260, 300]), np.full(5, 200.0), np.array([6.0, 30, 45, 60, 89.9])
    assert np.allclose(copy(turret, az, el), pointing_corr(turret, az, el), rtol=0, atol=1e-12)
