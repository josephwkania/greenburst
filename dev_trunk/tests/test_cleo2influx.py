"""
Tests for cleo2influx.py: reading the SETI server's lines into Influx points like pointing_dump.py's.

tests/data/cleo_stream.txt is 75 s of the server's output, recorded from greenburst on 2026-11-04 (telescope
stopped, Rcvr_2500 at turret angle 0). The tests use a stand-in for the astropy conversion, which CI doesn't
install.

Run from dev_trunk:  python -m pytest tests
"""

import os
import sys

import numpy as np
import pytest

from test_pointing_dump import DEV_TRUNK, functions

sys.path.insert(0, DEV_TRUNK)

import cleo2influx  # noqa: E402
from cleo2influx import CONNECTIONS, KEYS, Telescope  # noqa: E402

STREAM = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "cleo_stream.txt"
)
DUMP_LINES = 50  # the dump after connecting, then the server's changes

# The fields pointing_dump.py writes, less WEBCNTRL
POINTING_DUMP_FIELDS = {
    "tags", "MJD", "UTC", "IFV1TNCI", "ATRXOCTA", "SCPROJID", "ATMTLS", "ATAZIND", "AZMJD", "ATELIND", "ELMJD",
    "ATMFBS", "DATA_VALID", "RA_deg", "DEC_deg", "RA_drift", "DEC_drift", "GL_deg", "GB_deg", "ELCORR", "AZCORR",
}  # fmt: skip


def fake_sky(mjd, az, el):
    return az + 100, el - 30, az + 200, el - 60


def stream():
    with open(STREAM) as f:
        return f.read().splitlines()


def replay(lines, tel=None):
    tel = tel or Telescope(sky=fake_sky)
    return tel, [p for p in (tel.update(line) for line in lines) if p is not None]


def position(mjd, az=45.0, el=77.0):
    return [f"{mjd} {KEYS['ATAZIND']} {az}", f"{mjd} {KEYS['ATELIND']} {el}"]


def dump(**change):
    """The server's dump after connecting, with some values changed: SETI server key -> value."""
    lines = stream()[:DUMP_LINES]
    for key, value in change.items():
        lines = [
            f"{line.split()[0]} {key} {value}" if line.split()[1] == key else line
            for line in lines
        ]
    return lines


# ---------------------------------------------------------------- parse


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "61348.677603 AntennaManager,ccuData,Az,indicated 45.894802065755",
            (61348.677603, "AntennaManager,ccuData,Az,indicated", "45.894802065755"),
        ),
        (
            "61348.6776042 Antenna,TurretLocations 1:359.695:Rcvr1_2 2:299.665:Rcvr_MBA1_5",
            (
                61348.6776042,
                "Antenna,TurretLocations",
                "1:359.695:Rcvr1_2 2:299.665:Rcvr_MBA1_5",
            ),
        ),
        ("61348.6776042 Antenna,trackBeam", (61348.6776042, "Antenna,trackBeam", "")),
        ("61348.678287 Thump Thump", None),
        ("61348.678287 ERRORMSG Cannot connect", None),
        ("GOODDAY", None),
        ("", None),
        ("nan? Antenna,receiver Rcvr1_2", None),
    ],
)
def test_parse(line, expected):
    assert cleo2influx.parse(line) == expected


# ---------------------------------------------------------------- the recorded stream


def test_recorded_stream():
    lines = stream()
    _, points = replay(lines)
    az_lines = [
        line.split()
        for line in lines[DUMP_LINES:]
        if line.split()[1] == KEYS["ATAZIND"]
    ]
    # One point a second; none for the position in the dump, which comes before the ScanCoordinator's values,
    # or for the last azimuth, whose elevation came after the recording stopped
    assert len(points) == len(az_lines) - 1 == 75
    assert [p["fields"]["MJD"] for p in points] == [float(a[0]) for a in az_lines[:-1]]
    assert [p["fields"]["ATAZIND"] for p in points] == [
        float(a[2]) for a in az_lines[:-1]
    ]
    assert np.all(np.diff([p["time"] for p in points]) > 0.99e9)

    first = points[0]["fields"]
    assert set(first) == POINTING_DUMP_FIELDS | {
        "SCSTATE",
        "SCSCAN",
        "SCSOURCE",
        "ATEXECST",
    }
    assert first == {
        **first,
        "tags": "gbt",
        "MJD": 61348.6776146,
        "AZMJD": 61348.6776146,
        "ELMJD": 61348.6776146,
        "UTC": "16:15:45",
        "IFV1TNCI": "Rcvr_2500",
        "ATRXOCTA": 0.0,
        "SCPROJID": "JSCI_RYAN_06",
        "ATMTLS": "TURRET_LOCK_LOCKED",
        "ATMFBS": "BOOM_RETRACTED",
        "ATAZIND": 45.894791337013,
        "ATELIND": 77.799158125916,
        "AZCORR": 45.894791337013,  # L-band at turret angle 0: no correction
        "ELCORR": 77.799158125916,
        "DATA_VALID": 1,
        "RA_drift": 0.0,
        "SCSTATE": "Ready",
        "SCSCAN": "6",
        "SCSOURCE": "J1825-1227",
        "ATEXECST": "Stopped",
    }
    assert points[1]["fields"]["RA_drift"] == pytest.approx(
        points[1]["fields"]["ATAZIND"] - points[0]["fields"]["ATAZIND"], abs=1e-12
    )


def test_field_types_match_influx():
    # Influx keeps one type per field: these as in the telescope measurement pointing_dump writes
    _, points = replay(stream())
    for name, value in points[0]["fields"].items():
        if name == "DATA_VALID":
            assert type(value) is int
        elif name in (
            "tags",
            "UTC",
            "IFV1TNCI",
            "SCPROJID",
            "ATMTLS",
            "ATMFBS",
        ) or name.startswith(("SC", "ATEX")):
            assert type(value) is str
        else:
            assert type(value) is float, name


def test_influx_time():
    # pointing_dump's time stamp: astropy's ISO string for the MJD, to the millisecond
    assert cleo2influx.influx_time(61348.6776146) == 1793808945901000000
    assert cleo2influx.influx_time(40587.0) == 0


def test_reconnect_writes_no_duplicates():
    lines = stream()
    tel, points = replay(lines)
    tel.reset()
    # The server sends its dump again, with the last position
    _, again = replay(lines[:DUMP_LINES] + position(points[-1]["fields"]["MJD"]), tel)
    assert again == []
    _, later = replay(position(points[-1]["fields"]["MJD"] + 1 / 86400), tel)
    assert len(later) == 1


def test_reset_forgets_values():
    tel, _ = replay(stream())
    tel.reset()
    # A position before the dump has come again: no point from the old values
    assert replay(position(61349.0), tel)[1] == []


# ---------------------------------------------------------------- positions


def test_az_and_el_from_the_same_sample():
    tel, _ = replay(dump())
    mjd = 61348.7
    assert tel.update(f"{mjd} {KEYS['ATAZIND']} 45.0") is None
    assert tel.update(f"{mjd - 1e-5} {KEYS['ATELIND']} 77.0") is None
    point = tel.update(f"{mjd} {KEYS['ATELIND']} 77.1")
    assert point["fields"]["MJD"] == mjd
    assert (point["fields"]["ATAZIND"], point["fields"]["ATELIND"]) == (45.0, 77.1)


def test_correction_applied():
    _, points = replay(
        dump(**{KEYS["ATRXOCTA"]: 100.0}) + position(61348.7, 200.0, 45.0)
    )
    fields = points[0]["fields"]
    el, az = cleo2influx.pointing_corr(100.0, 200.0, 45.0)
    assert (fields["ELCORR"], fields["AZCORR"]) == (el, az)
    assert az != 200.0
    assert (fields["RA_deg"], fields["DEC_deg"]) == (
        az + 100,
        el - 30,
    )  # converted from the corrected position


@pytest.mark.parametrize(
    "change",
    [
        {KEYS["SCPROJID"]: "JMAINT"},
        {KEYS["ATMTLS"]: "TURRET_LOCK_UNLOCKED"},
        {KEYS["ATMFBS"]: "BOOM_EXTENDED"},
        {KEYS["ATRXOCTA"]: 179.67},
        {CONNECTIONS[0]: "down"},
        {CONNECTIONS[2]: "down"},
        {CONNECTIONS[4]: "down"},
    ],
)
def test_invalid(change):
    _, points = replay(dump(**change) + position(61348.7))
    assert points[0]["fields"]["DATA_VALID"] == 0


@pytest.mark.parametrize("turret", [0.0, 359.695, 299.665, 259.69, 99.65, 59.666])
def test_valid(turret):
    _, points = replay(dump(**{KEYS["ATRXOCTA"]: turret}) + position(61348.7))
    assert points[0]["fields"]["DATA_VALID"] == 1


def test_unreadable_position():
    _, points = replay(dump() + position(61348.7, "nan?", 77.0))
    assert points == []


# ---------------------------------------------------------------- the same as pointing_dump


def test_same_functions_as_pointing_dump():
    corr, flags = functions(
        os.path.join(DEV_TRUNK, "pointing_dump.py"), ["pointing_corr", "flag_maker"]
    )
    turret, az, el = (
        np.array([0.0, 60, 100, 260, 300]),
        np.full(5, 200.0),
        np.array([6.0, 30, 45, 60, 89.9]),
    )
    assert np.allclose(
        cleo2influx.pointing_corr(turret, az, el),
        corr(turret, az, el),
        rtol=0,
        atol=1e-12,
    )
    for key, values in [
        ("SCPROJID", ["JMAINT", "AGBT26A_001"]),
        ("ATMTLS", ["TURRET_LOCK_LOCKED", "TURRET_LOCK_UNLOCKED"]),
        ("ATMFBS", ["BOOM_RETRACTED", "BOOM_EXTENDED"]),
        ("ATRXOCTA", [0.0, 109.4, 109.6, 180.0, 250.4, 250.6, 359.7]),
    ]:
        for value in values:
            assert cleo2influx.flag_maker(key, value) == flags(key, value), (key, value)


def test_sky_as_pointing_dump():
    # test_influx_2df puts a stand-in astropy module in sys.modules when astropy is not installed
    pytest.importorskip("astropy.coordinates")
    # The first recorded position against the telescope's own J2000 position for it (ccuData RaJ2000, DcJ2000):
    # within 0.1"
    ra, dec, gl, gb = cleo2influx.sky(61348.6776146, 45.894791337013, 77.799158125916)
    assert (ra, dec) == (
        pytest.approx(220.47340899394, abs=3e-4),
        pytest.approx(46.399236070632, abs=3e-4),
    )
    assert 0 <= gl < 360 and -90 <= gb <= 90


# ---------------------------------------------------------------- the pointing file for udp2fil


def read_pointing(path):
    with open(path) as f:
        text = f.read().splitlines()
    assert text[0].startswith("# MJD RA_deg DEC_deg AZCORR ELCORR DATA_VALID")
    return [line.split() for line in text[1:]]


def test_pointing_file(tmp_path):
    path = str(tmp_path / "greenburst_pointing")
    _, points = replay(stream())
    pointing = cleo2influx.PointingFile(path)
    for point in points:
        pointing.add(point)
    rows = read_pointing(path)
    assert len(rows) == len(points) == 75
    assert all(len(row) == 6 for row in rows)
    last = points[-1]["fields"]
    mjd, ra, dec, az, el, valid = rows[-1]
    assert float(mjd) == pytest.approx(last["MJD"], abs=1e-8)
    assert (float(ra), float(dec)) == (
        pytest.approx(last["RA_deg"], abs=1e-6),
        pytest.approx(last["DEC_deg"], abs=1e-6),
    )
    assert (float(az), float(el)) == (
        pytest.approx(last["AZCORR"], abs=1e-6),
        pytest.approx(last["ELCORR"], abs=1e-6),
    )
    assert valid == "1"
    assert os.listdir(str(tmp_path)) == ["greenburst_pointing"]


def test_pointing_file_keeps_two_minutes(tmp_path):
    path = str(tmp_path / "greenburst_pointing")
    tel, _ = replay(dump())
    pointing = cleo2influx.PointingFile(path)
    start = 61348.7
    for s in range(300):
        for line in position(start + s / 86400):
            point = tel.update(line)
            if point is not None:
                pointing.add(point)
    mjds = [float(row[0]) for row in read_pointing(path)]
    assert len(mjds) == 121
    assert (mjds[-1] - mjds[0]) * 86400 == pytest.approx(120, abs=1e-3)
    assert mjds == sorted(mjds)


def test_pointing_file_after_a_gap(tmp_path):
    path = str(tmp_path / "greenburst_pointing")
    _, points = replay(stream())
    pointing = cleo2influx.PointingFile(path)
    pointing.add(points[0])
    _, later = replay(dump() + position(points[0]["fields"]["MJD"] + 1 / 24))
    pointing.add(later[0])
    assert len(read_pointing(path)) == 1


def test_pointing_file_invalid(tmp_path):
    path = str(tmp_path / "greenburst_pointing")
    _, points = replay(
        dump(**{KEYS["ATMTLS"]: "TURRET_LOCK_UNLOCKED"}) + position(61348.7)
    )
    cleo2influx.PointingFile(path).add(points[0])
    assert read_pointing(path)[-1][5] == "0"


def test_pointing_file_unwritable(tmp_path, caplog):
    pointing = cleo2influx.PointingFile(
        str(tmp_path / "missing" / "greenburst_pointing")
    )
    _, points = replay(stream())
    for point in points[:3]:
        pointing.add(point)  # logs, doesn't raise
    assert len([r for r in caplog.records if "Could not write" in r.getMessage()]) == 1
