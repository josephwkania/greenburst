"""
Tests for the low-DM target search (low_dm_targets.py and its use in
stage_1.py / stage_2.py).

The stage tests run the real begin_main() functions with the external
services (RabbitMQ, InfluxDB, Elasticsearch, Slack, the GPU queue, sigproc
reading) replaced by stubs, and compare against the unpatched stages from git
where the behaviour must not change.

Run from dev_trunk:  python -m pytest tests
"""

import builtins
import importlib
import os
import subprocess
import sys
import types
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

DEV_TRUNK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEV_TRUNK)

import low_dm_targets as ldt  # noqa: E402
import pulsar_plot_limit as ppl  # noqa: E402

J1022 = ldt.Target("J1022+1001", 155.74125, 10.031333, 10.26)
J2145 = ldt.Target("J2145-0750", 326.46, -7.838472, 9.01)
B0355 = ldt.Target("J0358+5413", 59.723849, 54.220496, 57.14)
MJD = 61000.5

# heimdall .cand columns: snr sample tcand width dm_trial dm members begin end
CAND_ROWS = [
    (8.03, 9280, 2.37, 1, 1, 10.2914, 6, 9280, 9281),      # in J1022 window
    (8.66, 95363, 24.42, 1, 2, 10.5828, 54, 95360, 95365),  # in J1022 window
    (8.45, 192200, 49.20, 1, 3, 10.8742, 1, 192200, 192201),  # in window, too few members
    (8.33, 2049992, 524.80, 12, 4, 11.1656, 13, 2049990, 2049995),  # in window, too late and too wide
    (6.50, 1000, 0.26, 1, 1, 10.2914, 30, 999, 1001),       # in window, S/N too low
    (9.00, 50000, 12.80, 2, 18, 15.5000, 9, 49990, 50010),  # below DM 20, no target
    (9.30, 1113000, 285.03, 5, 40, 22.5359, 3284, 1112990, 1113010),  # normal
    (9.99, 1898400, 485.99, 6, 60, 37.1408, 951, 1898390, 1898410),  # normal
    (110.49, 690470, 176.76, 7, 300, 380.287, 58758, 690460, 690480),  # normal
    (8.27, 1423600, 364.45, 8, 70, 52.0882, 56, 1423590, 1423610),  # too wide
]


# ---------------------------------------------------------------- unit tests

def test_shipped_target_list():
    targets = ldt.load_targets()
    names = {t.name for t in targets}
    assert len(targets) >= 60
    assert {"J1022+1001", "J2145-0750", "J1713+0747", "J1744-1134"} <= names
    assert all(t.dm < 20.2 and t.dec_deg > -46 and 0 <= t.ra_deg < 360 for t in targets)


def test_missing_target_list(tmp_path):
    assert ldt.load_targets(str(tmp_path / "nope.csv")) == []


def test_separation():
    assert ldt.angular_separation(10, 20, 10, 20) == pytest.approx(0)
    assert ldt.angular_separation(0, 0, 1, 0) == pytest.approx(1)
    assert ldt.angular_separation(359.9, 0, 0.1, 0) == pytest.approx(0.2)  # RA wrap
    assert ldt.angular_separation(0, 89.9, 180, 89.9) == pytest.approx(0.2)  # over the pole


def test_targets_in_beam():
    targets = [J1022, J2145]
    # 3 arcmin off J1022: in; 0.2 deg off: out
    assert ldt.targets_in_beam([155.74125], [10.081333], targets) == [J1022]
    assert ldt.targets_in_beam([155.74125], [10.231333], targets) == []
    # a slew that passes the source part way through the file
    ra = np.linspace(150, 160, 600)
    assert ldt.targets_in_beam(ra, np.full(600, 10.03), targets) == [J1022]
    # NaNs and empty input
    assert ldt.targets_in_beam([np.nan, 155.74125], [np.nan, 10.031333], targets) == [J1022]
    assert ldt.targets_in_beam([], [], targets) == []


def test_heimdall_dm_min():
    assert ldt.heimdall_dm_min([]) == 10.0
    assert ldt.heimdall_dm_min([J1022]) == pytest.approx(7.26)
    assert ldt.heimdall_dm_min([J1022, J2145]) == pytest.approx(6.01)
    assert ldt.heimdall_dm_min([J1022._replace(dm=1.0)]) == 0.0
    assert ldt.heimdall_dm_min([J1022._replace(dm=19.8)]) == 10.0  # never above the normal limit


def test_dm_window():
    mask = ldt.dm_window_mask([8.2, 8.3, 10.26, 12.26, 12.3], [J1022])
    assert mask.tolist() == [False, True, True, True, False]
    # 10 % for higher DMs: a DM 19.8 target accepts 17.82 - 21.78
    wide = ldt.Target("X", 0, 0, 19.8)
    assert ldt.dm_window_mask([17.7, 17.9, 21.7, 21.9], [wide]).tolist() == [False, True, True, False]


def test_window_cap():
    df = pd.DataFrame({"snr": np.arange(150, dtype=float) + 8, "dm": np.full(150, 10.3)})
    mask = ldt.window_only_mask(df, [J1022], np.ones(150, bool), max_cands=100)
    assert mask.sum() == 100
    assert df["snr"][mask].min() == 58  # the 100 brightest


def test_stage1_helper(tmp_path):
    on = pd.DataFrame({"DATA_VALID": [1] * 5 + [0], "RA_deg": [155.74] * 5 + [0.0], "DEC_deg": [10.03] * 5 + [0.0]})
    dm_min, matched = ldt.stage1_dm_min(on, str(tmp_path))
    assert matched[0].name == "J1022+1001" and dm_min == pytest.approx(7.26)
    assert [t.name for t in ldt.read_obs_targets(str(tmp_path))] == ["J1022+1001"]
    # invalid samples do not count
    only_invalid = on.assign(DATA_VALID=[0] * 5 + [1])
    assert ldt.stage1_dm_min(only_invalid, str(tmp_path / "x"))[0] == 10.0
    # anything unexpected falls back to the normal limit
    assert ldt.stage1_dm_min(None, str(tmp_path)) == (10.0, [])
    assert ldt.stage1_dm_min(pd.DataFrame({"x": [1]}), str(tmp_path)) == (10.0, [])


def h5(dm, snr, tcand=1.0):
    return "/x/cands/cand_tstart_{}_tcand_{:.7f}_dm_{:.5f}_snr_{:.5f}.h5".format(MJD, tcand, dm, snr)


def test_h5_dm_snr():
    assert ppl.h5_dm_snr(h5(10.2914, 8.03)) == (10.2914, 8.03)
    assert ppl.h5_dm_snr("/x/not_a_candidate.h5") == (None, None)


def test_assign_to_pulsars():
    cands = [h5(57.0, 9), h5(57.5, 30), h5(380.3, 110), h5(10.29, 8.1), h5(56.0, 12)]
    others, by_pulsar = ppl.assign_to_pulsars(cands, [B0355, J1022])
    assert others == [cands[2]]
    assert by_pulsar["J0358+5413"] == [cands[1], cands[4], cands[0]]  # highest S/N first
    assert by_pulsar["J1022+1001"] == [cands[3]]


def test_pulsars_in_beam():
    df = pd.DataFrame({"RA_deg": [59.8, None], "DEC_deg": [54.3, None]})
    assert ppl.pulsars_in_beam(df, [B0355, J1022]) == [B0355]
    assert ppl.pulsars_in_beam(df.assign(DEC_deg=[54.6, None]), [B0355]) == []  # 0.38 deg away
    assert ppl.pulsars_in_beam(pd.DataFrame({"x": [1]}), [B0355]) == []
    assert ppl.pulsars_in_beam(None, [B0355]) == []


def test_plot_limiter_sessions(tmp_path):
    path = str(tmp_path / ppl.STATE_NAME)
    cands = ["a", "b", "c"]
    file_step = 8.4 / 1440
    chosen = []
    for i in range(4):  # one visit, consecutive 8.4 min files
        limiter = ppl.PlotLimiter(path)  # state survives between files
        chosen.append(limiter.choose("J0358+5413", cands, MJD + i * file_step))
        limiter.save()
    assert chosen == [["a", "b"], ["a", "b"], [], []]
    # a new visit after more than 30 min
    limiter = ppl.PlotLimiter(path)
    assert limiter.choose("J0358+5413", cands, MJD + 3 * file_step + 31 / 1440) == ["a", "b"]
    # other pulsars are counted separately
    assert limiter.choose("J1705-1906", ["x"], MJD) == ["x"]


def test_plot_limiter_bad_state(tmp_path):
    path = tmp_path / ppl.STATE_NAME
    path.write_text("not json")
    assert ppl.PlotLimiter(str(path)).choose("J0358+5413", ["a", "b", "c"], MJD) == ["a", "b"]


def test_select_for_stage3(tmp_path):
    cands = [h5(57.0, 9), h5(57.5, 30), h5(380.3, 110), h5(56.0, 12)]
    cand_df = pd.DataFrame({"RA_deg": [59.72], "DEC_deg": [54.22]})
    to_stage3, summary = ppl.select_for_stage3(cands, [], cand_df, MJD, str(tmp_path))
    assert to_stage3 == [cands[2], cands[1], cands[3]]
    assert summary == "Known pulsars at MJD {}: J0358+5413: 3 FETCH-positive, 2 plotted".format(MJD)
    # no known pulsar in the beam: everything goes to stage 3, no summary
    off = pd.DataFrame({"RA_deg": [10.0], "DEC_deg": [0.0]})
    assert ppl.select_for_stage3(cands, [], off, MJD, str(tmp_path)) == (cands, None)
    # low-DM targets are always reported
    _, summary = ppl.select_for_stage3([], [J1022], off, MJD, str(tmp_path), n_window=4)
    assert summary == "Known pulsars at MJD {}: J1022+1001: 0 FETCH-positive; 4 low-DM window cands".format(MJD)
    # any failure sends everything
    assert ppl.select_for_stage3(cands, [], cand_df, MJD, str(tmp_path / "missing" / "dir")) == (cands, None)


def test_shipped_bright_list():
    bright = ldt.load_targets(ppl.BRIGHT_FILE)
    names = {t.name for t in bright}
    assert len(bright) >= 40
    assert {"J0358+5413", "J1705-1906", "J0332+5434", "J1939+2134"} <= names
    # one entry per globular cluster, and no near-duplicates
    assert sum(t.name.startswith("J1748-2446") for t in bright) == 1
    assert "J2023+2853_P" not in names


def test_make_bright_list(tmp_path):
    summary = tmp_path / "summary.csv"
    rows = [
        ("J0358+5413", "B0355+54", 57.14, 7191, 59.723849, 54.220496),
        ("J1748-2446ab", "", 242.2, 140, 267.02, -24.78),
        ("J1748-2446A", "B1744-24A", 242.36, 140, 267.009372, -24.777169),
        ("J2022+2854", "B2020+28", 24.63, 1318, 305.654446, 28.906418),
        ("J2023+2853_P", "", 22.74, 967, 305.8, 28.88),  # 0.13 deg away, same DM window
        ("J0147+5922", "B0144+59", 40.11, 9, 26.9, 59.4),  # too few
    ]
    with open(str(summary), "w") as f:
        f.write("pulsar,bname,dm,dm_match,ra_deg,dec_deg\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    kept = ppl.make_bright_list(str(summary), str(tmp_path / "bright.csv"))
    assert [t.name for t in kept] == ["J0358+5413", "J2022+2854", "J1748-2446A"]
    assert ldt.load_targets(str(tmp_path / "bright.csv")) == kept


def test_targets_from_psrcat():
    text = (
        "J1022+1001    10:22:57.9  +10:01:52.8  10.26\n"
        "J1154-19      11:54       -19          10.69\n"
        "J2124-3358    21:24:43.8  -33:58:44.6   4.60\n"
        "J0437-4715    04:37:15.9  -47:15:09.1   2.64\n"  # too far south
        "J9999+9999    *           *             1.0\n"
    )
    targets = ldt.targets_from_psrcat(text)
    assert [t.name for t in targets] == ["J1022+1001", "J1154-19", "J2124-3358"]
    assert targets[0].ra_deg == pytest.approx(155.74125, abs=1e-5)
    assert targets[2].dec_deg == pytest.approx(-33.979056, abs=1e-5)


# ---------------------------------------------------------------- stage tests

class Recorder:
    def __init__(self):
        self.gpu, self.queue, self.slack = [], [], []


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """
    Replaces the external modules the stages import, and records what the
    stages send to the GPU queue, RabbitMQ and Slack. Writes to /ldata/dev_tests
    (stage 2's pre-FETCH log) go to the test directory.
    """
    rec = Recorder()
    real_open = builtins.open

    def open_redirected(file, *args, **kwargs):
        if isinstance(file, str) and file.startswith("/ldata/dev_tests/"):
            file = str(tmp_path / "dev_tests" / os.path.basename(file))
            os.makedirs(os.path.dirname(file), exist_ok=True)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_redirected)
    state = {"influx": None, "tstart": MJD, "predict": None, "pointing": None}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    exceptions = module("pika.exceptions", StreamLostError=Exception, ConnectionResetError=Exception)
    module("pika", exceptions=exceptions, BlockingConnection=None, ConnectionParameters=None)

    class SigprocFile:
        def __init__(self, filename=None, fp=None):
            self.tstart = state["tstart"]
            self.chan_freqs = np.linspace(1920, 960, 4096)
            self.bandpass = np.ones(4096)

    module("pysigproc", SigprocFile=SigprocFile)
    module("elasticsearch", Elasticsearch=lambda *a, **k: None)
    module("dump_all", tel_df_to_es=lambda *a, **k: None)

    def send2gpuQ(command):
        rec.gpu.append(command)
        if command.startswith("predict.py") and state["predict"]:
            state["predict"](command)

    module("gpu_client", send2gpuQ=send2gpuQ)

    def extend_df(influx_df, cand_df):
        # What influx_2df.extend_df produces when the pointing lookup succeeds
        cand_df = cand_df.copy()
        for column in ("ATAZIND", "ATELIND", "AZCORR", "ELCORR", "RA_deg", "DEC_deg", "RA_drift", "DEC_drift",
                       "SCPROJID", "WEBCNTRL", "IFV1TNCI", "ATRXOCTA", "cand_gl", "cand_gb", "cand_ne2001",
                       "cand_ymw16"):
            cand_df.loc[:, column] = None
        if state["pointing"] is not None:
            cand_df.loc[:, "RA_deg"], cand_df.loc[:, "DEC_deg"] = state["pointing"]
        cand_df.loc[:, "cand_valid"] = 1
        return cand_df

    module("influx_2df", mjd2influx=lambda mjd: state["influx"], extend_df=extend_df)
    module("pika_send", send2Q=lambda queue, body: rec.queue.append((queue, body)))
    module("slack_send", send_msg_2_slack=lambda msg: rec.slack.append(msg),
           send_img_2_slack=lambda *a, **k: None)
    return rec, state


def load_stage(name, work_dir, original=False):
    """
    Imports stage_1/stage_2 (patched, or the unpatched version on the greenburst
    branch) with its work directory pointed at WORK_DIR.
    """
    if original:
        source = subprocess.run(
            ["git", "show", "greenburst:dev_trunk/{}.py".format(name)], cwd=DEV_TRUNK,
            stdout=subprocess.PIPE, check=True, universal_newlines=True,
        ).stdout.replace("/ldata/trunk", str(work_dir))
        mod = types.ModuleType(name + "_original")
        exec(compile(source, name + "_original.py", "exec"), mod.__dict__)
        return mod
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    mod.WORK_DIR = str(work_dir)
    return mod


class SyncProcess:
    """multiprocessing.Process stand-in that runs the target in this process."""

    def __init__(self, target, args=()):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


def pointing(ra, dec, n=540):
    return pd.DataFrame({"DATA_VALID": np.ones(n, int), "RA_deg": np.full(n, ra), "DEC_deg": np.full(n, dec),
                         "MJD": MJD + np.arange(n) / 86400.0})


def run_stage1(stubs, tmp_path, influx, original=False, skip_metadata=False):
    rec, state = stubs
    state["influx"] = influx
    work = tmp_path / ("orig" if original else "new")
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    fil = incoming / "data_2026-01-01_00-00-00.fil"
    fil.write_text("fake")
    stage = load_stage("stage_1", work, original)
    stage.Process = SyncProcess
    del rec.gpu[:], rec.queue[:]
    stage.begin_main(Namespace(file=str(fil), verbose=False, influx=not skip_metadata, ret=False))
    heimdall = [c for c in rec.gpu if c.startswith("heimdall")]
    assert len(heimdall) == 1
    return heimdall[0].replace(str(work), "WORK"), list(rec.queue), work / "data_2026-01-01_00-00-00"


def test_stage1_on_target(stubs, tmp_path):
    command, queue, obs_dir = run_stage1(stubs, tmp_path, pointing(155.742, 10.032))
    assert " -dm 7.26 10000 " in command
    assert [t.name for t in ldt.read_obs_targets(str(obs_dir))] == ["J1022+1001"]
    assert queue == [("stage02_queue", str(obs_dir))]


def test_stage1_off_target_unchanged(stubs, tmp_path):
    influx = pointing(83.633, 22.0145)  # Crab: DM 56.8, not a target
    new, new_queue, obs_dir = run_stage1(stubs, tmp_path, influx)
    old, _, _ = run_stage1(stubs, tmp_path, influx, original=True)
    assert new == old  # identical heimdall command
    assert " -dm 10 10000 " in new
    assert not (obs_dir / ldt.OBS_TARGETS_NAME).exists()


def test_stage1_bad_pointing_falls_back(stubs, tmp_path):
    influx = pointing(155.742, 10.032).drop(columns=["RA_deg"])
    command, _, _ = run_stage1(stubs, tmp_path, influx)
    assert " -dm 10 10000 " in command


def test_stage1_skip_metadata(stubs, tmp_path):
    # -m: no pointing is read, so the normal DM limit is used even on a target
    command, queue, obs_dir = run_stage1(stubs, tmp_path, pointing(155.742, 10.032), skip_metadata=True)
    assert " -dm 10 10000 " in command
    assert not (obs_dir / ldt.OBS_TARGETS_NAME).exists()
    assert queue == [("stage02_queue", str(obs_dir))]


def make_obs(work, rows, targets=None, name="data_2026-01-01_00-00-00"):
    obs = work / name
    obs.mkdir(parents=True)
    with open(str(obs / "2026-01-01-00:00:34_01.cand"), "w") as f:
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    (obs / (name + "_jb_4096.fil")).write_text("fake")
    (obs / "bad_chans.flag").write_text("")
    if targets:
        ldt.write_obs_targets(str(obs), targets)
    return obs


def run_stage2(stubs, tmp_path, rows, targets=None, original=False, label=1, pointing=None, tstart=MJD,
               name="data_2026-01-01_00-00-00"):
    rec, state = stubs
    work = tmp_path / ("orig" if original else "new")
    obs = make_obs(work, rows, targets, name)
    state["pointing"], state["tstart"] = pointing, tstart
    state["influx"] = pd.DataFrame({"x": [1]})

    def predict(command):
        # FETCH stand-in: label every candidate in the CSV
        cands = pd.read_csv(str(obs / (obs.name + ".csv")))
        names = [str(obs / "cands" / "cand_tstart_{}_tcand_{:.7f}_dm_{:.5f}_snr_{:.5f}.h5".format(
            tstart, r.tcand, r.dm, r.snr)) for r in cands.itertuples()]
        pd.DataFrame({"candidate": names, "probability": 0.9, "label": label}).to_csv(
            str(obs / "cands" / "results_a.csv"), index=False)

    state["predict"] = predict
    stage = load_stage("stage_2", work, original)
    del rec.gpu[:], rec.queue[:], rec.slack[:]
    stage.begin_main(Namespace(files=str(obs) + "/*cand", snr=8.0, width=7, dm=20.0, members=5, verbose=False,
                               outfile=False))
    csv_path = obs / (obs.name + ".csv")
    kept = pd.read_csv(str(csv_path)) if csv_path.exists() else None
    stage3 = [os.path.basename(body) for queue, body in rec.queue if queue == "stage03_queue"]
    return kept, stage3, list(rec.slack)


def test_stage2_no_targets_unchanged(stubs, tmp_path):
    new_kept, new_stage3, new_slack = run_stage2(stubs, tmp_path, CAND_ROWS)
    old_kept, old_stage3, old_slack = run_stage2(stubs, tmp_path, CAND_ROWS, original=True)
    pd.testing.assert_frame_equal(
        new_kept.drop(columns=["fil_file", "kill_mask"]), old_kept.drop(columns=["fil_file", "kill_mask"])
    )
    assert new_stage3 == old_stage3
    assert new_slack == old_slack
    assert sorted(new_kept["dm"]) == [22.5359, 37.1408, 380.287]


def test_stage2_pointing_elsewhere_unchanged(stubs, tmp_path):
    # with real pointing columns, but no known pulsar near
    new = run_stage2(stubs, tmp_path, CAND_ROWS, pointing=(10.0, 0.0))
    old = run_stage2(stubs, tmp_path, CAND_ROWS, pointing=(10.0, 0.0), original=True)
    assert new[1] == old[1] and new[2] == old[2]


def test_stage2_with_target(stubs, tmp_path):
    kept, stage3, slack = run_stage2(stubs, tmp_path, CAND_ROWS, targets=[J1022])
    # the normal three plus the two window candidates that pass the other cuts
    assert sorted(kept["dm"]) == [10.2914, 10.5828, 22.5359, 37.1408, 380.287]
    # the normal three go to stage 3, plus at most 2 plots of the target
    assert len(stage3) == 5 and sum("_dm_10." in name for name in stage3) == 2
    summary = [m for m in slack if m and m.startswith("Known pulsars")]
    assert summary == ["Known pulsars at MJD {}: J1022+1001: 2 FETCH-positive, 2 plotted; "
                       "2 low-DM window cands".format(MJD)]


def test_stage2_target_but_fetch_negative(stubs, tmp_path):
    kept, stage3, slack = run_stage2(stubs, tmp_path, CAND_ROWS, targets=[J1022], label=0)
    assert len(kept) == 5 and stage3 == []
    assert any(m and "J1022+1001: 0 FETCH-positive" in m for m in slack)


def test_stage2_bright_pulsar_visit(stubs, tmp_path):
    # 30 FETCH-positive pulses from B0355+54 per file, plus one unrelated candidate
    rows = [(9.0 + i, 1000 + i * 15000, 0.3 + i * 3.8, 3, 100, 57.0 + (i % 3) * 0.3, 20, 0, 0)
            for i in range(30)] + [(110.49, 690470, 176.76, 7, 300, 380.287, 58758, 0, 0)]
    step = 8.4 / 1440
    starts = [MJD, MJD + step, MJD + 2 * step, MJD + 2 * step + 2 / 24.0]  # 3 files, then a new visit
    sent = []
    for i, tstart in enumerate(starts):
        _, stage3, slack = run_stage2(stubs, tmp_path, rows, pointing=(59.72, 54.22), tstart=tstart,
                                      name="data_2026-01-01_00-0{}-00".format(i))
        pulsar_plots = [n for n in stage3 if "_dm_380." not in n]
        assert len(stage3) - len(pulsar_plots) == 1  # the unrelated candidate always goes
        sent.append(len(pulsar_plots))
        if pulsar_plots:
            assert all("_snr_38." in n or "_snr_37." in n for n in pulsar_plots)  # the brightest
        assert "J0358+5413: 30 FETCH-positive, {} plotted".format(len(pulsar_plots)) in slack[-1]
    assert sent == [2, 2, 0, 2]


def test_stage2_target_window_capped(stubs, tmp_path):
    assert ldt.MAX_WINDOW_CANDS == 100
    rows = [(8.0 + i * 0.1, 1000 + i * 4000, 0.25 + i, 1, 1, 10.2914, 6, 0, 0) for i in range(150)]
    kept, _, _ = run_stage2(stubs, tmp_path, rows, targets=[J1022])
    assert len(kept) == 100
    assert kept["snr"].min() == pytest.approx(13.0)
