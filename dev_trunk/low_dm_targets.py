#!/usr/bin/env python3
"""
Searches below the normal DM limits when the telescope points at a known
low-DM pulsar.

GREENBURST runs heimdall from DM 10 (stage_1.py) and stage_2.py drops
candidates below DM 20, which hides the bright low-DM millisecond pulsars
(e.g. J1022+1001, J2145-0750, J1713+0747). For files where the telescope was
within RADIUS of a pulsar in TARGETS_FILE:
    stage 1 starts heimdall at (lowest target DM - DM_MARGIN), never below 0,
            and writes the matched targets to <obs dir>/low_dm_targets.csv;
    stage 2 also keeps candidates within max(DM_TOL_ABS, DM_TOL_FRAC * DM) of
            a target's DM (at most MAX_WINDOW_CANDS per file, brightest
            first); their Slack plots are limited by pulsar_plot_limit.py.
All other files and candidates are processed exactly as before. Any error
here falls back to the normal limits.

The target list (low_dm_targets.csv) has two sets, told apart by its `class`
column:
    msp        every millisecond pulsar (P < 30 ms) with DM < 20.2 and Dec > -46
    canonical  the normal pulsars in CANONICAL_TARGETS: pulsars below the DM cut
               with real L-band time on source and many predicted pulses (sp_rates)
Its `nanograv` column is "15yr" for pulsars in the NANOGrav 15-year data set.
The single-pulse columns come from single_pulse_estimates.csv, made with sp_rates
(`sp-rates targets --targets low_dm_targets.csv -o single_pulse_estimates.csv`):
    sp_per_hour_low/mid/high  stage 2 candidates per hour on axis (pessimistic,
                              central, optimistic energy model; capped at 500/h)
    single_pulses             likely (pessimistic >= 1/h), possible (central >= 1/h),
                              marginal (optimistic >= 1/h), unlikely, or unknown
                              (no usable catalogue flux)
    sp_notes                  caveats: zero-DM filter, scintillation, rate limit

Only targets whose single_pulses rating is in SEARCH_RATINGS are searched (rows
without a rating are always kept). Override with the environment variable
GREENBURST_LOW_DM_RATINGS, e.g. "likely,possible", or "all" to search every target.

Make the target list (needs the psrcat program and catalogue):
    python3 low_dm_targets.py make -o low_dm_targets.csv
"""

import csv
import logging
import os
import re
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import namedtuple

import numpy as np

TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "low_dm_targets.csv")
OBS_TARGETS_NAME = "low_dm_targets.csv"  # written into each observation directory
RADIUS = 0.1  # deg; the L-band beam FWHM is 9.2 arcmin
DEFAULT_DM_MIN = 10.0  # heimdall's normal lower limit
DM_MARGIN = 3.0  # heimdall starts this far below the target DM
DM_TOL_ABS = 2.0  # stage 2 window: |DM - DM_psr| <= max(DM_TOL_ABS, DM_TOL_FRAC * DM_psr)
DM_TOL_FRAC = 0.1
MAX_WINDOW_CANDS = 100  # per file, so a bright pulsar cannot flood candmaker/FETCH
# Skip targets that cannot give single pulses (optimistic model < 1 candidate per hour)
SEARCH_RATINGS = ("likely", "possible", "marginal", "unknown")
RATINGS_ENV = "GREENBURST_LOW_DM_RATINGS"

Target = namedtuple("Target", ["name", "ra_deg", "dec_deg", "dm"])

# Normal pulsars below the DM cut chosen with sp_rates (Apr 2019 - Oct 2026 L-band
# pointing; predicted candidates from literature energy distributions):
#   tier 1, bright and never recorded; tier 2, already leaking through at the
#   lowest kept DM (20.2); tier 3, less time on source
CANONICAL_TARGETS = (
    "B0950+08", "B1133+16", "B1929+10", "B2327-20", "B0809+74", "B1237+25", "B2016+28",  # tier 1
    "B2310+42", "B0823+26", "B1540-06", "B1508+55", "B0301+19", "B1822-09", "B0450+55",  # tier 2
    "B0834+06", "B1919+21", "B2045-16", "B1322+83",  # tier 3
)

# NANOGrav 15-year data set (Agazie et al. 2023, ApJL 951, L9; 68 pulsars)
NANOGRAV_15YR = frozenset("""
J0023+0923 J0030+0451 J0340+4130 J0406+3039 J0437-4715 J0509+0856 J0557+1551 J0605+3757 J0610-2100
J0613-0200 J0614-3329 J0636+5128 J0645+5158 J0709+0458 J0740+6620 J0931-1902 J1012+5307 J1012-4235
J1022+1001 J1024-0719 J1125+7819 J1312+0051 J1453+1902 J1455-3330 J1600-3053 J1614-2230 J1630+3734
J1640+2224 J1643-1224 J1705-1903 J1713+0747 J1719-1438 J1730-2304 J1738+0333 J1741+1351 J1744-1134
J1745+1017 J1747-4036 J1751-2857 J1802-2124 J1811-2405 J1832-0836 J1843-1113 J1853+1303 J1857+0943
J1903+0327 J1909-3744 J1910+1256 J1911+1347 J1918-0642 J1923+2515 J1939+2134 J1944+0907 J1946+3417
J1955+2908 J2010-1323 J2017+0603 J2033+1734 J2043+1711 J2124-3358 J2145-0750 J2214+3000 J2229+2643
J2234+0611 J2234+0944 J2302+4442 J2317+1439 J2322+2057
""".split())
MSP_MAX_P0 = 0.03  # s

def search_ratings():
    """
    The single_pulses ratings to search: GREENBURST_LOW_DM_RATINGS if set
    (comma separated; "all" for no filter), else SEARCH_RATINGS. None means all.
    """
    value = os.environ.get(RATINGS_ENV)
    if value is None:
        return SEARCH_RATINGS
    value = value.strip().lower()
    if value in ("", "all", "*"):
        return None
    return tuple(r.strip() for r in value.split(",") if r.strip())


def load_targets(path=TARGETS_FILE, ratings=None):
    """
    Reads a target list (CSV with name, ra_deg, dec_deg, dm columns; any others
    are ignored).
    Args:
        path: The CSV file.
        ratings: If given, keep only rows whose single_pulses value is one of
            these; rows without a single_pulses value are kept.

    Returns:
        A list of Target; empty if the file is missing.
    """
    if not os.path.isfile(path):
        logging.warning("No low-DM target list at %s; using the normal DM limits", path)
        return []
    targets, skipped = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rating = (row.get("single_pulses") or "").strip().lower()
            if ratings is not None and rating and rating not in ratings:
                skipped.append(row["name"])
                continue
            targets.append(Target(row["name"], float(row["ra_deg"]), float(row["dec_deg"]), float(row["dm"])))
    if skipped:
        logging.debug("Not searching %d targets rated outside %s", len(skipped), ", ".join(ratings))
    return targets


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Haversine separation in degrees; arguments in degrees, numpy-broadcastable.
    """
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(x, dtype=float)) for x in (ra1, dec1, ra2, dec2))
    a = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def targets_in_beam(ra_deg, dec_deg, targets, radius=RADIUS):
    """
    Finds the targets that any pointing sample came within RADIUS of.
    Args:
        ra_deg, dec_deg: The pointing samples (e.g. the Influx RA_deg/DEC_deg
            columns of the valid samples of one file), in degrees.
        targets: Target list.
        radius: Degrees.

    Returns:
        The matched targets, in list order.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[good], dec[good]
    if ra.size == 0:
        return []
    matched = []
    for target in targets:
        # Cheap declination cut first
        close = np.abs(dec - target.dec_deg) <= radius
        if close.any() and (angular_separation(ra[close], dec[close], target.ra_deg, target.dec_deg) <= radius).any():
            matched.append(target)
    return matched


def heimdall_dm_min(targets, default=DEFAULT_DM_MIN, margin=DM_MARGIN):
    """
    heimdall's lower DM limit for a file pointed at TARGETS.
    """
    if not targets:
        return default
    return max(0.0, min(default, min(t.dm for t in targets) - margin))


def write_obs_targets(obs_dir, targets):
    path = os.path.join(obs_dir, OBS_TARGETS_NAME)
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(Target._fields)
        for target in targets:
            writer.writerow(target)
    return path


def stage1_dm_min(influx_df, obs_dir, targets_file=TARGETS_FILE):
    """
    Stage 1 helper: picks heimdall's lower DM limit for one file and records
    the matched targets for stage 2. Never raises; on any problem it returns
    the normal limit.
    Args:
        influx_df: The file's Influx pointing table (RA_deg, DEC_deg, DATA_VALID).
        obs_dir: The observation directory (already created).
        targets_file: The target list.

    Returns:
        (dm_min, matched targets)
    """
    if influx_df is None:  # stage 1 run without metadata (-m)
        return DEFAULT_DM_MIN, []
    try:
        valid = influx_df[influx_df["DATA_VALID"] == 1]
        matched = targets_in_beam(valid["RA_deg"].values, valid["DEC_deg"].values,
                                  load_targets(targets_file, search_ratings()))
        if matched:
            write_obs_targets(obs_dir, matched)
            logging.info(
                "Pointed at low-DM target(s) %s", ", ".join("{} (DM {:g})".format(t.name, t.dm) for t in matched)
            )
        return heimdall_dm_min(matched), matched
    except Exception as error:
        logging.warning("Low-DM target check failed (%s); using DM %g", error, DEFAULT_DM_MIN)
        return DEFAULT_DM_MIN, []


def read_obs_targets(obs_dir):
    """
    The targets stage 1 matched for this observation; empty if none (or unreadable).
    """
    path = os.path.join(obs_dir, OBS_TARGETS_NAME)
    if not os.path.isfile(path):
        return []
    try:
        return load_targets(path)
    except Exception as error:
        logging.warning("Cannot read %s (%s); ignoring it", path, error)
        return []


def dm_window_mask(dm, targets, tol_abs=DM_TOL_ABS, tol_frac=DM_TOL_FRAC):
    """
    True where a DM is within the window of any target.
    Args:
        dm: Array-like of candidate DMs.
        targets: Target list.

    Returns:
        A boolean numpy array.
    """
    dm = np.asarray(dm, dtype=float)
    mask = np.zeros(dm.shape, dtype=bool)
    for target in targets:
        mask |= np.abs(dm - target.dm) <= max(tol_abs, tol_frac * target.dm)
    return mask


def window_only_mask(cand_df, targets, normal_mask, max_cands=MAX_WINDOW_CANDS):
    """
    Stage 2 helper: candidates kept only because they fall in a target's DM
    window, i.e. they pass every threshold in NORMAL_MASK except the DM cut.
    Args:
        cand_df: heimdall candidates (snr, dm columns).
        targets: The observation's targets.
        normal_mask: The other thresholds (S/N, members, time, width), without the DM cut.
        max_cands: Keep at most this many, highest S/N first.

    Returns:
        A boolean numpy array aligned with cand_df.
    """
    mask = np.asarray(normal_mask, dtype=bool) & dm_window_mask(cand_df["dm"].values, targets)
    if mask.sum() > max_cands:
        snr = np.where(mask, cand_df["snr"].values.astype(float), -np.inf)
        keep = np.argsort(-snr, kind="mergesort")[:max_cands]
        trimmed = np.zeros(mask.shape, dtype=bool)
        trimmed[keep] = True
        logging.info("Keeping the %d brightest of %d target-window candidates", max_cands, mask.sum())
        mask = trimmed
    return mask


# ---------------------------------------------------------------- target list

def parse_sexagesimal(text, hours):
    match = re.match(r"^\s*([+-]?)(\d+)(?::(\d+(?:\.\d*)?))?(?::(\d+(?:\.\d*)?))?\s*$", text or "")
    if not match:
        return None
    sign, whole, minutes, seconds = match.groups()
    value = float(whole) + float(minutes or 0) / 60 + float(seconds or 0) / 3600
    value *= 15 if hours else 1
    return -value if sign == "-" else value


def targets_from_psrcat(output, min_dec=-46.0, with_p0=False):
    """
    Parses `psrcat -c "jname raj decj dm [p0]" -o short -nohead -nonumber` output.
    With WITH_P0, returns (Target, p0 in s or None) pairs.
    """
    targets = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        ra, dec = parse_sexagesimal(fields[1], True), parse_sexagesimal(fields[2], False)
        try:
            dm = float(fields[3])
        except ValueError:
            continue
        if ra is None or dec is None or dec < min_dec:
            continue
        target = Target(fields[0], round(ra, 6), round(dec, 6), dm)
        if with_p0:
            try:
                p0 = float(fields[4])
            except (IndexError, ValueError):
                p0 = None
            targets.append((target, p0))
        else:
            targets.append(target)
    return targets


ESTIMATES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_pulse_estimates.csv")
ESTIMATE_COLUMNS = ["sp_per_hour_low", "sp_per_hour_mid", "sp_per_hour_high", "single_pulses", "sp_notes"]
TARGET_COLUMNS = list(Target._fields) + ["p0_s", "class", "nanograv"] + ESTIMATE_COLUMNS


def target_rows(pairs):
    """(Target, p0) pairs -> CSV rows with the class and NANOGrav columns."""
    rows = []
    for target, p0 in pairs:
        cls = "msp" if p0 is not None and p0 < MSP_MAX_P0 else "canonical"
        rows.append(list(target) + ["" if p0 is None else p0, cls, "15yr" if target.name in NANOGRAV_15YR else ""])
    rows.sort(key=lambda r: (r[5] != "msp", r[0]))
    return rows


def add_estimates(rows, path=ESTIMATES_FILE):
    """Appends the single-pulse columns from an sp_rates estimates CSV (blank if missing)."""
    estimates = {}
    if path and os.path.isfile(path):
        with open(path) as f:
            estimates = {r["name"]: r for r in csv.DictReader(f)}
    elif path:
        logging.warning("No single-pulse estimates at %s; leaving those columns blank", path)
    missing = [r[0] for r in rows if r[0] not in estimates]
    if estimates and missing:
        logging.warning("No single-pulse estimate for %s", ", ".join(missing))
    return [row + [estimates.get(row[0], {}).get(c, "") for c in ESTIMATE_COLUMNS] for row in rows]


def run_psrcat(psrcat, args):
    return subprocess.run(
        [psrcat, "-c", "jname raj decj dm p0", "-o", "short", "-nohead", "-nonumber"] + list(args),
        stdout=subprocess.PIPE, check=True, universal_newlines=True,
    ).stdout


def make_target_list(out, psrcat="psrcat", max_p0=MSP_MAX_P0, max_dm=20.2, min_dec=-46.0,
                     canonical=CANONICAL_TARGETS, estimates=ESTIMATES_FILE):
    """
    Writes the target list: every MSP below the DM cut, plus the named normal pulsars.
    """
    condition = "p0 < {} && dm < {}".format(max_p0, max_dm)
    pairs = targets_from_psrcat(run_psrcat(psrcat, ["-l", condition]), min_dec, with_p0=True)
    named = targets_from_psrcat(run_psrcat(psrcat, list(canonical)), min_dec, with_p0=True) if canonical else []
    found = {t.name for t, _ in pairs}
    pairs += [pair for pair in named if pair[0].name not in found]
    if canonical and len(named) < len(canonical):
        logging.warning("psrcat found %d of the %d named pulsars", len(named), len(canonical))
    rows = add_estimates(target_rows(pairs), estimates)
    with open(out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        writer.writerows(rows)
    logging.info("Wrote %d targets (%d msp, %d canonical, %d in NANOGrav 15yr) to %s", len(rows),
                 sum(r[5] == "msp" for r in rows), sum(r[5] == "canonical" for r in rows),
                 sum(r[6] == "15yr" for r in rows), out)
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = ArgumentParser(description="Low-DM target list", formatter_class=ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("make", help="write the target list from psrcat",
                          formatter_class=ArgumentDefaultsHelpFormatter)
    make.add_argument("-o", "--out", default=TARGETS_FILE)
    make.add_argument("--psrcat", default="psrcat", help="psrcat program (uses $PSRCAT_FILE)")
    make.add_argument("--max-p0", type=float, default=0.03, help="seconds")
    make.add_argument("--max-dm", type=float, default=20.2)
    make.add_argument("--min-dec", type=float, default=-46.0, help="GBT southern limit")
    make.add_argument("--canonical", nargs="*", default=list(CANONICAL_TARGETS),
                      help="normal pulsars to add by name (default: CANONICAL_TARGETS)")
    make.add_argument("--estimates", default=ESTIMATES_FILE, help="sp_rates single-pulse estimates CSV")
    values = parser.parse_args()
    if values.command == "make":
        make_target_list(values.out, values.psrcat, values.max_p0, values.max_dm, values.min_dec, values.canonical,
                         values.estimates)
    else:
        parser.print_help()
