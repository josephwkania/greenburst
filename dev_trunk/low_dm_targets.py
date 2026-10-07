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
            first), and sends one Slack summary for them instead of passing
            each FETCH-positive one to stage 3.
All other files and candidates are processed exactly as before. Any error
here falls back to the normal limits.

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

Target = namedtuple("Target", ["name", "ra_deg", "dec_deg", "dm"])

H5_DM_PATTERN = re.compile(r"_dm_([0-9.]+)")


def load_targets(path=TARGETS_FILE):
    """
    Reads a target list (CSV with name, ra_deg, dec_deg, dm columns).

    Returns:
        A list of Target; empty if the file is missing.
    """
    if not os.path.isfile(path):
        logging.warning("No low-DM target list at %s; using the normal DM limits", path)
        return []
    targets = []
    with open(path) as f:
        for row in csv.DictReader(f):
            targets.append(Target(row["name"], float(row["ra_deg"]), float(row["dec_deg"]), float(row["dm"])))
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
        matched = targets_in_beam(valid["RA_deg"].values, valid["DEC_deg"].values, load_targets(targets_file))
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


def dm_from_h5_name(h5_path):
    match = H5_DM_PATTERN.search(os.path.basename(str(h5_path)))
    return float(match.group(1)) if match else None


def split_for_stage3(candidates, targets, dm_cut):
    """
    Separates FETCH-positive candidates that were only kept for a target's DM
    window (summarised, not sent to stage 3) from the rest (sent as before).
    Args:
        candidates: h5 paths of FETCH-positive candidates.
        targets: The observation's targets.
        dm_cut: stage 2's normal minimum DM.

    Returns:
        (to stage 3, {target name: count of window-only candidates})
    """
    to_stage3, per_target = [], {}
    for candidate in candidates:
        dm = dm_from_h5_name(candidate)
        hits = [t for t in targets if dm is not None and dm <= dm_cut and dm_window_mask([dm], [t])[0]]
        if hits:
            name = min(hits, key=lambda t: abs(t.dm - dm)).name
            per_target[name] = per_target.get(name, 0) + 1
        else:
            to_stage3.append(candidate)
    return to_stage3, per_target


def summary_message(per_target, targets, n_window, mjd):
    if not targets:
        return None
    parts = ["{}: {}".format(t.name, per_target.get(t.name, 0)) for t in targets]
    return "Low-DM targets at MJD {}: {} window cands, FETCH-positive per target {}".format(
        mjd, n_window, ", ".join(parts)
    )


# ---------------------------------------------------------------- target list

def parse_sexagesimal(text, hours):
    match = re.match(r"^\s*([+-]?)(\d+)(?::(\d+(?:\.\d*)?))?(?::(\d+(?:\.\d*)?))?\s*$", text or "")
    if not match:
        return None
    sign, whole, minutes, seconds = match.groups()
    value = float(whole) + float(minutes or 0) / 60 + float(seconds or 0) / 3600
    value *= 15 if hours else 1
    return -value if sign == "-" else value


def targets_from_psrcat(output, min_dec=-46.0):
    """
    Parses `psrcat -c "jname raj decj dm" -o short -nohead -nonumber` output.
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
        targets.append(Target(fields[0], round(ra, 6), round(dec, 6), dm))
    return targets


def make_target_list(out, psrcat="psrcat", max_p0=0.03, max_dm=20.2, min_dec=-46.0):
    condition = "p0 < {} && dm < {}".format(max_p0, max_dm)
    output = subprocess.run(
        [psrcat, "-c", "jname raj decj dm", "-l", condition, "-o", "short", "-nohead", "-nonumber"],
        stdout=subprocess.PIPE, check=True, universal_newlines=True,
    ).stdout
    targets = targets_from_psrcat(output, min_dec)
    with open(out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(Target._fields)
        for target in targets:
            writer.writerow(target)
    logging.info("Wrote %d targets (%s, Dec > %g) to %s", len(targets), condition, min_dec, out)
    return targets


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
    values = parser.parse_args()
    if values.command == "make":
        make_target_list(values.out, values.psrcat, values.max_p0, values.max_dm, values.min_dec)
    else:
        parser.print_help()
