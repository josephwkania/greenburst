#!/usr/bin/env python3
"""
Limits the Slack plots of pulses from known pulsars.

A FETCH-positive candidate counts as a pulse from a known pulsar when its DM
is within max(DM_TOL_ABS, DM_TOL_FRAC * DM) of a pulsar that is in the beam:
    - a pulsar in bright_pulsars.csv (pulsars with many FETCH-positive pulses
      in the greenburst database) within BRIGHT_RADIUS of a candidate's
      pointing, or
    - a low-DM target that stage 1 matched for this file (low_dm_targets.py).
Each pulsar gets at most MAX_PLOTS_PER_FILE plots (its highest-S/N pulses) in
each of the first MAX_PLOT_FILES files of an observing session, so 4 plots per
visit; a session ends after SESSION_GAP_MIN minutes without a file on that
pulsar. All other pulses from it are counted in one Slack summary per file.
Candidates that are not from a known pulsar go to stage 3 as before. The
session counts are kept in WORK_DIR/pulsar_plot_state.json.

Make the bright pulsar list from the greenburst database (pulsar_lookup.py):
    pulsar_lookup.py --db greenburst.db --all --dm-match --fetch-positive --csv bright
    python3 pulsar_plot_limit.py make-bright --summary bright_summary.csv
"""

import csv
import json
import logging
import os
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from low_dm_targets import DM_TOL_ABS, DM_TOL_FRAC, Target, angular_separation, dm_window_mask, load_targets

BRIGHT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bright_pulsars.csv")
STATE_NAME = "pulsar_plot_state.json"
BRIGHT_RADIUS = 0.25  # deg; bright pulsars are also seen off-axis
MAX_PLOTS_PER_FILE = 2
MAX_PLOT_FILES = 2
SESSION_GAP_MIN = 30.0

H5_PATTERN = re.compile(r"_dm_([0-9.]+)_snr_([0-9.]+)")


def h5_dm_snr(h5_path):
    """(dm, snr) from a candmaker file name, or (None, None)."""
    match = H5_PATTERN.search(os.path.basename(str(h5_path)))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2).rstrip("."))


def pulsars_in_beam(cand_df, pulsars, radius=BRIGHT_RADIUS):
    """
    The pulsars within RADIUS of any candidate pointing (RA_deg/DEC_deg columns).
    """
    if cand_df is None or not pulsars or "RA_deg" not in cand_df or "DEC_deg" not in cand_df:
        return []
    ra = np.asarray(cand_df["RA_deg"].values, dtype=float)
    dec = np.asarray(cand_df["DEC_deg"].values, dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[good], dec[good]
    if ra.size == 0:
        return []
    return [p for p in pulsars if (angular_separation(ra, dec, p.ra_deg, p.dec_deg) <= radius).any()]


def assign_to_pulsars(candidates, pulsars):
    """
    Splits FETCH-positive candidates into pulses from the given pulsars and the rest.
    Args:
        candidates: h5 paths.
        pulsars: Target list of pulsars in the beam.

    Returns:
        (other candidates, {pulsar name: [h5 paths, highest S/N first]})
    """
    others, by_pulsar = [], {}
    for candidate in candidates:
        dm, snr = h5_dm_snr(candidate)
        hits = [p for p in pulsars if dm is not None and dm_window_mask([dm], [p])[0]]
        if not hits:
            others.append(candidate)
            continue
        name = min(hits, key=lambda p: abs(p.dm - dm)).name
        by_pulsar.setdefault(name, []).append((snr or 0.0, candidate))
    return others, {name: [c for _, c in sorted(pairs, key=lambda x: -x[0])] for name, pairs in by_pulsar.items()}


class PlotLimiter:
    """
    Per-pulsar session state: how many files of the current session already
    had plots, and when the pulsar was last seen.
    """

    def __init__(self, path):
        self.path = path
        try:
            with open(path) as f:
                self.state = json.load(f)
        except (OSError, ValueError):
            self.state = {}

    def choose(self, name, candidates, mjd):
        """
        Returns the candidates of this pulsar to plot for a file starting at MJD.
        """
        entry = self.state.get(name)
        if entry is None or mjd - entry["last_mjd"] > SESSION_GAP_MIN / 1440.0 or mjd < entry["last_mjd"] - 1:
            entry = {"files": 0, "last_mjd": mjd}
        chosen = []
        if entry["files"] < MAX_PLOT_FILES and candidates:
            chosen = list(candidates[:MAX_PLOTS_PER_FILE])
            entry["files"] += 1
        entry["last_mjd"] = max(entry["last_mjd"], mjd)
        self.state[name] = entry
        return chosen

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


def select_for_stage3(candidates, low_dm_targets, cand_df, mjd, work_dir, n_window=0, bright_file=BRIGHT_FILE):
    """
    Stage 2 helper: which FETCH-positive candidates to send to stage 3 (plots).
    Never raises; on any problem every candidate is sent, as before.
    Args:
        candidates: h5 paths of FETCH-positive candidates.
        low_dm_targets: Targets stage 1 matched for this file.
        cand_df: The candidates written to the CSV (RA_deg/DEC_deg columns).
        mjd: The file's start MJD.
        work_dir: Where the session state file lives.
        n_window: Candidates kept only for the low-DM targets' DM windows.

    Returns:
        (h5 paths for stage 3, Slack summary text or None)
    """
    candidates = list(candidates)
    try:
        pulsars = {p.name: p for p in pulsars_in_beam(cand_df, load_targets(bright_file))}
        for target in low_dm_targets:
            pulsars.setdefault(target.name, target)
        if not pulsars:
            return candidates, None
        others, by_pulsar = assign_to_pulsars(candidates, list(pulsars.values()))
        limiter = PlotLimiter(os.path.join(work_dir, STATE_NAME))
        plots, parts = [], []
        for name in sorted(by_pulsar):
            chosen = limiter.choose(name, by_pulsar[name], mjd)
            plots.extend(chosen)
            parts.append("{}: {} FETCH-positive, {} plotted".format(name, len(by_pulsar[name]), len(chosen)))
        limiter.save()
        # Low-DM targets are always reported, even without FETCH-positive pulses
        parts.extend("{}: 0 FETCH-positive".format(t.name) for t in low_dm_targets if t.name not in by_pulsar)
        if low_dm_targets:
            parts.append("{} low-DM window cands".format(n_window))
        if not by_pulsar and not low_dm_targets:
            return others, None
        return others + plots, "Known pulsars at MJD {}: {}".format(mjd, "; ".join(parts))
    except Exception as error:
        logging.warning("Pulsar plot limit failed (%s); sending every candidate", error)
        return candidates, None


# ---------------------------------------------------------------- bright list

def same_source(a, b, merge_radius):
    sep = angular_separation(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg)
    if sep <= merge_radius:
        return True
    tol = max(DM_TOL_ABS, DM_TOL_FRAC * a.dm) + max(DM_TOL_ABS, DM_TOL_FRAC * b.dm)
    return sep <= BRIGHT_RADIUS and abs(a.dm - b.dm) <= tol


def make_bright_list(summary_csv, out=BRIGHT_FILE, min_cands=10, merge_radius=0.05):
    """
    Writes the bright pulsar list from a pulsar_lookup.py --all --dm-match
    --fetch-positive summary CSV: pulsars with at least MIN_CANDS
    FETCH-positive DM-matched candidates. A pulsar is left out when a
    brighter one is within MERGE_RADIUS (globular clusters, duplicate catalogue
    entries), or within BRIGHT_RADIUS with an overlapping DM window, since both
    would claim the same pulses; so each source counts once.
    """
    rows = []
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if int(row["dm_match"]) >= min_cands and row.get("ra_deg") and row["dm"]:
                rows.append(row)
    # Brightest first; among equals prefer catalogue entries with a B name
    # (precise, well-known positions)
    rows.sort(key=lambda r: (-int(r["dm_match"]), not r.get("bname"), r["pulsar"]))
    kept = []
    for row in rows:
        ra, dec = float(row["ra_deg"]), float(row["dec_deg"])
        target = Target(row["pulsar"], ra, dec, float(row["dm"]))
        if any(same_source(target, k, merge_radius) for k in kept):
            continue
        kept.append(target)
    with open(out, "w") as f:
        writer = csv.writer(f)
        writer.writerow(Target._fields)
        for target in kept:
            writer.writerow(target)
    logging.info("Wrote %d bright pulsars (of %d with >= %d FETCH-positive pulses) to %s",
                 len(kept), len(rows), min_cands, out)
    return kept


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = ArgumentParser(description="Known-pulsar Slack plot limit", formatter_class=ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("make-bright", help="write bright_pulsars.csv", formatter_class=ArgumentDefaultsHelpFormatter)
    make.add_argument("--summary", required=True, help="pulsar_lookup.py ... --csv PREFIX output PREFIX_summary.csv")
    make.add_argument("-o", "--out", default=BRIGHT_FILE)
    make.add_argument("--min-cands", type=int, default=10, help="FETCH-positive DM-matched candidates")
    values = parser.parse_args()
    if values.command == "make-bright":
        make_bright_list(values.summary, values.out, values.min_cands)
    else:
        parser.print_help()
