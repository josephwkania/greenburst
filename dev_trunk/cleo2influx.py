#!/usr/bin/env python3
"""
Writes the telescope's pointing to Influx straight from Cleo, without serendip6's Redis: reads Cleo's SETI
server (SetiServer.tcl) and writes the same points as pointing_dump.py, apart from WEBCNTRL, which is
SERENDIP VI's on/off switch on serendip6 and not in Cleo.

The server sends everything it knows when a client connects, then a line for each change,
    <MJD> <Manager>,<field> <value>
with Cleo's timestamp of the value, and "<MJD> Thump Thump" every minute. The telescope position (ccuData)
comes once a second. A client only reads: it must never send anything to the server.

It also keeps the last two minutes of beam positions in a small file (--pointing-file), one line a second,
for udp2fil to put the position at the start of each filterbank file in its header:
    # MJD RA_deg DEC_deg AZCORR ELCORR DATA_VALID ...
    61348.67761460 220.473365 46.399251 45.894791 77.799158 1
The file is replaced whole each second, so a reader never sees half of it.

Run from dev_trunk. It reads config/conf.yaml, which also gives the server's address unless --host and
--port do:
    cleo:
      host: <the SETI server's host>
      port: <its port>
    ./cleo2influx.py --measurement telescope_cleo
    ./cleo2influx.py --dry-run          # print the points instead of writing them or the file
"""

import argparse
import collections
import json
import logging
import os
import socket
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Influx field: the SETI server's key
KEYS = {
    "ATAZIND": "AntennaManager,ccuData,Az,indicated",
    "ATELIND": "AntennaManager,ccuData,El,indicated",
    "ATRXOCTA": "Antenna,rxOpticsConfig,turretAngle",
    "IFV1TNCI": "Antenna,receiver",
    "SCPROJID": "ScanCoordinator,projectId",
    "ATMTLS": "AntennaManager,turretLockState",
    "ATMFBS": "AntennaManager,feedBoomState",
}
# Not written by pointing_dump: what the telescope is doing
SCAN_KEYS = {
    "SCSTATE": "ScanCoordinator,state",
    "SCSCAN": "ScanCoordinator,scanNumber",
    "SCSOURCE": "ScanCoordinator,source",
    "ATEXECST": "AntennaManager,executorState",
}
# The server's connections to the managers the values come from; while one is down, its values are stale
CONNECTIONS = [
    "Antenna,Antenna,ConnectionToManager",
    "AntennaManager,AntennaManager,ConnectionToManager",
    "AntennaManager,AntennaManager,ConnectionToTransporter",
    "AntennaManager,AntennaManager,ConnectionToAccessor",
    "ScanCoordinator,ScanCoordinator,ConnectionToManager",
]


# The same as pointing_dump.py's
def flag_maker(key, val):
    if key == "SCPROJID" and val != "JMAINT":
        return 1

    elif key == "ATMTLS" and val == "TURRET_LOCK_LOCKED":
        return 1

    elif key == "ATMFBS" and val == "BOOM_RETRACTED":
        return 1

    elif key == "ATRXOCTA" and (250 < np.round(val)) | (np.round(val) < 110):
        return 1

    else:
        return 0


# The same as pointing_dump.py's; see the comment there
def pointing_corr(turret_angle, tel_az, tel_el):
    cos_el = np.maximum(np.cos(tel_el * np.pi / 180), np.cos(89.5 * np.pi / 180))
    Az = (25.18 / 60) * np.sin(turret_angle * np.pi / 180.0) / cos_el + tel_az
    El = (23.30 / 60) * (1 - np.cos(turret_angle * np.pi / 180)) + tel_el
    return El, Az


def sky(mjd, az, el):
    """RA and Dec (FK5) and Galactic l and b of an azimuth and elevation, as pointing_dump.py calculates them."""
    # Imported here so that the rest can be tested without astropy
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    location = EarthLocation(
        lat=38.4331294 * u.deg, lon=-79.8398397 * u.deg, height=824.36 * u.m
    )
    obj = AltAz(
        obstime=Time(mjd, format="mjd"),
        location=location,
        az=az * u.deg,
        alt=el * u.deg,
    )
    c = SkyCoord(obj).transform_to("fk5")
    return c.ra.deg, c.dec.deg, c.galactic.l.deg, c.galactic.b.deg


def parse(line):
    """'<MJD> <key> <value>' -> (MJD, key, value); None for heartbeats, error messages and anything else."""
    parts = line.strip().split(None, 2)
    if len(parts) < 2 or "," not in parts[1]:
        return None
    try:
        mjd = float(parts[0])
    except ValueError:
        return None
    return mjd, parts[1], parts[2] if len(parts) == 3 else ""


def influx_time(mjd):
    """Nanoseconds since 1970 for an MJD (UTC), to the millisecond as pointing_dump writes them."""
    return int(round((mjd - 40587) * 86400 * 1000)) * 1000000


def utc(mjd):
    return time.strftime("%H:%M:%S", time.gmtime((mjd - 40587) * 86400))


class Telescope:
    """The server's latest values; update() gives an Influx point for each new telescope position."""

    def __init__(self, sky=sky):
        self.sky = sky
        self.values, self.times = {}, {}
        self.last = None  # (MJD, RA, Dec) of the last point

    def reset(self):
        """After a reconnection: forget the values (the server sends them again) but not the last point."""
        self.values.clear()
        self.times.clear()

    def update(self, line):
        parsed = parse(line)
        if parsed is None:
            return None
        mjd, key, value = parsed
        self.values[key], self.times[key] = value, mjd

        az, el = KEYS["ATAZIND"], KEYS["ATELIND"]
        if key not in (az, el) or self.times.get(az) != self.times.get(el):
            return None
        if self.last is not None and mjd <= self.last[0]:
            return None
        return self.point(mjd)

    def point(self, mjd):
        missing = [k for k in list(KEYS.values()) + CONNECTIONS if k not in self.values]
        if missing:
            # Only while the server's dump after connecting is still arriving
            logging.debug(f"No point at {mjd}: no value yet for {missing}")
            return None
        v = {field: self.values[key] for field, key in KEYS.items()}
        try:
            az, el, turret = (
                float(v["ATAZIND"]),
                float(v["ATELIND"]),
                float(v["ATRXOCTA"]),
            )
        except ValueError:
            logging.warning(f"No point at {mjd}: {v}")
            return None

        el_corr, az_corr = pointing_corr(turret, az, el)
        ra, dec, gl, gb = (float(x) for x in self.sky(mjd, az_corr, el_corr))
        valid = all(self.values[c] == "up" for c in CONNECTIONS) and all(
            flag_maker(f, turret if f == "ATRXOCTA" else v[f])
            for f in ("SCPROJID", "ATMTLS", "ATMFBS", "ATRXOCTA")
        )
        fields = {
            "tags": "gbt",
            "MJD": mjd,
            "UTC": utc(mjd),
            "IFV1TNCI": v["IFV1TNCI"],
            "ATRXOCTA": turret,
            "SCPROJID": v["SCPROJID"],
            "ATMTLS": v["ATMTLS"],
            "ATAZIND": az,
            "AZMJD": mjd,
            "ATELIND": el,
            "ELMJD": mjd,
            "ATMFBS": v["ATMFBS"],
            "DATA_VALID": int(valid),
            "RA_deg": ra,
            "DEC_deg": dec,
            "RA_drift": ra - self.last[1] if self.last else 0.0,
            "DEC_drift": dec - self.last[2] if self.last else 0.0,
            "GL_deg": gl,
            "GB_deg": gb,
            "ELCORR": float(el_corr),
            "AZCORR": float(az_corr),
        }
        for field, key in SCAN_KEYS.items():
            if key in self.values:
                fields[field] = self.values[key]
        self.last = (mjd, ra, dec)
        return {"time": influx_time(mjd), "fields": fields}


class PointingFile:
    """The last two minutes of beam positions, for udp2fil: one line a second, oldest first."""

    HEADER = (
        "# MJD RA_deg DEC_deg AZCORR ELCORR DATA_VALID: the GREENBURST beam, "
        "RA and Dec FK5 J2000, from cleo2influx.py\n"
    )

    def __init__(self, path, seconds=120):
        self.path, self.seconds = path, seconds
        self.lines = collections.deque()  # (MJD, line)
        self.failing = False

    def add(self, point):
        f = point["fields"]
        self.lines.append(
            (
                f["MJD"],
                "%.8f %.6f %.6f %.6f %.6f %d\n"
                % (
                    f["MJD"],
                    f["RA_deg"],
                    f["DEC_deg"],
                    f["AZCORR"],
                    f["ELCORR"],
                    f["DATA_VALID"],
                ),
            )
        )
        # Half a second more for the rounding of Cleo's MJDs
        while self.lines[0][0] < f["MJD"] - (self.seconds + 0.5) / 86400:
            self.lines.popleft()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as out:
                out.write(self.HEADER)
                out.writelines(line for _, line in self.lines)
            os.replace(tmp, self.path)
        except OSError as e:
            if not self.failing:
                logging.warning(f"Could not write {self.path}: {e!r}")
            self.failing = True
        else:
            if self.failing:
                logging.info(f"Writing {self.path} again")
            self.failing = False


def lines(sock):
    buf = b""
    while True:
        data = sock.recv(65536)
        if not data:
            raise ConnectionError("The server closed the connection")
        *complete, buf = (buf + data).split(b"\n")
        for line in complete:
            yield line.decode("utf-8", "replace")


def run(args, write):
    tel = Telescope()
    while True:
        try:
            # The server sends a heartbeat every minute
            with socket.create_connection((args.host, args.port), timeout=90) as sock:
                logging.info(f"Connected to {args.host}:{args.port}")
                tel.reset()
                for line in lines(sock):
                    if " ERRORMSG " in line:
                        logging.warning(line)
                    point = tel.update(line)
                    if point is not None:
                        write(point)
        except (OSError, ConnectionError) as e:
            logging.warning(f"{e!r}; reconnecting in 10 s")
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host", help="host of the SETI server (default: cleo: host: in --config)"
    )
    parser.add_argument(
        "--port", type=int, help="its port (default: cleo: port: in --config)"
    )
    parser.add_argument("--measurement", default="telescope")
    parser.add_argument("--config", default="config/conf.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the points instead of writing them or the pointing file",
    )
    parser.add_argument(
        "--pointing-file",
        default="/dev/shm/greenburst_pointing",
        help="the last two minutes of beam positions, for udp2fil; '' for none",
    )
    args = parser.parse_args()

    if args.host is None or args.port is None:
        import yaml

        with open(args.config) as stream:
            cleo = yaml.safe_load(stream).get("cleo") or {}
        args.host = args.host or cleo.get("host")
        args.port = args.port or cleo.get("port")
        if not args.host or not args.port:
            parser.error(
                f"no SETI server address: give --host and --port, or cleo: host: and port: in {args.config}"
            )

    pointing_file = (
        PointingFile(args.pointing_file)
        if args.pointing_file and not args.dry_run
        else None
    )

    if args.dry_run:

        def write_point(point):
            print(json.dumps(point), flush=True)

    else:
        import yaml
        from influxdb import InfluxDBClient

        with open(args.config) as stream:
            login = yaml.safe_load(stream)["influxdb"]
        client = InfluxDBClient(
            host=login["host"],
            port=login["port"],
            username=login["uname"],
            password=login["passw"],
            database=login["db"],
        )

        def write_point(point):
            point["measurement"] = args.measurement
            try:
                client.write_points([point], time_precision="n")
            except Exception as e:
                logging.warning(f"Could not write to Influx: {e!r}")

    def write(point):
        if pointing_file is not None:
            pointing_file.add(point)
        write_point(point)

    run(args, write)


if __name__ == "__main__":
    main()
