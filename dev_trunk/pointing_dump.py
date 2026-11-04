#!/usr/bin/env python3

__author__ = 'Devansh Agarwal'
__email__  = 'da0017@mix.wvu.edu'


import redis
import json
import pandas as pd
import collections
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
from astropy.time import Time, TimeMJD
from influxdb import InfluxDBClient, DataFrameClient
import time
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from datetime import datetime, timedelta
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)

OLD_RA = OLD_DEC = 0

# redis connection
r = redis.StrictRedis(
    host='serendip6',
    port=6379)

# push stuff to influxdb
with open("config/conf.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)
login_detail = data_loaded['influxdb']
client = DataFrameClient(host=login_detail['host'],
                         port=login_detail['port'],
                         username=login_detail['uname'],
                         password=login_detail['passw'],
                         database=login_detail['db'])
# this will be used for the data validity
def flag_maker(key,val):
    if key == 'SCPROJID' and val != "JMAINT":
        return 1

    elif key == 'ATMTLS' and val == 'TURRET_LOCK_LOCKED':
        return 1

    elif key == 'ATMFBS' and val == 'BOOM_RETRACTED':
        return 1

    elif key == 'ATRXOCTA' and (250 < np.round(val)) | (np.round(val) < 110 ):
        return 1

    else:
        return 0

def data_valid(dict):
    maintainance_flag = flag_maker('SCPROJID', dict['SCPROJID'])
    turret_lock_flag  = flag_maker('ATMTLS', dict['ATMTLS'])
    boom_flag = flag_maker('ATMFBS', dict['ATMFBS'])
    turret_angle_flag = flag_maker('ATRXOCTA', dict['ATRXOCTA'])
    obs_permission = dict['WEBCNTRL']
    if obs_permission == 0:
        logging.info('Observer does not permitt data')
    if turret_angle_flag == 0:
        logging.debug('Unfavorable turret position')
    return maintainance_flag and turret_lock_flag and boom_flag and obs_permission and turret_angle_flag


# Ron's corrections, check GBT memo for commensal observations. The memo's offsets are where to point the
# telescope to put a source in the L-band beam, so the beam itself is on the other side of the pointing, and
# the azimuth offset is an angle on the sky, so it grows as 1/cos(el); known pulsar detections at the other
# turret positions confirm both (greenburst_db analysis/pointing_correction_validation.md). Near the zenith
# cos(el) is held at its value at 89.5 deg.
def pointing_corr(turret_angle,tel_az,tel_el):
    cos_el = np.maximum(np.cos(tel_el*np.pi/180), np.cos(89.5*np.pi/180))
    Az = (25.18/60)*np.sin(turret_angle*np.pi/180.0)/cos_el + tel_az
    El = (23.30/60)*(1 - np.cos(turret_angle*np.pi/180)) + tel_el
    return El, Az


# we want these keys
'''
MJD:        MJD
UTC:        UTC
IFV1TNCI:   Rcvr in use?
SCPROJID:   Project ID
ATMTLS:     Turret lock
ATMFBS:     Boom status
ATRXOCTA:   Turret Angle for L band.
ATAZIND:    Az.
ATELIND:    El.
WEBCNTRL:   Has observer allowed us to get a copy?
'''

# data types for key and values
f=float
integer=int
s=str
val_type=[f,s,s,f,s,s,f,f,f,f,s,integer]

# keys
names=['MJD','UTC','ATRECVR', 'ATRXOCTA', 'SCPROJID', 'ATMTLS', 'ATAZIND', 'ATELIND','ATMFBS','WEBCNTRL']
influx_names=['MJD','UTC','IFV1TNCI', 'ATRXOCTA', 'SCPROJID', 'ATMTLS', 'ATAZIND', 'AZMJD','ATELIND','ELMJD','ATMFBS','WEBCNTRL']
def get_pipe():
    pipe =  r.pipeline()
    
    # redis pipeline
    for name in names:
        if name == 'WEBCNTRL':
            pipe.get('WEBCNTRL')
        elif name == 'ATAZIND' or name == 'ATELIND':
            pipe.hmget(name,'VALUE')
            pipe.hmget(name,'MJD')
        else:
            pipe.hmget(name,'VALUE')
    return pipe

def main():
    global OLD_RA
    global OLD_DEC
    pipe=get_pipe()
    value=pipe.execute()
    
    # append in list
    #value=list(map(lambda x: x[0].decode(),value))
    
    telescope_status= collections.OrderedDict()
    telescope_status['tags']='gbt'
    # if float make float
    for i,val in enumerate(value):
        if val is None and influx_names[i] == 'WEBCNTRL':
            telescope_status[f'{influx_names[i]}'] = int(1)
        else:
            if val_type[i] == f:
                telescope_status[f'{influx_names[i]}'] = float(val[0].decode())
            elif val_type[i] == integer:
                telescope_status[f'{influx_names[i]}'] = int(val)
            else:
                telescope_status[f'{influx_names[i]}'] = str(val[0].decode())

    # the Alt Az values are corrected for the turret location
    alt,az=pointing_corr(telescope_status['ATRXOCTA'],telescope_status['ATAZIND'],telescope_status['ATELIND'])
    assert telescope_status['AZMJD'] == telescope_status['ELMJD']
    #assert abs(telescope_status['AZMJD'] - telescope_status['ELMJD']) < 2/(3600*24)
    
    # Use the redis time to get the RA DEC from the alt, az values
    telescope_status['MJD'] = float(telescope_status['AZMJD'])
    time=Time(telescope_status['AZMJD'],format='mjd')
    location=EarthLocation(lat=38.4331294*u.deg, lon=-79.8398397*u.deg, height=824.36*u.m)
    obj=AltAz(obstime=time,location=location,az=az*u.deg, alt=alt*u.deg)
    c=SkyCoord(obj).transform_to('fk5')
    
    #add few more values to the dict
    telescope_status['DATA_VALID'] = data_valid(telescope_status)
    telescope_status['RA_deg']=c.ra.deg
    telescope_status['DEC_deg']=c.dec.deg
    telescope_status['RA_drift'] = c.ra.deg - OLD_RA
    OLD_RA = c.ra.deg
    telescope_status['DEC_drift'] = c.dec.deg - OLD_DEC
    OLD_DEC = c.dec.deg
    telescope_status['GL_deg']=c.galactic.l.deg
    telescope_status['GB_deg']=c.galactic.b.deg
    telescope_status['ELCORR']=alt
    telescope_status['AZCORR']=az
    df=pd.DataFrame(telescope_status,index=[pd.to_datetime(Time(telescope_status['MJD'],format='mjd').iso)])
    logging.debug(df.values.tolist())
    val=client.write_points(df,measurement='telescope',time_precision='n')
    for key in df.keys():
        logging.debug(f'{key}, {df[key]}')
    #logging.info(val)
    return val

if __name__ == '__main__':
    main()
    scheduler = BackgroundScheduler()
    sleep_time=int(5*3600)
    scheduler.add_job(main, 'interval', seconds=1,start_date=datetime.now(),end_date=datetime.now()+timedelta(seconds=sleep_time))
    scheduler.start()

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(sleep_time) # do this for 5 min then break
            break
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
