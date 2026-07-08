#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:23:41 2026

@author: jimbradford
"""


import numpy as np
import pandas as pd
import os
import sys
import obspy
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy import read_inventory
from obspy.io.sac import SACTrace
from obspy.signal.trigger import recursive_sta_lta
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from glob import glob
import warnings
import matplotlib.dates as dates
import pygmt
import subprocess
import math
import geopandas as gpd
from pyproj import CRS, Transformer

import shapely
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import radians, sin, cos, sqrt, atan2
import io
from shapely.ops import unary_union



warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


active_dir = './'

os.chdir(active_dir)



stations = pd.read_csv('../DATA/SPREADSHEETS/LDM_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'LDM'




def closest_idx(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Earth radius in kilometers (approx.)
    return distance

def Return_Line_Elements(joints):


    points_per_km = 10

    # distance between joint-points
    d = np.array([0])
    num_points = np.array([])
    for i in range(len(joints)-1):
        d = np.append(d, haversine(joints[i,1], joints[i,0],joints[i+1,1], joints[i+1,0]))
        num_points = np.append(num_points, np.round(d[-1] * points_per_km))

    joints_dist = np.cumsum(d)


    # lat-lon coords of line
    line = np.array([0,0])
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))
        line = np.vstack((line, l))

    line = line[1:len(line)]


    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(line[i,1], line[i,0],line[i+1,1], line[i+1,0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist





stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes


event_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingEvents.csv'.format(region))


gmt_region = [-71.05, -70, -36.325, -35.625]
west, east, south, north = gmt_region

joints =  np.loadtxt('../DATA/MAPPING/LDM_Line3.txt', delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)


fig_x = 15

fig = pygmt.Figure()
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):
    
    prj = 'M{}c'.format(fig_x)            # set the map projection


    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.5f.25g.5+lLongitude', 'ya0.25f0.25g.5+lLatitude', 'wSnE'])

    # overlay the grid image DEM over the basemap
    grid = pygmt.datasets.load_earth_relief(resolution="01s", region=gmt_region)
    shade = pygmt.grdgradient(grid=grid, azimuth="315/45", normalize="e1")
    fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 55)


    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    fig.plot(data = '../DATA/MAPPING/Lagunas.shp', pen = '1p,black', fill = 'skyblue')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')

    
    
    # now plot station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
   
   
   
    try:
        # add ZR network stations from file, there are missing stations due to RF QC
        ZR_stations = pd.read_csv('../DATA/MAPPING/ZR_stations_all.txt', delimiter = ' ',  index_col = False, names = ['network', 'station', 'latitude', 'longitude' ])
        fig.plot( x = ZR_stations.longitude, y =  ZR_stations.latitude, style='i0.3c', fill = 'lightred', pen='.75p,black')

        # fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='.75p,black')
    except:
        pass
   
    try:
        fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='.5p,black')
    except:
        pass
   
    try:
        fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='.75,black')
    except:
        pass
    
    
    sub_joints =  np.loadtxt('../DATA/MAPPING/LDM_Line-SN2.txt', delimiter = ',')
    fig.plot(x = sub_joints[:,0], y = sub_joints[:,1], pen = '1.2p,black,--')
    Numerals = ["S", "N"]
    for i in range(len(sub_joints)):
        fig.text(x = sub_joints[i,0], y = sub_joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    sub_joints =  np.loadtxt('../DATA/MAPPING/LDM_Line-WE2.txt', delimiter = ',')
    fig.plot(x = sub_joints[:,0], y = sub_joints[:,1], pen = '1.2p,black,--')
    Numerals = ["W", "E"]
    for i in range(len(sub_joints)):
        fig.text(x = sub_joints[i,0], y = sub_joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')



    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')

    Numerals = "ABCDEFGHIJKLMNOP"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    ### GPS Stations MAU2
    fig.plot(x = -70.533, y = -36.063, style='s0.4c', fill = 'purple',  pen='0.75p,black')


    fig.basemap(map_scale="jTR+o0.75c/0.25c+w20k+u")



spec = io.StringIO(
    """
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.5c i 0.3c lightred 0.6p,black 0.9c LaMa Broadband
S -0.40c s 0.4c purple 0.6p,black 0.0c GPS station MAU2


S 0.20c kvolcano 0.40c red 0.75p,black 0.65c Holocene Volcano
S 0.20c kvolcano 0.40c orange 0.75p,black 0.65c Pleistocene Volcano
S 0.5c f+l+t 0.45c/-1/0.15c black 1.5,black 0.9c Chile Trench
S -0.40c - 0.5 black 1p,black 0.0c Country Border



G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position='jBL+w20c+o-2c/-2.5c')



with fig.inset(position="jBL+w8c/12c+o-4.5c/-1c", box = False):

    inset_region = [-76, -53, -56, -17]

    fig.coast(
       region= inset_region,
       projection="U19S/?",
       dcw="CL,AR+gwhite+p0.5p",
       area_thresh=1000,
       transparency = 0)

    fig.coast(
       region= inset_region,
       projection="U19S/?",
       dcw="CL,AR+gcoral+p0.5p",
       area_thresh=1000,
       transparency = 30)

    
    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')

    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = inset_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')

    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.2c', fill = 'red', pen='0.1p,black')

    # plot the general region we are focused on
    fig.plot(x = [west, west, east, east, west], y = [north, south, south, north, north], pen = '1.25p,cyan')



fig.show()
fig.savefig('../FIGURES/Fig1.png', dpi = 1200)


   
    
    
    



