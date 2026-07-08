#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:41:54 2026

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
from scipy.interpolate import interp1d



warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


active_dir = './'

region = 'LDM'

os.chdir(active_dir)


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




fig_x = 15
fig = pygmt.Figure()    # initialize the main map

with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection


    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f.2g.2+lLongitude', 'ya0.2f0.2g.2+lLatitude', 'wSne'])


    slice_depth = -4
    

    # overlay CCP depth slice
    bin_hits_limit = 20
    std_hits_limit = 2
    std_limit = 0.5

    model_name = 'CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc'
    with xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name) as model:

        grid_x = np.arange(west, east, 0.01)
        grid_y = np.arange(south, north, 0.01)
    
        model_slice = model.interp(depth = slice_depth, longitude = grid_x, latitude = grid_y, method='linear', kwargs={"fill_value": None})
    
    
        metric = 'Amplitude'
        sca = [-0.5, 0.5]
    
        # truncate model_slice values to be within scale range
        model_slice[metric].values = np.clip(
            model_slice[metric].values, sca[0], sca[1])
    
        # nan grid space with bins less than bin hit limit
        model_slice[metric].values[np.where(
            model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
        model_slice[metric].values[np.where(
            model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
        model_slice[metric].values[np.where(
            model_slice['Bootstrap_std'] > std_limit)] = np.nan
    
        grid_color = pygmt.makecpt(
            cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True, transparency = 50)
        fig.grdimage(grid=model_slice[metric], projection = prj, cmap=True, transparency = 50)
        with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
            fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.2c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lAmplitude')
    
    
        # draw contour within WE-SN range
        SN =  np.loadtxt('../DATA/MAPPING/LDM_Line-SN2.txt', delimiter = ',')
        S = SN[0,1]; N = SN[1,1]
        WE =  np.loadtxt('../DATA/MAPPING/LDM_Line-WE2.txt', delimiter = ',')
        W = WE[0,0]; E = WE[1,0]
    
        fig.grdcontour(grid = model_slice['Amplitude'].where((model_slice['Amplitude'] < 0)
                                                             & (model_slice['longitude'] < E) & (model_slice['longitude'] > W)#
                                                             & (model_slice['latitude'] < N) & (model_slice['latitude'] > S)),
                       annotation = [-0.1], pen = '1p,black,--')
        
    




    Lagunas = gpd.read_file('../DATA/MAPPING/Lagunas.shp')
    fig.plot(data = Lagunas, pen = '1p,black', fill = 'skyblue')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')

    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')

    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')



    try:
        fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='.75p,black')
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

    fig.plot(data = '../DATA/MAPPING/Wespestad_Anomaly_4km.shp', pen = '3p,red,4_2:2p',)
    fig.plot(data = '../DATA/MAPPING/Bai_2020_4km_anomaly.shp', pen = '3p,green,4_2:2p',)
    with xr.open_dataset('../DATA/MAPPING/nc_models/Cordell_2018_MT.nc') as model:
        model_slice = model.interp(depth = 4, method='linear', kwargs={"fill_value": None})
        model_slice.MT.values = np.clip(model_slice.MT.values, 0, 1000)
        
        # cmap = pygmt.makecpt(cmap  = 'jet', reverse = True, log = True, series = [0, 3])
        # fig.grdimage(grid = model_slice.MT, cmap = True)
        fig.grdcontour(grid = model_slice.MT, levels = [10], pen = '3p,orange,4_2:2:p')



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

    joints =  np.loadtxt('../DATA/MAPPING/LDM_Line3.txt', delimiter = ',')
    joints_dist, line, line_dist = Return_Line_Elements(joints)

    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEFGHIJKLMNOP"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')


    ### GPS Stations MAU2
    fig.plot(x = -70.533, y = -36.063, style='s0.4c', fill = 'purple',  pen='0.75p,black')

    ### label for depth
    fig.text(x = -70.975, y = -36.3, text = f'z = {slice_depth}', pen = '1p,black', fill = 'white', font = '16p,Helvetica,bold')




    fig.basemap(map_scale="jTR+o0.75c/0.25c+w20k+u")
    
    
    
    
    
    


    # plot depth slice at 28 km
    
    fig.shift_origin(xshift = f'{fig_x + 1}c')

    slice_depth = -28
    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f.2g.2+lLongitude', 'ya0.2f0.2g.2+lLatitude', 'wSnE'])



    # overlay CCP depth slice
    bin_hits_limit = 20
    std_hits_limit = 2
    std_limit = 0.5

    model_name = 'CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc'
    with xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name) as model:

        grid_x = np.arange(west, east, 0.01)
        grid_y = np.arange(south, north, 0.01)
    
        model_slice = model.interp(depth = slice_depth, longitude = grid_x, latitude = grid_y, method='linear', kwargs={"fill_value": None})
    
    
        metric = 'Amplitude'
        sca = [-0.5, 0.5]
    
        # truncate model_slice values to be within scale range
        model_slice[metric].values = np.clip(
            model_slice[metric].values, sca[0], sca[1])
    
        # nan grid space with bins less than bin hit limit
        model_slice[metric].values[np.where(
            model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
        model_slice[metric].values[np.where(
            model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
        model_slice[metric].values[np.where(
            model_slice['Bootstrap_std'] > std_limit)] = np.nan
    
        grid_color = pygmt.makecpt(
            cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True, transparency = 50)
        fig.grdimage(grid=model_slice[metric], projection = prj, cmap=True, transparency = 50)
        with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
            fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.2c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lAmplitude')
    
    
        # draw contour within WE-SN range
        SN =  np.loadtxt('../DATA/MAPPING/LDM_Line-SN2.txt', delimiter = ',')
        S = SN[0,1]; N = SN[1,1]
        WE =  np.loadtxt('../DATA/MAPPING/LDM_Line-WE2.txt', delimiter = ',')
        W = WE[0,0]; E = WE[1,0]
    
        fig.grdcontour(grid = model_slice['Amplitude'].where((model_slice['Amplitude'] < 0)
                                                             & (model_slice['longitude'] < E) & (model_slice['longitude'] > W)#
                                                             & (model_slice['latitude'] < N) & (model_slice['latitude'] > S)),
                       annotation = [-0.1], pen = '1p,black,--')
   
    



    Lagunas = gpd.read_file('../DATA/MAPPING/Lagunas.shp')
    fig.plot(data = Lagunas, pen = '1p,black', fill = 'skyblue')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')

    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')

    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')



    try:
        fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='.75p,black')
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

    joints =  np.loadtxt('../DATA/MAPPING/LDM_Line3.txt', delimiter = ',')
    joints_dist, line, line_dist = Return_Line_Elements(joints)

    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEFGHIJKLMNOP"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')


    ### GPS Stations MAU2
    fig.plot(x = -70.533, y = -36.063, style='s0.4c', fill = 'purple',  pen='0.75p,black')

    ### label for depth
    fig.text(x = -70.975, y = -36.3, text = f'z = {slice_depth}', pen = '1p,black', fill = 'white', font = '16p,Helvetica,bold')



    fig.basemap(map_scale="jTR+o0.75c/0.25c+w20k+u")




spec = io.StringIO(
    """
N 6
S 0.20c i 0.3c cyan 0.4p,black 0.6c TANGO Node
S -0.20c i 0.3c magenta 0.6p,black 0.4c TANGO Broadband
S -0.20c i 0.3c lightred 0.6p,black 0.4c LaMa Broadband
S -0.20c s 0.4c purple 0.6p,black 0.4c GPS station MAU2

# S 0.20c + 0.25c black 0.6p,black 0.6c Pierce Point
S -0.20c kvolcano 0.40c red 0.75p,black 0.4c Holocene Volcano
S -0.20c kvolcano 0.40c orange 0.75p,black 0.4c Pleistocene Volcano
S -0.20c -- 0.6c red 1p,black 0.40c Wespestad et al., (2019)
S 1c -- 0.6c green 1p,black 1.5c Bai et al., (2020)
S 1c -- 0.6c orange 1p,black 1.5c Cordell et al., (2018)


G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position='jBL+w30c+o-15c/-2c')


fig.show()

fig.savefig('../FIGURES/Fig5_PlanView.png', dpi = 1200)
