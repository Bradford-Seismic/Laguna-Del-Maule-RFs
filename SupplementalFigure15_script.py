#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:43:43 2026

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
import shapely
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import radians, sin, cos, sqrt, atan2
import io
from shapely.ops import unary_union
from scipy.spatial import KDTree
from obspy.geodetics import degrees2kilometers
from scipy.signal import hilbert

from scipy.interpolate import interp1d
import gstools

from pyproj import CRS, Transformer


warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)



active_dir = './'

os.chdir(active_dir)

stations = pd.read_csv('../DATA/SPREADSHEETS/LDM_Stations.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)








amplitudes = pd.read_csv('../DATA/MAPPING/LdM_LVZ_TimeDomainPicks.csv')


stat_table = pd.read_csv('../DATA/SPREADSHEETS/LDM_PassingStations.csv')
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes


gmt_region = [-71.05, -70, -36.325, -35.625]
west, east, south, north = gmt_region


fig_x = 15
fig = pygmt.Figure()    # initialize the main map

with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection


    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f0.1g0.1+lLongitude', 'ya0.1f0.1g.1+lLatitude', 'wsne'])

    fig.plot(data = '../DATA/MAPPING/Lagunas.shp', pen = '1p,black', fill = 'skyblue')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')

    cmap = pygmt.makecpt(cmap = 'ibcso', reverse = True, series = [0, 2])
    fig.plot(x = amplitudes.longitude, y = amplitudes.latitude, style = 's0.75c', fill = amplitudes.time, cmap = True )
    with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
        fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.5c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lTime')


    
    # now plot station data
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
    
    
    
    fig.text(x = -70.90, y = -36.25, text = 'Time to LVZ', pen = '1p,black', fill = 'white', font = '14p,Helvetica')
    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f0.1g0.1+lLongitude', 'ya0.1f0.1g.1+lLatitude', 'wsne'])





    fig.shift_origin(xshift = f'{fig_x + 1}c')





    fig.plot(data = '../DATA/MAPPING/Lagunas.shp', pen = '1p,black', fill = 'skyblue')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')


    cmap = pygmt.makecpt(cmap = 'ibcso', series = [-1, 0])
    
    fig.plot(x = amplitudes.longitude, y = amplitudes.latitude, style = 's0.75c', fill = amplitudes.amplitude, cmap = True )
    with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
        fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.5c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lAmplitude')

    
    
    # now plot station data
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
    
    
    fig.text(x = -70.90, y = -36.25, text = 'Amplitude to LVZ', pen = '1p,black', fill = 'white', font = '14p,Helvetica')
    
    
    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f0.1g0.1+lLongitude', 'ya0.1f0.1g.1+lLatitude', 'wsnE'])

    
    
    fig.shift_origin(xshift = f'-{fig_x + 1}c', yshift = f'-{13}c')
    
    
    
    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.5f.25g.5+lLongitude', 'ya0.25f0.25g.5+lLatitude', 'wsne'])
    

    with xr.open_dataset('../DATA/MAPPING/nc_models/LVZ_pick_time.nc') as model:

        cmap = pygmt.makecpt(cmap = 'ibcso', reverse = True, series = [0, 2])
        fig.grdimage(grid = model.time, cmap = True)
        fig.grdcontour(grid = model.time, annotation = "0.25", pen = '0.5p,black')
        with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
            fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.5c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lTime (s)')
    
    
    fig.plot(data = '../DATA/MAPPING/Lagunas.shp', pen = '1p,black')
    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')

    

    # now plot station data
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
    
    fig.text(x = -70.80, y = -36.25, text = 'Time to LVZ - Kriged', pen = '1p,black', fill = 'white', font = '14p,Helvetica')

    
    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f0.1g0.1+lLongitude', 'ya0.1f0.1g.1+lLatitude', 'wSne'])





    fig.shift_origin(xshift = f'{fig_x + 1}c')

    

    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.5f.25g.5+lLongitude', 'ya0.25f0.25g.5+lLatitude', 'wsne'])

    with xr.open_dataset('../DATA/MAPPING/nc_models/LVZ_pick_amplitude.nc') as model:

        cmap = pygmt.makecpt(cmap = 'ibcso', series = [-2, 0])
        fig.grdimage(grid = model.amplitude, cmap = True)
        fig.grdcontour(grid = model.amplitude, annotation = [-0.20], pen = '0.5p,black')
        with pygmt.config(FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p"):
            fig.colorbar(cmap=True, position='jRB+w{}/0.2c+o0.5c/0.2c+h+m'.format(fig_x/2.2), frame='xaf+lAmplitude')



    fig.plot(data = '../DATA/MAPPING/Lagunas.shp', pen = '1p,black')
    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.6c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.6c', fill = 'orange', pen='0.75p,black')


    # now plot station data
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
    
    
    fig.text(x = -70.80, y = -36.25, text = 'Amplitude to LVZ - Kriged', pen = '1p,black', fill = 'white', font = '14p,Helvetica')

    fig.basemap(region=gmt_region, projection=prj, frame=['xa0.2f0.1g0.1+lLongitude', 'ya0.1f0.1g.1+lLatitude', 'wSnE'])





    
spec = io.StringIO(
    """
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.6c TANGO Node
S -0.20c i 0.3c magenta 0.6p,black 0.4c TANGO Broadband
S -0.20c i 0.3c lightred 0.6p,black 0.4c LaMa Broadband
S -0.20c s 0.4c purple 0.6p,black 0.4c GPS station MAU2

# S 0.20c + 0.25c black 0.6p,black 0.6c Pierce Point
S -0.20c kvolcano 0.40c red 0.75p,black 0.4c Holocene Volcano
S -0.20c kvolcano 0.40c orange 0.75p,black 0.4c Pleistocene Volcano
# S -0.20c - 0.4c black 0.8p,black 0.40c Country Border



G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position='jBL+w20c+o-3.2c/-2.5c')


fig.show()

fig.savefig('../FIGURES/SupplementalFigure15_LVZPick.png', dpi = 600)

    
    
    


