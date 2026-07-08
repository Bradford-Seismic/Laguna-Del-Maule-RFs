#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:16:46 2026

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
model = TauPyModel(model="iasp91")



stations = pd.read_csv('../DATA/SPREADSHEETS/LDM_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'LDM'







stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes


event_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingEvents.csv'.format(region))




gmt_region = [-71.05, -70, -36.325, -35.625]
west, east, south, north = gmt_region





### EQ + Rose diagram Figure



save_figs = 0
EQ_fig_name = 'F2_EQ+Rose_Region-{}.png'.format(region)


# produce a gmt map of the accepted events with respect to the phase type
fig = pygmt.Figure()
prj = "A-71.2/-20/100/14c"



fig.coast(projection=prj, region = 'g', frame = 'g60', land="gray", shorelines = '0.5p, black', water = 'white')


fig.plot(x = event_table.evlo[event_table.sense == '/ZR'], y = event_table.evla[event_table.sense == '/ZR'], style='c0.3c', pen='0.5p,black', fill = 'lightred', label = 'LaMa-BB')
fig.plot(x = event_table.evlo[event_table.sense == '/XM'], y = event_table.evla[event_table.sense == '/XM'], style='c0.3c', pen='0.5p,black', fill = 'magenta',label = 'TANGO-BB')
fig.plot(x = event_table.evlo[event_table.sense == '/XN'], y = event_table.evla[event_table.sense == '/XN'], style='c0.3c', pen='0.5p,black', fill = 'cyan',label = 'TANGO-Node')
fig.plot(x = event_table.evlo[event_table.sense == '/XM/XN'], y = event_table.evla[event_table.sense == '/XM/XN'], style='c0.3c', pen='0.5p,black',fill="p9+fcyan+bmagenta",label = 'TANGO-BB & Node')

# plot the general region we are focused on
fig.plot(x = [west, west, east, east, west], y = [north, south, south, north, north], pen = '1p,blue')

with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend( position='jTL+w5.5c+o0c/-.5c',  box='+gwhite+p1p')





# Back-Azimuth Distribution
fig.shift_origin(xshift = '15c', yshift = '11c')
with fig.subplot(nrows=1, ncols =1, figsize = ('4c', '4c')):
    with fig.set_panel(panel = 0):

        baz_hits = []
        baz_bin = np.arange(0, 370, 10)
        for i in range(len(baz_bin-1)):
            hit_count = 0
            for j in event_table.baz[event_table.sense.str.contains('/ZR', na = False)].values:
                if (j > baz_bin[i]) and (j < baz_bin[i+1]):
                    hit_count += 1

            baz_hits.append(hit_count)

        fig.rose(
            # use columns of the sample dataset as input for the length and azimuth
            # parameters
            length = baz_hits,
            azimuth = baz_bin,
            # specify the "region" of interest in the (r,azimuth) space
            # [r0, r1, az0, az1], here, r0 is 0 and r1 is 1, for azimuth, az0 is 0 and
            # az1 is 360 which means we plot a full circle between 0 and 360 degrees
            region=[0, np.max(baz_hits), 0, 360],
            # set the diameter of the rose diagram to 7.5 cm
            diameter="4c",
            # define the sector width in degrees, we append +r here to draw a rose
            # diagram instead of a sector diagram
            sector="10",
            # normalize bin counts by the largest value so all bin counts range from
            # 0 to 1

            # use red3 as color fill for the sectors
            fill="lightred",
            # define the frame with ticks and gridlines every 0.2
            # length unit in radial direction and every 30 degrees
            # in azimuthal direction, set background color to
            # lightgray
            labels = "W,E,S,N",
            frame=["x1g1", "y30g30", '+gwhite'],
            # use a pen size of 1p to draw the outlines
            pen="1p",
            no_scale = True

        )

        fig.text(position = 'BL', text = 'n = {} Events'.format(np.sum(baz_hits)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')


fig.shift_origin(yshift = '-5.5c')
with fig.subplot(nrows=1, ncols =1, figsize = ('4c', '4c')):
    with fig.set_panel(panel = 0):

        baz_hits = []
        baz_bin = np.arange(0, 370, 10)
        for i in range(len(baz_bin-1)):
            hit_count = 0
            for j in event_table.baz[event_table.sense.str.contains('/XM', na = False)].values:
                if (j > baz_bin[i]) and (j < baz_bin[i+1]):
                    hit_count += 1

            baz_hits.append(hit_count)

        fig.rose(
            # use columns of the sample dataset as input for the length and azimuth
            # parameters
            length = baz_hits,
            azimuth = baz_bin,
            # specify the "region" of interest in the (r,azimuth) space
            # [r0, r1, az0, az1], here, r0 is 0 and r1 is 1, for azimuth, az0 is 0 and
            # az1 is 360 which means we plot a full circle between 0 and 360 degrees
            region=[0, np.max(baz_hits), 0, 360],
            # set the diameter of the rose diagram to 7.5 cm
            diameter="4c",
            # define the sector width in degrees, we append +r here to draw a rose
            # diagram instead of a sector diagram
            sector="10",
            # normalize bin counts by the largest value so all bin counts range from
            # 0 to 1

            # use red3 as color fill for the sectors
            fill="magenta",
            # define the frame with ticks and gridlines every 0.2
            # length unit in radial direction and every 30 degrees
            # in azimuthal direction, set background color to
            # lightgray
            labels = "W,E,S,N",
            frame=["x1g1", "y30g30", '+gwhite'],
            # use a pen size of 1p to draw the outlines
            pen="1p",
            no_scale = True

        )

        fig.text(position = 'BL', text = 'n = {} Events'.format(np.sum(baz_hits)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')


fig.shift_origin(yshift = '-5.5c')
with fig.subplot(nrows=1, ncols =1, figsize = ('4c', '4c')):
    with fig.set_panel(panel = 0):

        baz_hits = []
        baz_bin = np.arange(0, 370, 10)
        for i in range(len(baz_bin-1)):
            hit_count = 0
            for j in event_table.baz[event_table.sense.str.contains('/XN', na = False)].values:
                if (j > baz_bin[i]) and (j < baz_bin[i+1]):
                    hit_count += 1

            baz_hits.append(hit_count)

        fig.rose(
            # use columns of the sample dataset as input for the length and azimuth
            # parameters
            length = baz_hits,
            azimuth = baz_bin,
            # specify the "region" of interest in the (r,azimuth) space
            # [r0, r1, az0, az1], here, r0 is 0 and r1 is 1, for azimuth, az0 is 0 and
            # az1 is 360 which means we plot a full circle between 0 and 360 degrees
            region=[0, np.max(baz_hits), 0, 360],
            # set the diameter of the rose diagram to 7.5 cm
            diameter="4c",
            # define the sector width in degrees, we append +r here to draw a rose
            # diagram instead of a sector diagram
            sector="10",
            # normalize bin counts by the largest value so all bin counts range from
            # 0 to 1

            # use red3 as color fill for the sectors
            fill="cyan",
            # define the frame with ticks and gridlines every 0.2
            # length unit in radial direction and every 30 degrees
            # in azimuthal direction, set background color to
            # lightgray
            labels = "W,E,S,N",
            frame=["x1g1", "y30g30", '+gwhite'],
            # use a pen size of 1p to draw the outlines
            pen="1p",
            no_scale = True

        )

    fig.text(position = 'BL', text = 'n = {} Events'.format(np.sum(baz_hits)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')





fig.show()

fig.savefig('../FIGURES/SupplementalFigure9.png', dpi = 600)





