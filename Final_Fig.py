#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:01:16 2025

@author: bradford
"""


import os
import sys
import io


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import pygmt
import math
from math import radians, sin, cos, sqrt, atan2
import xarray as xr
import datetime


import obspy
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.interpolate import interpn
import geopandas as gpd
import shapely
from pyproj import CRS, Transformer


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors


active_dir = ''

os.chdir(active_dir)

region = 'LDM'



joints = np.loadtxt(
    '../DATA/MAPPING/{}_Line3.txt'.format(region), delimiter=',')


model_name = 'CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc'
model = xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name)
z_min = model.depth.values.min()

fig_y = 11  # height of CCP fig
fig_topo_y = fig_y / 10     # height of topo on CCP fig
aspect = 1        # dimensions of x-axis km to y-axis km in CCP

eq_range = 15#model.attrs['Bin Min Smoothing']
eq_xyz = pd.read_csv('../DATA/MAPPING/catalog_zr+tango.csv')
o_time = ['{}:{}:{}T{}:{}:{}'.format(eq_xyz.year[i], eq_xyz.month[i], eq_xyz.day[i],
                                     eq_xyz.hour[i], eq_xyz.minute[i], eq_xyz.second[i]) for i in range(len(eq_xyz))]
eq_xyz['O_Time'] = o_time



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
        d = np.append(d, haversine(
            joints[i, 1], joints[i, 0], joints[i+1, 1], joints[i+1, 0]))
        num_points = np.append(num_points, np.round(d[-1] * points_per_km))

    joints_dist = np.cumsum(d)

    # lat-lon coords of line
    line = np.array([0, 0])  # holding values
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))

        if i == 0:  # append all values for first iteration
            line = np.vstack((line, l))

        else:  # append only values after the first so we don't duplicate values
            line = np.vstack((line, l[1:len(l)]))

    line = line[1:len(line)]  # remove holding value

    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(
            line[i, 1], line[i, 0], line[i+1, 1], line[i+1, 0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist

def plot_topo(fig, joints, fig_x, fig_y, Numerals='ABCDEFG'):

    joints_dist, line, line_dist = Return_Line_Elements(joints)

    fig.shift_origin(yshift='+h')
    with pygmt.config(FONT='12p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p/28p"):
        fig.basemap(region=[0, joints_dist[-1], 0, 5], projection='X{}c/{}c'.format(
            fig_x, fig_y / 10), frame=['wsnE', 'xa20f10+lProfile Distance (km)', 'ya4f1+e+lElev. (km)'])


        topo_x = np.append(line_dist[0], line_dist)
        topo_x = np.append(topo_x, line_dist[-1])


        # on-line topography

        x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
        y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')

        dem = grid.interp(lat=y, lon=x, method='linear')
        dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

        topo = np.append(0, dem.values)
        topo = np.append(topo, 0)

        fig.plot(x=topo_x, y=topo/1000, fill='lightgray', pen='0.5p,black')




        # project volcanoes, repeat for PE volcanoes
        try:
            volcanoes = pd.read_csv(
                '../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
            volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                     'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
            used_data = []
            for i in range(len(joints) - 1):
                track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[
                                      -12, 12], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
                track = track.rename(columns={
                                     0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'volc_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

                if len(track) > 0:
                    try:
                        data_used_check = [track.volc_id[j]
                                           in used_data for j in range(len(track))]
                        data_used_check = [not (bool(data_used_check[j]))
                                           for j in range(len(data_used_check))]
                        track = track[data_used_check].reset_index(drop=True)
                        used_data = np.append(
                            used_data, np.unique(track.volc_id.values))

                        volc_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                            track.longitude, dims='Distance'), method='linear')

                        fig.plot(x=track.p + joints_dist[i], y=volc_elev_interp.values/1000,
                                 style='kvolcano/0.6c', fill='orange', pen='0.3,black')
                    except:
                        pass
        except:
            pass




        # project volcanoes
        try:
            volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
            volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                     'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
            used_data = []
            for i in range(len(joints) - 1):

                # note the little buffer of on the 'length' parameter, for some reason, some lines if the vertex is
                # right on the point, it won't project
                track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[-12, 12], length=[
                                      0, joints_dist[i+1] - joints_dist[i]], unit=True)
                track = track.rename(columns={
                                     0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'volc_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

                if len(track) > 0:
                    data_used_check = [track.volc_id[j]
                                       in used_data for j in range(len(track))]
                    data_used_check = [not (bool(data_used_check[j]))
                                       for j in range(len(data_used_check))]
                    track = track[data_used_check].reset_index(drop=True)
                    used_data = np.append(
                        used_data, np.unique(track.volc_id.values))

                    volc_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                        track.longitude, dims='Distance'), method='linear')

                    fig.plot(x=track.p + joints_dist[i], y=volc_elev_interp.values/1000,
                             style='kvolcano/0.6c', fill='red', pen='0.3,black')
        except:
            pass

        # project stations
        codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i])
                 for i in range(len(stat_table))]
        stat_table['code'] = codes

        stat_data = pd.DataFrame({'longitude': stat_table.stlo.values, 'latitude': stat_table.stla.values,
                                 'elevation': stat_table.stel.values, 'code': stat_table.code.values})
        used_data = []
        for i in range(len(joints) - 1):
            track = pygmt.project(data=stat_data, center=joints[i], endpoint=joints[i+1], width=[
                                  -12, 12], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
            track = track.rename(columns={
                                 0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'p', 4: 'q', 5: 'r', 6: 's', 7: 'code'})

            if len(track) > 0:
                data_used_check = [track.code[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(used_data, np.unique(track.code.values))

                stat_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                    track.longitude, dims='Distance'), method='linear')

                fig.plot(x=track.p + joints_dist[i], y=stat_elev_interp.values /
                         1000, style='i0.3c', fill='gold', pen='0.3,black')

                try:
                    fig.plot(x=track.p[track.code.str.contains('ZR')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                        'ZR')]/1000, style='i0.3c', fill='lightred', pen='0.3,black')
                except:
                    pass

                try:
                    fig.plot(x=track.p[track.code.str.contains('1X')] + joints_dist[i],
                             y=stat_elev_interp.values[track.code.str.contains('1X')]/1000, style='i0.3c', fill='cyan', pen='0.3,black')
                except:
                    pass
                try:
                    fig.plot(x=track.p[track.code.str.contains('XM')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                        'XM')]/1000, style='i0.3c', fill='magenta', pen='0.3,black')
                except:
                    pass


        fig.text(x=joints_dist, y=np.ones(len(joints_dist)) * 4.9, text=list(Numerals[0:len(
            joints_dist)]), font='12p, Helvetica-Bold', fill='white', pen='0.3p,black', no_clip=True)


        # GPS station MAU2 is approximately at joint B, it's position lies outside of
        # pygmt.project range, so we have to add it manually
        gps_x = -70.533; gps_y = -36.063
        gps_elev = grid.interp(lat=gps_y, lon=gps_x,method='linear').values / 1000

        fig.plot(x = joints_dist[1]+1, y = gps_elev, style = 's0.4c', fill = 'purple', pen = '0.6p,black')

    return fig



fig = pygmt.Figure()



###### Subplot 1 GPS detrended

gps = 'MAU2_corr_relNED.txt'
gps = pd.read_csv('../DATA/MAPPING/'+gps, delim_whitespace = True, names = ['time', 'N', 'E', 'D'])


# Fit linear trend
m, b = np.polyfit(gps.time[(gps.time>2015) & (gps.time<2020)], gps.D[(gps.time>2015) & (gps.time<2020)], 1)

# Compute detrended values
gps["D_detrend"] = gps.D - (m * gps.time + b)


##### fit polyline to detrended data

# Fit a 3rd-degree polynomial
coeffs = np.polyfit(gps.time[(gps.time>2015) & (gps.time<2020)], gps.D_detrend[(gps.time>2015) & (gps.time<2020)], 3)

# Create the polynomial function
poly = np.poly1d(coeffs)

# Generate smooth line for plotting
gps_x_fit = np.linspace(gps.time.min(), gps.time.max(), 200)
gps_y_fit = poly(gps_x_fit)





fig_x2 = 12
fig_y2 = fig_x2 / 2.7


with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p", MAP_GRID_PEN="0.01p", MAP_FRAME_PEN="0.25p"):
    prj = 'X{}/{}'.format(fig_x2, fig_y2)
    fig.basemap(region=[2015, 2020, -0.09, 0.11], projection=prj, frame=[
                'WSne', 'xa1f0.5g1', 'ya0.05f0.05g0.05+lDetrended Displacement (m)'])

    fig.plot(x = gps.time, y = gps.D_detrend, style = 'c0.05c', pen = '0.01c,black', fill = 'black')
    fig.plot(x = [2018.37, 2018.37], y = [-0.09, 0.11], pen = '1p,darkgreen,--')
    fig.plot(x = [2018.125, 2018.125], y = [-0.09, 0.11], pen = '1p,red,--')


##### Subplot 3

fig.shift_origin(yshift = '{}c'.format(fig_y2+0.5))

fig_y3 = fig_y - fig_y2-0.5 + fig_topo_y
fig_x3 = fig_x2

with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p", MAP_GRID_PEN="0.01p", MAP_FRAME_PEN="0.25p"):
    prj = 'X{}/{}'.format(fig_x3, fig_y3)
    fig.basemap(region=[2015, 2020, 0.5, 1.9], projection=prj, frame=[
                'Wsn', 'xa1f0.5g1', 'ya0.2f0.2g0.2+lVertical Displacement (m)'])

    fig.plot(x = gps.time, y = gps.D, style = 'c0.05c', pen = '0.01c,black', fill = 'black')
    fig.plot(x = [2018.37, 2018.37], y = [0.5, 1.9], pen = '1p,darkgreen,--')
    # fig.plot(x = [2018.125, 2018.125], y = [0.5, 1.9], pen = '1p,red,--')



##### Subplot 4

dat = eq_xyz[eq_xyz.depth > 10]

dat["utc"] = dat["O_Time"].apply(utc)

# Compute decimal year
def to_decimal_year(UTC):
    year_start = utc(UTC.year, 1, 1)
    next_year_start = utc(UTC.year + 1, 1, 1)
    year_length = next_year_start - year_start
    elapsed = UTC - year_start
    return UTC.year + elapsed / year_length

dat["decimal_year"] = dat["utc"].apply(to_decimal_year)



with pygmt.config(FONT="12p,red", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p", MAP_GRID_PEN="0.01p", MAP_FRAME_PEN="0.25p"):
    prj = 'X{}/{}'.format(fig_x3, fig_y3)

    fig.histogram(
        data=dat.decimal_year,
        region=[2015, 2020, 0, 325],
        projection=prj,
        frame=['E', 'x', 'ya50+lNumber of Events'],
        series=0.05,
        pen="0.25p,black",
        fill="red"
    )




# configure legend
spec = io.StringIO(
"""
N 1

S 0.1c c 0.1c black 0.05p,black 0.65c GPS DATA at MAU2 Station
S 0.1c r 0.6/0.25 red 0.25p,black 0.65c Deep Events Histogram


""".format(eq_range)
    )


# plot legend
with pygmt.config(FONT_ANNOT_PRIMARY="10p"):
    fig.legend(spec=spec,  position='jTL+o0.25c/0.1c', box="+gwhite")




### Construct line and grid area ############################
stat_table = pd.read_csv(
    '../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))

edge_round = 1
edge_gap = 0.1
north = np.ceil(stat_table.stla.max()*edge_round)/edge_round + edge_gap
south = np.floor(stat_table.stla.min()*edge_round)/edge_round - edge_gap

east = np.ceil(stat_table.stlo.max() * edge_round) / edge_round + edge_gap
west = np.floor(stat_table.stlo.min() * edge_round) / edge_round - edge_gap


gmt_region = '{}/{}/{}/{}'.format(west, east, south, north)
grid = pygmt.datasets.load_earth_relief(resolution="01m", region=gmt_region)

joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')


### Interpolate data grids ############################

# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 20
std_hits_limit = 2
std_limit = 0.5


model_slice = model.interp(longitude=x, latitude=y,
                           method='linear', kwargs={"fill_value": None})
model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

# truncate model_slice values to be within scale range
model_slice['Amplitude'].values = np.clip(
    model_slice['Amplitude'].values, -0.5, 0.5)

# nan grid space with bins less than bin hit limit
model_slice['Amplitude'].values[np.where(
    model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
model_slice['Amplitude'].values[np.where(
    model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
model_slice['Amplitude'].values[np.where(
    model_slice['Bootstrap_std'] > std_limit)] = np.nan




############### Plotting objects ############################
z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)


fig.shift_origin(xshift = '{}c'.format(fig_x2+2.5), yshift = '-{}c'.format(fig_y2+0.5))


with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [0, joints_dist[-1],
                   model_slice.depth.values.min(), 0]
    prj = 'X{}/{}'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wSnE', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[-0.5, 0.5], continuous=True)
    fig.grdimage(grid=model_slice['Amplitude'], cmap=True, transparency = 50)


    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20', 'ya20f10g20'])


# configure legend
spec = io.StringIO(
        """
N 3
S 0.20c i 0.4c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.4c cyan 0.6p,black 0.65c TANGO Node
S 0.20c i 0.4c lightred 0.6p,black 0.65c ZR Network


S 0.2c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano
S 0.2c s 0.4c purple 0.6p,black 0.65c GPS station MAU2
S 0.2c c 0.2c green 0.5p,black 0.65c Hypocenter

G 0.07c
G 0.07c

""".format(eq_range)
    )


# plot legend
with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec=spec,  position='jBL+w14c+o{}c/-3.5c'.format(-2.5))


# Plot topography
fig = plot_topo(fig, joints, fig_x, fig_y)








fig.show()




fig.savefig('../FIGURES/Summary_Fig.png', dpi = 300)


#%%

fig_y = 11
aspect = 1        # dimensions of x-axis km to y-axis km


############### Plotting objects ############################
z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)


fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [0, joints_dist[-1],
                   model_slice.depth.values.min(), 0]
    prj = 'X{}/{}'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wSnE', 'xaf100+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    # grid_color = pygmt.makecpt(
    #     cmap='../DATA/MAPPING/no_green.cpt',   series=[-0.5, 0.5], continuous=True)
    # fig.grdimage(grid=model_slice['Amplitude'], cmap=True, transparency = 50)

    # with pygmt.config(FONT_ANNOT_PRIMARY="14p", FONT_LABEL="14p"):
    #     fig.colorbar(cmap=True, position='jLB+w{}/0.3c+o0c/-.75c+h'.format(fig_x),
    #                  frame='xa{}f{}+l{}'.format((0.5 - -0.5) / 2, (0.5 - -0.5) / 5, 'Amplitude'))

    # Plot Hypocenters
    eq_data = pd.DataFrame({'longitude': eq_xyz.longitude.values, 'latitude': eq_xyz.latitude.values,
                           'depth': eq_xyz.depth.values, 'eq_id': eq_xyz.O_Time.values})
    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -eq_range, eq_range], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'p', 4: 'q', 5: 'r', 6: 's', 7: 'eq_id'})

        if len(track) > 0:
            data_used_check = [track.eq_id[j]
                               in used_data for j in range(len(track))]
            data_used_check = [not (bool(data_used_check[j]))
                               for j in range(len(data_used_check))]
            track = track[data_used_check].reset_index(drop=True)
            used_data = np.append(used_data, np.unique(track.eq_id.values))

            fig.plot(x=track.p + joints_dist[i], y=-track.depth,
                     style='c0.1c', fill='green', pen='0.5p,black')





fig.show()


fig.savefig('../FIGURES/Suumary_Fig_EQ-Frame.png', dpi = 300, transparent = True)
