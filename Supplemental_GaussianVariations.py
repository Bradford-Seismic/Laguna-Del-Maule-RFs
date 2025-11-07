#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 18:33:39 2025

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





# active_dir = 'C:/Users/7418888/Documents/TANGO/Laguna_Del_Maule/SCRIPTING'
active_dir = '/tango/bradford/Laguna_Del_Maule/SCRIPTING'

os.chdir(active_dir)

region = 'LDM'



joints = np.loadtxt(
    '../DATA/MAPPING/{}_Line3.txt'.format(region), delimiter=',')


model_name = 'CCP_LDM_G-2.5_GDW_VELEST-ZR-mod.nc'
model = xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name)
z_min = model.depth.values.min()

fig_y = 11  # height of CCP fig
fig_topo_y = fig_y / 10     # height of topo on CCP fig
aspect = 1        # dimensions of x-axis km to y-axis km in CCP





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
            fig_x, fig_y / 10), frame=['wsNe', 'xa20f10+lProfile Distance (km)', 'ya4f1+e+lElev. (km)'])


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


    return fig



fig = pygmt.Figure()

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




############### Plotting objects ############################
z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)



with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [0, joints_dist[-1],
                   model_slice.depth.values.min(), 0]
    prj = 'X{}/{}'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=[
                'Wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True)
    fig.grdimage(grid=model_slice[metric], cmap=True)

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap=True, position='jLB+w{}/0.5c+o0c/-1c+h'.format(fig_x), frame='xa{}f{}+l{}'.format(
            (sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, 'Amplitude'))

    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20', 'ya20f10g20'])


# Plot topography
fig = plot_topo(fig, joints, fig_x, fig_y)






fig.shift_origin(xshift='{}c'.format(fig_x + 0.5), yshift = '{}c'.format(-fig_y))

model_name = 'CCP_LDM_G-5.0_GDW_VELEST-ZR-mod.nc'
model = xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name)

model_slice = model.interp(longitude=x, latitude=y,
                           method='linear', kwargs={"fill_value": None})
model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})


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




############### Plotting objects ############################
z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)



with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [0, joints_dist[-1],
                   model_slice.depth.values.min(), 0]
    prj = 'X{}/{}'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True)
    fig.grdimage(grid=model_slice[metric], cmap=True)


    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20', 'ya20f10g20'])


# Plot topography
fig = plot_topo(fig, joints, fig_x, fig_y)






fig.shift_origin(xshift='{}c'.format(fig_x + 0.5), yshift = '{}c'.format(-fig_y))

model_name = 'CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc'
model = xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name)

model_slice = model.interp(longitude=x, latitude=y,
                           method='linear', kwargs={"fill_value": None})
model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})


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




############### Plotting objects ############################
z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)



with pygmt.config(FONT="12p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [0, joints_dist[-1],
                   model_slice.depth.values.min(), 0]
    prj = 'X{}/{}'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True)
    fig.grdimage(grid=model_slice[metric], cmap=True)



    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=[
                'wsne', 'xaf100g20', 'ya20f10g20'])


# Plot topography
fig = plot_topo(fig, joints, fig_x, fig_y)






fig.show()

fig.savefig('../FIGURES/CCP_Model_AllG.png', dpi = 300)

