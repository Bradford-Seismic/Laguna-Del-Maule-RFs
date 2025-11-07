#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:58:49 2025

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

itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'LDM'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')

stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)


joints = np.loadtxt(
    '../DATA/MAPPING/{}_Line-WE2.txt'.format(region), delimiter=',')



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


# depth, left dist, right dist

# W-E primaries
primaries = [
            [-9, 3, 17],
            [-5, 19, 30],
            [-3, 21, 33]]

# # SN primaries
# primaries = [
#             [-4, 3, 20],
#             [-5, 20, 35],
#             [-14, 5, 10],
#             ]

joints_dist, line, line_dist = Return_Line_Elements(joints)
x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')


gmt_region = [-71.05, -70, -36.325, -35.625]
grid = pygmt.datasets.load_earth_relief(resolution="01s", region=gmt_region)

fig = pygmt.Figure()
with pygmt.config(FONT = '14p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    # establish figure dimension
    fig_x = 20
    depth = 60
    aspect = 1

    ratio = line_dist.max() / 20 # km per figure cm
    fig_y = depth/ratio*aspect

    total_y = fig_y


    fig.basemap(region = [0, joints_dist[-1], -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa20f10g20', 'ya10f10g20+lDepth'])
    model = xr.open_dataset('../DATA/MAPPING/nc_models/CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc')
    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
    model_slice.Amplitude.values = np.clip(model_slice.Amplitude.values, -0.5, 0.5)


    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[-0.5, 0.5], continuous=True)
    fig.grdimage(grid=model_slice['Amplitude'], cmap=True)




    v_model_1D = 'velmodel_zr-mod.csv'
    v_data = pd.read_csv('../DATA/MAPPING/nc_models/' +
                         v_model_1D,   skiprows=1, header=None)
    vsu = v_data[2].to_numpy()
    vpu = v_data[1].to_numpy()
    vd = v_data[0].to_numpy()

    dzi = 0.5
    z = np.arange(np.round((np.min(vd)) * 1/dzi) * dzi, 100 + dzi, dzi)

    # Interpolate vs and k at mid-points between layers in z using nearest values
    vs_interp_func = interp1d(vd, vsu, kind='nearest',
                              bounds_error=False, fill_value='extrapolate')
    vp_interp_func = interp1d(vd, vpu, kind='nearest',
                              bounds_error=False, fill_value='extrapolate')

    vs = vs_interp_func(z + 0.5*dzi)
    vp = vp_interp_func(z + 0.5*dzi)

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

    p_ref = 0.064
    for primary in primaries:
        x_len = np.arange(primary[1], primary[2], 1)
        fig.plot(x = x_len, y = np.ones(x_len.shape)*primary[0], pen = '2p,white,--')

        topo = dem.interp(Distance_Along_Trend = np.arange(primary[1], primary[2], 1), method = 'nearest').values/1000

        se = np.round(topo / 0.5) * 0.5

        # find bulk velocity of material above primary
        vs_rms = []; vp_rms = []
        for i, x_l in enumerate(x_len):
            vs_rms.append(np.sqrt(np.mean(vs_interp_func(np.arange(-se[i], -primary[0], dzi))**2)))
            vp_rms.append(np.sqrt(np.mean(vp_interp_func(np.arange(-se[i], -primary[0], dzi))**2)))
        vs_rms = np.array(vs_rms); vp_rms = np.array(vp_rms)


        H = topo + -primary[0]  # total thickness from topography to primary

        qa = np.sqrt((1/vs_rms**2) - p_ref**2)
        qb = np.sqrt((1/vp_rms**2) - p_ref**2)

        tps = H * (qa - qb)     # arrival time of primary
        tppps = tps * ( (qa + qb) / (qa - qb) )     # arrival time of first mulitple


        # migrate estimated times as we do with full velocity profile
        qa = np.sqrt((1/(vs)**2) - p_ref**2)
        qb = np.sqrt((1/(vp)**2) - p_ref**2)


        dt = 0.5 * (qa - qb)

        tbot = np.cumsum(dt, axis = 0)    # total travel time at bottom of interval
        ttop = tbot - dt        # total travel time at top of interval
        tmid = np.mean([tbot, ttop], axis = 0) # estimated time at middle of interval

        # cycle through columns
        z_interp = interp1d(tmid, z)

        first_mult = z_interp(tppps)


        fig.plot(x = x_len, y = -first_mult, pen = '3p,black,--')





    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap=True, position='jLB+w{}/0.5c+o0c/-1c+h'.format(fig_x),
                     frame='xafg')

    fig.basemap(region = [0, joints_dist[-1], -depth, 0], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa20f10g20', 'ya10f10g20'])


    fig.shift_origin(yshift='{}c'.format(fig_y))
    fig.basemap(region=[0, joints_dist[-1], 0, 5], projection='X{}c/{}c'.format(
        fig_x, 1.5), frame=['WsNe', 'xa20f10+lProfile Distance (km)', 'ya4f1+e+lElev. (km)'])

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

    Numerals = 'ABCDEFG'
    fig.text(x=joints_dist, y=np.ones(len(joints_dist)) * 4.9, text=list(Numerals[0:len(
        joints_dist)]), font='14.p, Helvetica-Bold', fill='white', pen='0.3p,black', no_clip=True)



fig.show()
