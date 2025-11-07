#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:51:43 2025

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


# %% Go CCP

plt.close('all')


# Plotting Params


model_name = 'CCP_LDM_G-7.5_GDW_VELEST-ZR-mod.nc'


use_line = 1
plot_EQS = 1
plot_RF = 0
plot_params = 0
plot_points = 0


# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 20
std_hits_limit = 2
std_limit = 0.5


if use_line:
    joints = np.loadtxt(
        '../DATA/MAPPING/{}_Line-WE2.txt'.format(region), delimiter=',')
    make_joints = False

else:
    make_joints = True


metric = 'Amplitude'
sca = [-0.5, 0.5]  # value bounds for model slice


# fig_x = 8   # length of figure in cm
fig_y = 11
aspect = 1        # dimensions of x-axis km to y-axis km


fix_lim = 0
x_lim = [0, 100]
y_lim = [-60, 0]


model = xr.open_dataset('../DATA/MAPPING/nc_models/' + model_name)
z_min = model.depth.values.min()


if plot_EQS:
    eq_range = 15#model.attrs['Bin Min Smoothing']
    eq_xyz = pd.read_csv('../DATA/MAPPING/catalog_zr+tango.csv')
    o_time = ['{}:{}:{}T{}:{}:{}'.format(eq_xyz.year[i], eq_xyz.month[i], eq_xyz.day[i],
                                         eq_xyz.hour[i], eq_xyz.minute[i], eq_xyz.second[i]) for i in range(len(eq_xyz))]
    eq_xyz['O_Time'] = o_time


if plot_RF:
    trace_interval = 10
    rf_scale = 10

if plot_points:
    points_files = ['LdM_D1-1_WE.txt',
                    'LdM_D1-1_WE_mult1.txt',
                    'LdM_D1-1_WE_mult2.txt',
                    'LdM_D1-2_WE.txt',
                    'LdM_D1-2_WE_mult1.txt',
                    'LdM_D1-2_WE_mult2.txt',
                    'LdM_D1-3_WE.txt',
                    'LdM_D1-3_WE_mult1.txt',
                    'LdM_D1-3_WE_mult2.txt',]

# Functions


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


def shift_line_ns(coords: np.ndarray, km: float) -> np.ndarray:
    """
    Shift a polyline north or south by a set number of kilometers.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with columns [lon, lat].
    km : float
        Shift distance in kilometers. Positive = north, Negative = south.

    Returns
    -------
    np.ndarray
        Shifted coordinates (N, 2) in [lon, lat].
    """
    # Make sure coords is Nx2
    coords = np.asarray(coords)
    assert coords.shape[1] == 2, "Input must be Nx2 [lon, lat]"

    # Center point for choosing an appropriate local projection
    lon0, lat0 = np.mean(coords[:, 0]), np.mean(coords[:, 1])

    # Define a local azimuthal equidistant projection centered on the line
    crs_geodetic = CRS.from_epsg(4326)  # WGS84
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84")

    fwd = Transformer.from_crs(crs_geodetic, crs_local, always_xy=True)
    inv = Transformer.from_crs(crs_local, crs_geodetic, always_xy=True)

    # Project to local meters
    x, y = fwd.transform(coords[:, 0], coords[:, 1])

    # Shift in north/south = add to y
    y_shifted = y + km * 1000.0

    # Transform back
    lon_shift, lat_shift = inv.transform(x, y_shifted)
    return np.column_stack([lon_shift, lat_shift])


def plot_topo(fig, joints, fig_x, fig_y, Numerals='ABCDEFG'):

    joints_dist, line, line_dist = Return_Line_Elements(joints)

    fig.shift_origin(yshift='+h')
    with pygmt.config(FONT='16p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p/28p"):
        if fix_lim:
            fig.basemap(region=[x_lim[0], x_lim[1], 0, 5], projection='X{}c/{}c'.format(
                fig_x, fig_y / 10), frame=['WsNe', 'xa20f10+lProfile Distance (km)', 'ya4f1+e+lElev. (km)'])
        else:
            fig.basemap(region=[0, joints_dist[-1], 0, 5], projection='X{}c/{}c'.format(
                fig_x, fig_y / 10), frame=['WsNe', 'xa20f10+lProfile Distance (km)', 'ya4f1+e+lElev. (km)'])

        line_n = shift_line_ns(line, 18)   # shift 10 km north
        line_s = shift_line_ns(line, -18)  # shift 10 km south

        topo_x = np.append(line_dist[0], line_dist)
        topo_x = np.append(topo_x, line_dist[-1])

        # north topography

        x = xr.DataArray(line_n[:, 0], dims='Distance_Along_Trend')
        y = xr.DataArray(line_n[:, 1], dims='Distance_Along_Trend')

        dem = grid.interp(lat=y, lon=x, method='linear')
        dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

        topo_n = np.append(0, dem.values)
        topo_n = np.append(topo_n, 0)

        fig.plot(x=topo_x, y=topo_n/1000, fill='darkgray', pen='0.5p,black')

        # on-line topography

        x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
        y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')

        dem = grid.interp(lat=y, lon=x, method='linear')
        dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

        topo = np.append(0, dem.values)
        topo = np.append(topo, 0)

        fig.plot(x=topo_x, y=topo/1000, fill='lightgray', pen='0.5p,black')

        # south topography

        x = xr.DataArray(line_s[:, 0], dims='Distance_Along_Trend')
        y = xr.DataArray(line_s[:, 1], dims='Distance_Along_Trend')

        dem = grid.interp(lat=y, lon=x, method='linear')
        dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

        topo_s = np.append(0, dem.values)
        topo_s = np.append(topo_s, 0)

        fig.plot(x=topo_x, y=topo_s/1000, fill='lightgray',
                 pen='0.5p,black', transparency=50)
        fig.plot(x=topo_x, y=topo_s/1000,  pen='0.5p,black')





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
            joints_dist)]), font='14.p, Helvetica-Bold', fill='white', pen='0.3p,black', no_clip=True)

    return fig


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
shade = pygmt.grdgradient(grid=grid, azimuth="0/90", normalize="e1")


Lon, Lat = np.meshgrid(grid.lon.values, grid.lat.values)

fig, ax = plt.subplots(figsize=(12, 12))
ax.pcolormesh(Lon, Lat,  grid, cmap='terrain', vmin=-
              grid.values.max(), vmax=grid.values.max())


ax.scatter(stat_table.stlo, stat_table.stla,
           marker='v', color='r', edgecolor='black')

ax.set_xlim([west, east])
ax.set_ylim([south, north])
ax.set_aspect('equal')


if make_joints:
    joints = plt.ginput(n=99, show_clicks=True)
    joints = np.array(joints)


joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')

ax.plot(x.data, y.data, '--', color='k',  linewidth=3)
plt.pause(0.01)
fig.canvas.draw()

### Interpolate data grids ############################

model_slice = model.interp(longitude=x, latitude=y,
                           method='linear', kwargs={"fill_value": None})
model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

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


X, Z = np.meshgrid(
    model_slice['Distance_Along_Trend'].values, model_slice.depth.values)


#### Configure plot sizing ############################

# if fix_lim:

#     xdist = x_lim[1] - x_lim[0]
#     ydist = y_lim[1] - y_lim[0]

#     ratio = xdist / fig_x     # ratio of km to cm
#     fig_y = ydist / (ratio * aspect)


# else:
#     ratio = joints_dist[-1] / fig_x     # ratio of km to cm
#     z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
#     fig_y = z_lim / (ratio * aspect)


z_lim = np.abs(model_slice.depth.values.min())  # depth of CCP grid
ratio =  z_lim / fig_y     # ratio of km to cm
fig_x = joints_dist[-1] / (ratio * aspect)




############### Plotting objects ############################


fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
    if fix_lim:

        grid_region = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]
        prj = 'X{}/{}'.format(fig_x, fig_y)

        fig.basemap(region=grid_region, projection=prj, frame=[
                    'Wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])
    else:

        grid_region = [0, joints_dist[-1],
                       model_slice.depth.values.min(), 0]
        prj = 'X{}/{}'.format(fig_x, fig_y)
        fig.basemap(region=grid_region, projection=prj, frame=[
                    'Wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True)
    fig.grdimage(grid=model_slice[metric], cmap=True)

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap=True, position='jLB+w{}/0.5c+o0c/-1c+h'.format(fig_x),
                     frame='xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, metric))

    # Plot hypocenters
    if plot_EQS:
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

    # Plot RF trace lines
    if plot_RF:
        for dist in line_dist:
            if dist % 10 < 0.1:
                RF = model_slice.interp(
                    Distance_Along_Trend=dist, method='nearest')
                fig.plot(x=RF[metric].values*rf_scale + dist,
                         y=RF.depth.values, pen='0.5p,black')

    # plot picked points
    if plot_points:
        for f in points_files:
            xz = np.loadtxt('../DATA/MAPPING/{}'.format(f),
                            delimiter=',', skiprows=2)
            fig.plot(x=xz[:, 0], y=xz[:, 1], pen='1p,black,--')

    # Plot CCP parameters text box
    if plot_params:
        params = 'node space: {}, bin radius: {}, bin std-rad: {}'.format(model.attrs['Node Spacing'],
                                                                          model.attrs['Bin Min Smoothing'],
                                                                          model.attrs['Bin Gauss'])
        fig.text(
            text=params,
            position="BL",  # Top Left
            justify="BL",  # Top Left
            offset=".1c/.1c",
        )

    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=[
                'Wsne', 'xaf100g20+lProfile Distance (km)', 'ya20f10g20+lDepth (km)'])


# configure legend
if plot_EQS:
    spec = io.StringIO(
        """
N 3
S 0.20c i 0.4c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.4c cyan 0.6p,black 0.65c TANGO Node
S 0.20c i 0.4c lightred 0.6p,black 0.65c ZR Network


S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano
S 0.2c c 0.2c green 0.5p,black 0.65c Hypocenter (\\261 {} km radius)

G 0.07c
G 0.07c

""".format(eq_range)
    )
else:
    spec = io.StringIO(
        """
N 3
S 0.20c i 0.4c lightred 0.6p,black 0.65c ZR Network
S 0.20c i 0.4c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.4c cyan 0.6p,black 0.65c TANGO Node

S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano

G 0.07c
G 0.07c

"""
    )


# plot legend
with pygmt.config(FONT_ANNOT_PRIMARY="14p"):
    fig.legend(spec=spec,  position='jBL+w{}c+o0c/-4.25c'.format(fig_x*2))


# Plot topography
fig = plot_topo(fig, joints, fig_x, fig_y)

fig.show()

sys.exit('Plot Finished')



# %% Plot Velocity Model

joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:, 0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:, 1], dims='Distance_Along_Trend')
z = xr.DataArray(-model.depth.values, dims='depth')

vs = model.attrs['Velocity Model']
vpvs = model.attrs['VpVs Model']

vs = xr.open_dataset('../DATA/MAPPING/nc_models/' + vs)
vpvs = xr.open_dataset('../DATA/MAPPING/nc_models/' + vpvs)

vs_slice = vs.interp(longitude=x, latitude=y, depth=z,
                     method='linear', kwargs={"fill_value": None})
vpvs_slice = vpvs.interp(longitude=x, latitude=y,
                         method='linear', kwargs={"fill_value": None})

vs_slice = vs_slice.assign_coords({'Distance_Along_Trend': line_dist})
vs_slice = vs_slice.assign_coords({'depth': -vs_slice.depth.values})

vpvs_slice = vpvs_slice.assign_coords({'Distance_Along_Trend': line_dist})
vpvs_slice = vpvs_slice.assign_coords({'depth': -vpvs_slice.depth.values})

minv = 2.0  # lowest allowable shear wave velocity in model, any point lower will be raised to this
max_vpvs = 1.82  # maximum allowable Vp/Vs in model
min_vpvs = 1.72  # lowest allowable Vp/Vs in model


vs_slice['vs'].values = np.clip(
    vs_slice['vs'].values, minv, np.nanmax(vs.vs.values))
vpvs_slice['VpVs'].values = np.clip(
    vpvs_slice['VpVs'].values, min_vpvs, max_vpvs)


fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
    if fix_lim:
        fig.basemap(region=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], projection='X{}/{}'.format(
            fig_x, fig_y), frame=['Wsne', 'xaf100+lProfile Distance (km)', 'ya20f10+lDepth (km)'])
    else:
        fig.basemap(region=[0, np.ceil(line_dist.max()), model_slice.depth.values.min(
        ), 0], projection='X{}/{}'.format(fig_x, fig_y), frame=['Wsne', 'xaf100+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='jet', series=[minv, 6.0, 0.1], reverse=True)
    fig.grdimage(grid=vs_slice.vs, cmap=True)
    fig.grdcontour(grid=vs_slice.vs, levels=[
                   2.9, 3.2, 3.5, 3.8, 4.1], annotation=[2.9, 3.2, 3.5, 3.8, 4.1])

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap=True, position='jLB+w{}/0.5c+o0c/-1c+h'.format(fig_x/2),
                     frame='xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, 'vs (km/s)'))

    spec = io.StringIO(
        """
N 3
S 0.20c i 0.6c gold 0.6p,black 0.65c ZR Network
S 0.20c i 0.6c magenta 0.2p,black 0.65c TANGO Broadband
S 0.20c i 0.6c cyan 0.4p,black 0.65c TANGO Node

S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano

G 0.07c
G 0.07c

"""
    )


# plot legend
with pygmt.config(FONT_ANNOT_PRIMARY="16p"):
    fig.legend(spec=spec,  position='jBL+w{}c+o0c/-5c'.format(fig_x))


fig = plot_topo(fig, joints, fig_x, fig_y)

fig.show()


sys.exit('here')


fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
    if fix_lim:
        fig.basemap(region=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], projection='X{}/{}'.format(
            fig_x, fig_y), frame=['WSne', 'xaf100+lProfile Distance (km)', 'ya20f10+lDepth (km)'])
    else:
        fig.basemap(region=[0, np.ceil(line_dist.max()), model_slice.depth.values.min(
        ), 0], projection='X{}/{}'.format(fig_x, fig_y), frame=['WSne', 'xaf100+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(
        cmap='jet', series=[min_vpvs, max_vpvs, 0.0005], reverse=True)
    fig.grdimage(grid=vpvs_slice.VpVs, cmap=True)
    fig.grdcontour(grid=vpvs_slice.VpVs, levels=[1.79], annotation=[1.79])

    with pygmt.config(FONT_ANNOT_PRIMARY="24p", FONT_LABEL="28p"):
        fig.colorbar(
            cmap=True, position='JMR+w{}/0.5c+o0.5c/0c'.format(fig_y), frame='xaf+lVp/Vs')

spec = io.StringIO(
    """
N 3
S 0.20c i 0.6c gold 0.6p,black 0.65c ZR Network
S 0.20c i 0.6c magenta 0.2p,black 0.65c TANGO Broadband
S 0.20c i 0.6c cyan 0.4p,black 0.65c TANGO Node

S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano

G 0.07c
G 0.07c

"""
)


# plot legend
with pygmt.config(FONT_ANNOT_PRIMARY="16p"):
    fig.legend(spec=spec,  position='jBL+w{}c+o0c/-5c'.format(fig_x))


fig = plot_topo(fig)

fig.show()


# %% Pick Points

save_picks = 0
pick_name = "LdM_D1-1_SN"

print("Picking points in file: {}".format(model_name))


plt.close('all')


def parse_gmt_cpt(cpt_text):
    lines = [line.strip() for line in cpt_text.strip().splitlines()
             if line.strip() and not line.startswith('#')]
    stops = []
    for line in lines:
        parts = line.split()
        if len(parts) == 8:
            x0, r0, g0, b0, x1, r1, g1, b1 = parts
            stops.append((float(x0), (int(r0)/255, int(g0)/255, int(b0)/255)))
    # Add last color stop (end of last line)
    last_line = lines[-1].split()
    stops.append((float(last_line[4]), (int(
        last_line[5])/255, int(last_line[6])/255, int(last_line[7])/255)))

    # Remove duplicates and sort
    stops = sorted(list(dict(stops).items()))

    values, colors = zip(*stops)

    # Normalize values to [0,1]
    vmin, vmax = values[0], values[-1]
    values_norm = [(v - vmin) / (vmax - vmin) for v in values]

    # Create color tuples for colormap
    color_list = list(zip(values_norm, colors))

    cmap = LinearSegmentedColormap.from_list(
        "custom_gmt_cpt", color_list, N=256)
    return cmap, vmin, vmax


with open('../DATA/MAPPING/no_green.cpt', 'r') as cpt_file:
    cpt_text = cpt_file.read()
    cmap, vmin, vmax = parse_gmt_cpt(cpt_text)


fig, ax = plt.subplots(figsize=(fig_x, fig_y))


ax.pcolormesh(model_slice['Distance_Along_Trend'].values, model_slice['depth'].values,
              model_slice[metric].values, cmap=cmap, vmin=sca[0], vmax=sca[1])


ax.set_xlim(x_lim)
ax.set_ylim(y_lim)


# sys.exit('here')

picks = np.array(plt.ginput(99))

if save_picks:
    header = "PickModel: {} Jointlon: {} Jointlat: {}\nlon,lat".format(model_name, ",".join(joints[:, 0].astype(str)),
                                                                       ",".join(joints[:, 1].astype(str)))

    np.savetxt('../DATA/MAPPING/{}.txt'.format(pick_name),
               picks, delimiter=',', header=header)


# %% 1D vel model

dzi = 0.5   # depth increment in km
z_max = 60  # max depth in km


v_model_1D = 'velmodel_zr-mod.csv'

# Plot v_model

v_data = pd.read_csv('../DATA/MAPPING/nc_models/' +
                     v_model_1D,   skiprows=1, header=None)

vsu = v_data[2].to_numpy()
vpu = v_data[1].to_numpy()
vd = v_data[0].to_numpy()

z = np.arange(np.round((np.min(vd)) * 1/dzi) * dzi, z_max + dzi, dzi)


# Interpolate vs and k at mid-points between layers in z using nearest values
vs_interp_func = interp1d(vd, vsu, kind='nearest',
                          bounds_error=False, fill_value='extrapolate')
vp_interp_func = interp1d(vd, vpu, kind='nearest',
                          bounds_error=False, fill_value='extrapolate')

vs = vs_interp_func(z + 0.5*dzi)
vp = vp_interp_func(z + 0.5*dzi)


fig = pygmt.Figure()    # initialize the map

with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
    fig.basemap(region=[vs.min() - 1, vp.max() + 1, z.min(), z.max()],
                projection='X3c/-{}c'.format(fig_y), frame=['wSnE', 'xa2f1+lvel.', 'ya10f5g20+lz (km)'])

    fig.plot(x=vp, y=z, pen='1p,blue')
    fig.plot(x=vs,  y=z, pen='1p,red,--')

spec = io.StringIO(
    """
N 2
S 0.20c - 0.6c - 1p,blue 0.7c vp
S 0.20c - 0.6c - 1p,red 0.7c vs

    """
)


with pygmt.config(FONT_ANNOT_PRIMARY="16p"):
    fig.legend(spec=spec,  position='jTL+w3.5c+o0c/-1c')


fig.show()


# %% Depth Slice


z_slice = model.interp(depth=-3, method='linear')


z_slice[metric].values = np.clip(z_slice[metric].values, sca[0], sca[1])

gmt_region = [-70.66, -70.36, -36.2, -35.8]


fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p", FORMAT_GEO_MAP="ddd.x"):

    prj = 'M{}c'.format(fig_x)            # set the map projection

    fig.basemap(region=gmt_region, projection=prj, frame=[
                'xafg1+lLongitude', 'ya0.25g1+lLatitude', 'wSnE'])

    grid_color = pygmt.makecpt(
        cmap='../DATA/MAPPING/no_green.cpt',   series=[sca[0], sca[1]], continuous=True)
    fig.grdimage(grid=z_slice[metric], cmap=True)


Lagunas = gpd.read_file('../DATA/MAPPING/Lagunas.shp')
fig.plot(data=Lagunas, pen='1p,black', fill='skyblue')

fig.show()


