# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:50:17 2025

@author: 7418888
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
import time




warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


active_dir = ''



os.chdir(active_dir)


model = TauPyModel(model="iasp91")



stations = pd.read_csv('../DATA/SPREADSHEETS/LDM_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'LDM'



#%% Check itd files


pass_percentage = 60        # Correlation %
check_G = [5.0]             # assess specific Gaussian filter
check_net = []              # assess specific Network





run_auto_QC = False          # run Auto-QC only
run_manual_QC = True       # run Manual-QC (also runs an auto QC)
"""

                Auto-QC: scan for negative first pulses and nan traces
                Manual-QC manually identify traces that passed Auto_QC

"""

check_specific_area = False  # make map of all station locations and choose stations
                            # within region for QC, make updates for zone in area

reuse_results = True  # if True, do not check ITD results where 'Accept' entries list that they have already passed





"""
            Filter the master itd file to comform to listed parameters
"""



# Data Frame 'itd' is the master file that contains all QC information
# filter down 'itd' to contain only files within the region and pass the 'pass_percentage'

itd_check = itd[itd.Region == region].copy()
region_len = len(itd_check)
print('{} Calculated .itr files in {}'.format(region_len, region))

itd_check = itd_check[itd_check.Corr > pass_percentage].copy().reset_index(drop = True)
corr_len = len(itd_check)
print('{} / {} passing files based on Correlation - {:.2f}%'.format(corr_len, region_len, (corr_len / region_len) *100))



# further filter down to only G values within G
itd_G = pd.DataFrame(columns = itd.columns)
for G in check_G:
    itd_G = pd.concat((itd_G, itd_check[itd_check.G == G].copy()), ignore_index=True)

itd_check = itd_G.copy()
G_len = len(itd_check)
print('{} passing files containing G values'.format(G_len))





# further filter down to only specified networks within 'check_net'
if len(check_net) != 0:
    itd_net = pd.DataFrame(columns = itd.columns)
    for net in check_net:
        itd_net = pd.concat((itd_net, itd_check[itd_check.Net == net].copy()), ignore_index=True)

    itd_check = itd_net.copy()
    print('{} passing files containing specified networks'.format(len(itd_check)))






if check_specific_area:


    print('plotting interactive map, switch to interactive graphics')
    # dem grid for plot
    edge_round = 1
    edge_gap = 0.1
    north = np.ceil(stations.Latitude.max()*edge_round)/edge_round + edge_gap
    south = np.floor(stations.Latitude.min()*edge_round)/edge_round - edge_gap

    east = np.ceil(stations.Longitude.max() *edge_round)/ edge_round + edge_gap
    west = np.floor(stations.Longitude.min() *edge_round)/ edge_round - edge_gap


    gmt_region = '{}/{}/{}/{}'.format(west, east, south, north)
    grid = pygmt.datasets.load_earth_relief(resolution="01m", region=gmt_region)
    Lon, Lat = np.meshgrid(grid.lon.values, grid.lat.values)




    # Plot fig with all stations

    plt.close('all')

    fig, ax = plt.subplots(figsize = (12, 12))

    ax.pcolormesh(Lon, Lat,  grid, cmap = 'terrain', vmin = -grid.values.max(), vmax = grid.values.max())

    ax.scatter(stations.Longitude, stations.Latitude, marker = 'v', color = 'r', edgecolor = 'black')
    ax.set_aspect('equal')



    # choose nw and se corners of box
    nw_se_corner = plt.ginput(n = 2, show_clicks = True)

    north = np.ceil(nw_se_corner[0][1]*1e2)/1e2
    south = np.ceil(nw_se_corner[1][1]*1e2)/1e2

    west = np.ceil(nw_se_corner[0][0]*1e2)/1e2
    east = np.ceil(nw_se_corner[1][0]*1e2)/1e2


    plt.close('all')



    # append stations within selected area
    check_list = pd.DataFrame(columns = itd.columns)

    print('Collecting Stations within Area')
    for i in range(len(stations)):
        stla = stations.Latitude[i]
        stlo = stations.Longitude[i]
        net = stations.Network[i]
        stat = stations.Station[i]
        if (stla > north) or (stla < south) or (stlo < west) or (stlo > east):
            continue
        else:
            check_list = pd.concat((check_list, itd_check[itd_check.Net == net][itd_check.Stat == stat].copy()), ignore_index=True)

    itd_check = check_list.copy()
    print('{} files within specified region'.format(len(itd_check)))






if reuse_results:
    itd_check = itd_check[(itd_check.Accept != 'Man QC Pass') & (itd_check.Accept != 'Man QC Fail')].reset_index(drop = True)
    print('{} files to examine not including prior results'.format(len(itd_check)))






#### Read the check_list
st = Stream()
print('\n')
for i in range(len(itd_check)):
    # pull file data for each item itd_check
    code = itd_check.Code[i].replace('-', '_')
    file = itd_check.File[i]

    # read the RF
    st += read('../DATA/{}/Data_By_Station/{}/{}'.format(region, code, file))

    # append file name into stats for later tracking
    st[-1].stats.fname = file
    print("\r", end="")
    print("Reading Passing Files: {:.1%} ".format(i/(len(itd_check))), end="")








def input_status():
    while True:
        status = input('is good? [y/n]')
        if not((status == 'y') or (status == 'n') or (status == 'quit')):
            print('Enter y/n')
        else:
            break

    return status



def Auto_QC_Stream(st):
    st_copy = st.copy()
    # Quality control the stream
    # first check for nan trace, then check negative first

    num_pass = 0
    num_fail = 0
    num_null = 0


    print('\n')
    t=0
    for tr in st_copy:
        t+=1
        print("\r", end="")
        print("RF QC: {:.1%} ".format((t/len(st_copy))), end="")

        if any(np.isnan(tr.data)):
            tr.stats.QC_result = 'Null Data'
            num_null += 1

        else:
            extrema_limit = tr.data.max() * 0.10
            # cycle through the data array to find all major extrema
            extrema=[]
            extrema_i = []

            # Note, maintain k as iterable variable
            k = 1
            for dat in tr.data[1:len(tr.data)-2]:
                left = tr.data[k - 1]
                right = tr.data[k + 1]
                if (dat > left) and (dat > right):
                        if dat > extrema_limit:
                            extrema.append(dat)
                            extrema_i.append(k)
                elif (dat < left) and (dat < right):
                    if dat < -extrema_limit:
                        extrema.append(dat)
                        extrema_i.append(k)
                k += 1

            p_amp = extrema[0]
            tr.data = tr.data / p_amp


            # now that we have the major extrema, if the first spike is negative
            # we want to reject it from the stream
            if  (len(extrema) == 0) or (extrema[0] < 0):
                tr.stats.QC_result = 'Auto QC Fail'
                num_fail += 1

            else:
                tr.stats.QC_result = 'Auto QC Pass'
                num_pass += 1

    print('{} Pass, {} Fail, {} Null'.format(num_pass, num_fail, num_null))
    return st_copy








if run_auto_QC:
    st_Auto = Auto_QC_Stream(st)
    k = 0
    print('\n')

    for tr in st_Auto:
        k+=1
        print("\r", end="")
        print("Writing QC: {:.1%} ".format(k/len(st_Auto)), end="")

        file = tr.stats.fname
        result = tr.stats.QC_result

        # locate file within itd table
        row_index = itd.loc[itd['File'] == file][itd.Region == region].index[0]
        itd.loc[row_index, 'Accept'] = result






### continue here
print('Switch plot output for Man QC')
sys.exit('here')





#%% Sub cell for Man QC

if run_manual_QC:
    st_Auto = Auto_QC_Stream(st)



    k = 0
    for tr in st_Auto:
        k += 1
        print(str(k/len(st_Auto)*100) + '%')

        if tr.stats.QC_result == 'Auto QC Pass':

            file = tr.stats.fname

            tr.plot()
            status = input_status()


            if status == 'y':
                tr.stats.QC_result = 'Man QC Pass'
            elif status == 'n':
                tr.stats.QC_result = 'Man QC Fail'

            elif status == 'quit':
                break

            file = tr.stats.fname
            row_index = itd.loc[itd['File'] == file][itd.Region == region].index[0]
            itd.loc[row_index, 'Accept'] = tr.stats.QC_result



        elif tr.stats.QC_result == 'Auto QC Fail':
            file = tr.stats.fname
            row_index = itd.loc[itd['File'] == file][itd.Region == region].index[0]
            itd.loc[row_index, 'Accept'] = tr.stats.QC_result




    ## Update the itd results file, hopefully you have a backup saved
    itd.to_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', index = False)







#%% Assemble set of .itr files to be used in CCP Migration


Gs = [2.5, 5.0, 7.5]
use_phases = ['P', 'p']
use_radial_convention = 'N'




# Read the data from the iterdecon summary file and gather data meeting the gaussian
# and phase requirements.

# any station that has been judged to have passed Auto QC or Manual QC will be incorporated

itd_pass = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv',keep_default_na=False)
itd_pass = itd_pass[itd_pass.Region == region][(itd_pass.Accept == 'Man QC Pass')]#[itd_pass.Stack_Radius == 10]
itd_pass = itd_pass.reset_index(drop = True)


# further filter down to only G values within G
itd_G = pd.DataFrame(columns = itd_pass.columns)
for G in Gs:
    itd_G = pd.concat((itd_G, itd_pass[itd_pass.G == G].copy()), ignore_index=True)

itd_pass = itd_G.copy()


# further filter down to only Phases within use_phases
itd_P = pd.DataFrame(columns = itd_pass.columns)
for phase in use_phases:
    itd_P = pd.concat((itd_P, itd_pass[itd_pass.Phase == phase].copy()), ignore_index=True)

itd_pass = itd_P.copy()







# Make a new directory, DATA_CPP, that contains all .itr files to be used in CCP stack migrations
if not(os.path.exists('../DATA/{}/DATA_CCP/'.format(region))):
    os.mkdir(os.path.join('../DATA/{}/DATA_CCP/'.format(region)))



# empty the current DATA_CCP folder
if len(glob('../DATA/{}/DATA_CCP/*'.format(region))) > 0:
    for f in glob('../DATA/{}/DATA_CCP/*'.format(region)):
        os.remove(f)



# Assess the desired content for DATA_CCP

codes = np.array([])
itd_use = pd.DataFrame(columns = itd_pass.columns)
for i, f in enumerate(itd_pass.File):
    print("\r", end="")
    print("Filtering for desired RF Files: {:.1%} ".format(i/(len(itd_pass))), end="")



    net, stat, event, G = itd_pass.Net[i], itd_pass.Stat[i], itd_pass.Event[i], itd_pass.G[i]
    code = '{}-{}-{}-{}'.format(net, stat, event, G)
    if code in codes:
        continue

    else:
        codes = np.append(codes, code)

    records = itd_pass[itd_pass.Net == net][itd_pass.Stat == stat][itd_pass.Event == event][itd_pass.G == G]

    # Pulling more than one record implies that there is a .1. and .N. rotation options for this
    # .itr file, we want to select the one that follows our radial component convention
    # Otherwise, if there is only one available, we will default to using the .N. convention file
    if len(records) > 1:
        for file in records.File:
            comp = file.split('.')[2]
            if comp == use_radial_convention:
                record_to_append = records[records.File == file]

                itd_use = pd.concat([itd_use, record_to_append], ignore_index=True)

    else:
        for file in records.File:
            comp = file.split('.')[2]
            if comp == 'N':
                record_to_append = records[records.File == file]

                itd_use = pd.concat([itd_use, record_to_append], ignore_index=True)




# Pull grid data for elevation

stat_table = pd.read_csv('../DATA/SPREADSHEETS/LDM_Stations.csv'.format(region))

edge_gap = 0.2

north = np.ceil(stat_table.Latitude.values.max()*1e2)/1e2+edge_gap
south = np.ceil(stat_table.Latitude.values.min()*1e2)/1e2-edge_gap

east = np.ceil(stat_table.Longitude.values.max()*1e2)/1e2+edge_gap
west = np.round(np.ceil(stat_table.Longitude.values.min()*1e2)/1e2-edge_gap, 2)


stlas = stat_table.Latitude.values
stlos = stat_table.Longitude.values


gmt_region = '{}/{}/{}/{}'.format(west, east, south, north)
# gmt_region = '{}/{}/{}/{}'.format(-68, -61.2, south, north)
# gmt_region = '{}/{}/{}/{}'.format(west, east, -38, -32.5)
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)




# write each file into DATA_CCP directory, note that copied files are not p-wave normalized
print('\n\n')
for i, f in enumerate(itd_use.File):
    print("\r", end="")
    print("Moving Passed Files to DATA_CPP: {:.1%} ".format(i/(len(itd_use))), end="")



    # read the iterdecon file
    code = itd_use.Code[i]
    st = read(os.path.join("../DATA/{}/Data_By_Station/{}/{}".format(region, code, f)))
    stla, stlo = st[0].stats.sac.stla, st[0].stats.sac.stlo

    # assign elevation balue based on station location within dem
    stel = grid.sel(lat = stla, lon = stlo, method = 'nearest').values.item()

    st[0].stats.sac.stel = stel


    tr = st[0]
    tr.write('../DATA/{}/DATA_CCP/{}'.format(region, f), format = 'SAC')








