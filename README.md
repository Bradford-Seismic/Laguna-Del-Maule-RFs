Bradford Receiver Functions (July 7, 2026)

Python scripts for Receiver Function analysis used in the article 'Tracking Transcrustal Magma Ascent Beneath Laguna Del Maule Chile'

The intended purpose of these codes is for readers interested in the article to utilize and replicate our analyses as well as develop some of the figure found in the articles main text and supplementary material. These python codes are written by Jim Bradford (current affiliation: University of Arizona), but are inspired by similar codes in MatLab by Ryan Porter (current affiliation: Northern Arizona University) and techniques by Kevin Eagar (currently at Shell).

The python scripts follow a sequential series from data donwload to receiver function calculation and migration. These codes require several libraries, but mainly utilize pandas, numpy, obspy, scipy, and pygmt.

Furthermore, receiver function calculation calls on the fortran codes by Charles Ammon which can be found here: http://eqseis.geosc.psu.edu/cammon/HTML/RftnDocs/thecodes01.html

I'm curently a PhD. student at the time of these scripts being uploaded. In my experience so far, I have found it frustrating to see an amazing figure in an article and have now idea how they were made. These scripts being open to view and use is my attempt to make the lives of researchers and workers easier who may be interested in performing similar research, and I encourage anyone to reach out to me with questions about designing and assessing receiver function datasets.

-Jim Bradford


These scripts are intended to be run in sequential order (__Bradord_RF_0#.py) to produce the receiver functions presented in this work

Step 1: Download_Data
Calculate estimated arrival times from a list of stations and global earthquakes, then download event-receiver waveforms in an mseed format

Step 2: Sort_and _Convert
Move the downloaded into a more usable and informative SAC format within specified data directories

Step 3: RecordSection_and_SNR
Produce event moveout record sections for each event and visually inspect their quality. Accepted events are recorded onto a Moveout_Summary.csv file. Those passing events undergo a SNR calculation for each waveform. 

Step 4: Calc_RFs
For each passing event-receiver data based on moveout quality and SNR, the receiver function is calculated using the Iterative Time Deconvolution algorithm by Ligorria and Ammon (1999). 

Step 5: Mapping_and_Plotting
Preliminary data are plotted in map view along with time domain receiver function cross sections. 
The user may return to this script multiple times with different data constraints

Step 6: QC_RFs.py
From the entire dataset or a subset, manually or automatically reject receiver functions from the dataset. This produces list of passing files, and organizes those passing data into a DATA_CCP folder where passing data can be more easily accessed.

Step 7: CCP
Ray Tracing and Common Conversion Point stacking based on algorithms  originally by Ryan Porter (Northern Arizona University) adapted into Python by Jim Bradford. A netcdf file output of migration receiver function amplitudes and statistics is saved into the MAPPING/nc_files/ folder. Users can select any data they want to be included in the CCP calculation.

Step 8: Multiples 
Using defined set of horizontal interfaces, calculate the estimated multiple based on the Ps-delay time arrival.

The following Figure files contain the exact steps used to produce this works main text and supplementary text receiver function figures. 
