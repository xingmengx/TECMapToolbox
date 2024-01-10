#!/usr/bin/env python3
'''
PURPOSE:
    download 15-min resolution JPLD TEC maps for a specified year
    from jpl sideshow website

AUTHOR:
    Xing Meng        

LICENSE:
    Copyright 2023 California Institute of Technology                                             
    Licensed under the Apache License, Version 2.0 (the "License");                               
    you may not use this code except in compliance with the License.                              
    You may obtain a copy of the License at                                                       
    http://www.apache.org/licenses/LICENSE-2.0

'''

from os import path, makedirs
import argparse
import urllib.request


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', dest='year', required=True, \
                        help='year of data to fetch, can be any year ' + \
                        'from 2004 to 2022')
    parser.add_argument('-o', '--outdir', dest='outdir', required=True, \
                        help='directory to store the downloaded data')
     
    args = parser.parse_args()
    year = args.year

    leap_years = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

    if year in leap_years:
        ndays = 366
    else:
        ndays = 365
        
    url_base = 'https://sideshow.jpl.nasa.gov/pub/iono_daily/gim_for_research/jpld/'
    url_full = url_base + year + '/'

    # the downloaded data files will go here
    outdir = args.outdir
    if not path.exists(outdir):
        makedirs(outdir)

    # download all files for the year
    # jpld for year 2003 has a missing day (doy 303) so the following
    # does not work for 2003.
    for iday in range(1,ndays+1):
        filename = 'jpld'+format(iday,'03d')+'0.'+year[2:]+'i.nc.gz'
        urllib.request.urlretrieve(url_full+filename, outdir+'/'+filename)
    
