#!/usr/bin/env python3
'''
PURPOSE:
    download 5-min resolution OMNI data for a user-defined 
    year range from SPDF

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
    parser.add_argument('--beginyear', dest='beginyear', required=True, \
                        help='begin year of data to fetch', type=int)
    parser.add_argument('--endyear', dest='endyear', required=True, \
                        help='end year of data to fetch', type=int)
    parser.add_argument('-o', '--outdir', dest='outdir', required=True, \
                        help='directory to store the downloaded data')
     
    args = parser.parse_args()
    beginyear = args.beginyear
    endyear = args.endyear
    
    # the downloaded data files will go here
    outdir = args.outdir
    if not path.exists(outdir):
        makedirs(outdir)

    url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/'
    ### to download modified (level-3) data (less coverage):
    #url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/'
    ###
    
    # download files 
    for year in range(beginyear,endyear+1):
        filename = 'omni_5min' + format(year,'4d') +'.asc'
        print('... downloading ', filename)
        urllib.request.urlretrieve(url+filename, outdir+filename)
    
