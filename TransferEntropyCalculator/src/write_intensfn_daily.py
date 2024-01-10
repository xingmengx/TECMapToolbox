#!/usr/bin/env python3

'''
PURPOSE:
    Read 15-min TEC intensification data, compute the daily maxima and averages
    for number of intensifications and total REC, write the daily data to 
    intensfn_daily file for a user-defined year range

NOTE:
    Other intensification characteristics can be added.

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
import numpy as np
from datetime import datetime, timedelta

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--beginyear', dest='beginyear', required=True, \
                        help='begin year of data to fetch', type=int)
    parser.add_argument('--endyear', dest='endyear', required=True, \
                        help='end year of data to fetch', type=int)
    parser.add_argument('-i', '--indir', dest='indir', required=True, \
                        help='directory of input 15-min TEC intensification data')
    parser.add_argument('-o', '--outdir', dest='outdir', \
                        help='directory to store output intensification daily data', \
                        default='../sample_datain')
    
    args = parser.parse_args()
    beginyear = args.beginyear
    endyear = args.endyear
    years = list(range(beginyear,endyear+1))    
    indir = args.indir
    outdir = args.outdir
    if not path.exists(outdir):
        makedirs(outdir)


    # read intensification data
    t_prior = datetime(1999,1,1)
    t_seq = []  # t sequence, every 15 mins, no repeat
    num_intensfn = []
    rectotal = []
    sizetotal = []
    for year in years:
        print('reading data of ', year)
    
        t_seq_thisyear = []
        num_intensfn_thisyear = []
        rectotal_thisyear = []
        sizetotal_thisyear = []
        f = open(indir + 'intensifications_' + str(year) + '.dat', 'r')
        for line in f:
            if line[0] == '#':
                continue
            record = line.split()
            t = datetime.strptime(' '.join(record[0:6]),"%Y %m %d %H %M %S")
            num_intensfn_thisyear.append(int(record[6]))
            rec = float(record[12])
            size = float(record[13])
            
            if t == t_prior:
                num_intensfn_thisyear.pop(-2)
                rectotal_thisyear[-1] = rectotal_thisyear[-1] + rec
                sizetotal_thisyear[-1] = sizetotal_thisyear[-1] + size
            else:
                t_seq_thisyear.append(t)
                rectotal_thisyear.append(rec)
                sizetotal_thisyear.append(size)
            t_prior = t
        f.close()
    
        # append this year to all years of data
        t_seq.extend(t_seq_thisyear)
        num_intensfn.extend(num_intensfn_thisyear)
        rectotal.extend(rectotal_thisyear)
        sizetotal.extend(sizetotal_thisyear)

    assert(len(t_seq) == len(num_intensfn))
    assert(len(t_seq) == len(rectotal))
    print(len(t_seq))
    
    # get daily maxima and averages of num and rectotal
    time_daily = []
    num_avg = []
    rectotal_avg = []
    sizetotal_avg = []
    num_max = []
    rectotal_max = []
    sizetotal_max = []
    day_current = 1
    nums = []
    recs = []
    sizes = []
    for it, t in enumerate(t_seq):
        if t.day == day_current:
            nums.append(num_intensfn[it])
            recs.append(rectotal[it])
            sizes.append(sizetotal[it])
        else:
            # last time from the previous day
            t_previous = t_seq[it-1]
            # time_daily at 12UT of each day
            time_daily.append(datetime(t_previous.year, t_previous.month, t_previous.day) + \
                              timedelta(hours=12))
            num_avg.append(np.mean(nums))
            rectotal_avg.append(np.mean(recs))
            sizetotal_avg.append(np.mean(sizes))
            num_max.append(np.amax(nums))
            rectotal_max.append(np.amax(recs))
            sizetotal_max.append(np.amax(sizes))
            day_current = t.day
            nums = [num_intensfn[it]]
            recs = [rectotal[it]]
            sizes = [sizetotal[it]]
    # append the last day of data
    if len(nums) > 0:
        time_daily.append(datetime(t.year, t.month, t.day) + \
                          timedelta(hours=12))
        num_avg.append(np.mean(nums))
        rectotal_avg.append(np.mean(recs))
        sizetotal_avg.append(np.mean(sizes))
        num_max.append(np.amax(nums))
        rectotal_max.append(np.amax(recs))
        sizetotal_max.append(np.amax(sizes))
    assert(len(time_daily) == len(num_avg))
    print(len(time_daily))
    
    
    # write into intensification daily file
    outfile = outdir + '/intensifications_daily_' + str(beginyear) + \
        'to' + str(endyear) + '.dat'
    with open(outfile,'w') as intensfn_file:
        intensfn_file.write('# Daily TEC intensification parameters extracted from ' + \
                            'intensifications_yyy.dat\n')
        intensfn_file.write('# year month day ' + \
                            'number_max number_avg totalREC_max[GECU] ' + \
                            'totalREC_avg[GECU] totalsize_max totalsize_avg\n')
    
        for it, t_daily in enumerate(time_daily):
            intensfn_file.write(('{:4d}'*4+'{:9.3f}'+'{:12.6f}'*4+'\n') \
                                .format(t_daily.year, t_daily.month, t_daily.day, \
                                        num_max[it], num_avg[it], \
                                        rectotal_max[it], rectotal_avg[it], \
                                        sizetotal_max[it], sizetotal_avg[it]))        
    
