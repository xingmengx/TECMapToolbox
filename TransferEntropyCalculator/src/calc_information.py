#!/usr/bin/env python3
'''
PURPOSE:
    Core script of the transfer entropy calculator:
    calculate mutual information (MI), transfer entropy (TE) for
    daily F10.7/15-min solar wind and TEC intensification data;
    calculate TE from shuffled daily F10.7/15-min solar wind to TEC 
    intensification data;
    calculate entropy rate (ER) for daily/15-min TEC intensification data.

NOTE: 
    The calculation of TE and ER uses omega=tau. Formulation of 
    omega=1 has been given as comments. A varying omega should be
    considered in the future updates.
    
AUTHOR:
    Xing Meng

LICENSE:
    Copyright 2023 California Institute of Technology                                             
    Licensed under the Apache License, Version 2.0 (the "License");                               
    you may not use this code except in compliance with the License.                              
    You may obtain a copy of the License at                                                       
    http://www.apache.org/licenses/LICENSE-2.0

'''

import argparse
from os import path, makedirs
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------------------------
# Defined functions
# --------------------------------------------------------------------------------

def calc_mutual_info(data1, threshold, threshold_dir, data2, dt):
    """
    Calculated time-shifted mutual information between data1 and data2
    Input: data1(1D array), threshold(to neglect some outliners in data1),
           threshold_dir(direction of threshold, 1 -- one-sided, 2 -- two-sided),
           data2(1D array), dt(time delay for data2)
    Note:  dt = the number of indices to shift upon data2. Example: for 
           15-min TEC data, the delay time = dt*15min.
    Output: mutual information between data1 and time-delayed data2

    """
    assert(data1.shape == data2.shape)    

    # optimal binning for a normal distribution (Sturges 1926)
    nbins = int(np.log2(len(data1))) + 1
    #print('... calc_mutual_info: nbins = ', nbins)

    # skip some outliner data in data1, apply time shift to data2
    if threshold_dir == 1:
        if dt == 0:
            data3 = data1[data1<threshold]
        else:
            data3 = data1[data1<threshold][:-dt] # keep data1 same length as shifted data2
        data4 = data2[data1<threshold][dt:]
    elif threshold_dir == 2:
        if dt == 0:
            data3 = data1[abs(data1)<threshold]
        else:
            data3 = data1[abs(data1)<threshold][:-dt] # keep data1 same length as shifted data2
        data4 = data2[abs(data1)<threshold][dt:]
    #print(dt, data1.shape, data2.shape, data3.shape, data4.shape)
    
    # calculate probablities
    hist, bin_edges = np.histogram(data3, bins=nbins)
    P_data1 = hist/hist.sum()
    hist, bin_edges = np.histogram(data4, bins=nbins)
    P_data2 = hist/hist.sum()
    hist2d, bin_edges1, bin_edges2 = np.histogram2d(data3, data4, bins=nbins)
    P_joint = hist2d/hist2d.sum() #joint probability of data1 and time-delayed data2

    # check to make sure the sum of the probablities is almost 1
    assert(abs(P_data1.sum() - 1.) < 1.e-8)
    assert(abs(P_data2.sum() - 1.) < 1.e-8)
    assert(abs(P_joint.sum() - 1.) < 1.e-8)

    # compute MI, ignore 0 probabilities to avoid nan
    P_joint_masked = np.ma.masked_where(P_joint==0.0, P_joint)
    mutual_info = -np.sum(P_data1[P_data1>0]*np.log2(P_data1[P_data1>0])) \
                  -np.sum(P_data2[P_data2>0]*np.log2(P_data2[P_data2>0])) \
                  +np.sum(P_joint_masked*np.log2(P_joint_masked))

    #print('MI = ', mutual_info)
    return mutual_info

# --------------------------------------------------------------------------------

def calc_trans_entropy(data1, threshold, threshold_dir, data2, dt, direction):
    """
    Calculated one-sided transfer entropy between data1 and data2    
    Input: data1(1D array), threshold(to neglect some outliners in data1),      
           threshold_dir(direction of threshold, 1 -- one-sided, 2 -- two-sided),
           data2(1D array), dt(time delay for data2),
           direction (1: data1 -> data2 or 2:data2 -> data1)
    Note:  dt = the number of indices to shift upon data2. Example: for 
           15-min TEC data, the delay time = dt*15min.
    Output: transfer entropy between data1 and time-delayed data2
    References: Schreiber2000; Balasis2013; Wing2016

    """
    assert(data1.shape == data2.shape)    

    # optimal binning for a normal distribution (Sturges 1926)
    nbins = int(np.log2(len(data1))) + 1
    #print('... calc_trans_entropy: nbins = ', nbins)

    if direction == 1:
        # driver to iono TEC
        # skip outliner data in data1, same to data2
        if threshold_dir == 1:
            data3 = data1[data1<threshold]
            data4 = data2[data1<threshold]
        elif threshold_dir == 2:
            data3 = data1[abs(data1)<threshold]
            data4 = data2[abs(data1)<threshold]
    elif direction == 2:
        # iono TEC to driver
        if threshold_dir == 1:
            data4 = data1[data1<threshold]
            data3 = data2[data1<threshold]
        elif threshold_dir == 2:
            data4 = data1[abs(data1)<threshold]
            data3 = data2[abs(data1)<threshold]
    ndata = len(data3)
    assert(len(data4) == ndata)

    # calculate probablities
    hist2d, bin_edges1, bin_edges2 = np.histogram2d(data3, data4, bins=nbins)
    ### FOR omega=1
    #hist2d, bin_edges1, bin_edges2 = np.histogram2d(data3[1:ndata-dt], \
    #                                                data4[dt:ndata-1], bins=nbins)
    ### 
    P_data3_data4 = hist2d/hist2d.sum() #joint probability of data1 and data2
    hist2d, bin_edges1, bin_edges2 = np.histogram2d(data4[dt:], data4[:ndata-dt], \
                                                    bins=nbins)
    ### FOR omega=1
    #hist2d, bin_edges1, bin_edges2 = np.histogram2d(data4[dt+1:], data4[dt:ndata-1], \
    #                                                bins=nbins)
    ###
    P_data4dt_data4 = hist2d/hist2d.sum() #joint probability of delayed data2 and data2
    histdd, bin_edges = np.histogramdd((data3[:ndata-dt], data4[dt:], \
                                        data4[:ndata-dt]), bins=nbins)
    ### FOR omega=1
    #histdd, bin_edges = np.histogramdd((data3[1:ndata-dt], data4[dt+1:], \
    #                                    data4[dt:ndata-1]), bins=nbins)
    ###
    P_data3_data4dt_data4 = histdd/histdd.sum() #joint probability of all three
    hist, bin_edges = np.histogram(data4, bins=nbins, density=True)
    ### FOR omega=1
    #hist, bin_edges = np.histogram(data4[dt:ndata-1], bins=nbins, density=True)
    ###
    P_data4 = hist/hist.sum() # probablity of data2

    # check to make sure the sum of the probablities is almost 1
    assert(abs(P_data3_data4.sum() - 1.) < 1.e-8)
    assert(abs(P_data4dt_data4.sum() - 1.) < 1.e-8)
    assert(abs(P_data3_data4dt_data4.sum() - 1.) < 1.e-8)
    assert(abs(P_data4.sum() - 1.) < 1.e-8)
    
    # TE = CMI(data1, data2(dt)|data2) = CMI(data2(dt), data1|data2) for driver to iono
    # ignore zero probability data points
    P_34joint = np.ma.masked_where(P_data3_data4==0.0, P_data3_data4)
    P_44joint = np.ma.masked_where(P_data4dt_data4==0.0, P_data4dt_data4)
    P_344joint = np.ma.masked_where(P_data3_data4dt_data4==0.0, P_data3_data4dt_data4)
    trans_entropy = -np.sum(P_34joint*np.log2(P_34joint)) \
                  -np.sum(P_44joint*np.log2(P_44joint)) \
                  +np.sum(P_344joint*np.log2(P_344joint)) \
                  +np.sum(P_data4[P_data4>0]*np.log2(P_data4[P_data4>0]))

    return trans_entropy

# --------------------------------------------------------------------------------

def calc_te_shuffle(npermutations, data1, threshold, threshold_dir, data2, dt):
    """
    Calculated one-sided transfer entropy from shuffled data1 to data2, for
    computing effective transfer entropy ETE_1->2 = TE_1->2 - TE_shuffled1->2.
    Input: npermutations = number of times to randomly shuffle data1
           data1(1D array), threshold(to neglect some outliners in data1),
           threshold_dir(direction of threshold, 1 -- one-sided, 2 -- two-sided),
           data2(1D array), dt(time delay for data2).
    Note:  dt = the number of indices to shift upon data2. Example: for
           15-min TEC data, the delay time = dt*15min. 
    Output: mean and 3-sigma of transfer entropies between shuffled data1 and 
            time-delayed data2.
    References: https://cran.r-project.org/web/packages/RTransferEntropy/
                vignettes/transfer-entropy.html

    """

    te = np.zeros(npermutations)
    for ipermutation in range(npermutations):
        data1_shuffled = np.random.permutation(data1)
        te[ipermutation] = calc_trans_entropy(data1_shuffled, threshold, \
                                              threshold_dir, data2, dt, 1)
    
    te_mean = np.mean(te)
    te_3sigma = 3*np.std(te)

    #print('te = ',te)
    #print('mean, 3*std = ',te_mean,te_3sigma)
    
    return te_mean, te_3sigma

# --------------------------------------------------------------------------------

def calc_entropy_rate(data, dt):
    """
    Calculate entropy rate for data
    Input: data(1D array), dt(time delay for data),
    Note:  dt = the number of indices to shift upon data. Example: for 
           15-min TEC data, the delay time = dt*15min.
    Output: entropy rate of data
    References: 2016 book "an introduction to transfer entropy" chapter 4.2.2

    """
    ndata = len(data)
    
    # optimal binning for a normal distribution (Sturges 1926)
    nbins = int(np.log2(ndata)) + 1

    # calculate probablities
    ### omega = 1
    #hist2d, bin_edges1, bin_edges2 = np.histogram2d(data[dt+1:], data[dt:ndata-1], \
    #                                                bins=nbins)
    ###
    ### omega = tau (dt)
    hist2d, bin_edges1, bin_edges2 = np.histogram2d(data[dt:], data[:ndata-dt], \
                                                    bins=nbins)
    P_datadt_data = hist2d/hist2d.sum() #joint probability of delayed data and data
    ###omega=1
    #hist, bin_edges = np.histogram(data[dt:ndata-1], bins=nbins, density=True) 
    ###
    ### omega = tau
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)  
    P_data = hist/hist.sum() # probablity of data

    # check to make sure the sum of the probablities is almost 1
    assert(abs(P_datadt_data.sum() - 1.) < 1.e-8)
    assert(abs(P_data.sum() - 1.) < 1.e-8)
    
    # Entropy Rate H' = H(datadt|data) = H(datadt,data) - H(data)
    # ignore zero probability data points
    P_joint = np.ma.masked_where(P_datadt_data==0.0, P_datadt_data)
    entropy_rate = -np.sum(P_joint*np.log2(P_joint)) \
                  +np.sum(P_data[P_data>0]*np.log2(P_data[P_data>0]))

    return entropy_rate

# --------------------------------------------------------------------------------

def write_info(datain, label, yearspan, outdir):
    """
    Write MI or TE into ASCII data files
    Input: datain can be MI or TE
           label = [A, B, C] where A = "MI" or "TE", 
               B = "solarwind" or "f107",
               C = "15min" or "daily"
           yearspan: a string containg the years coverd by datain
           outdir: directory to save the output file

    """
    (ntau, ndrivers, ntec) = datain.shape
    filename = outdir + '/' + label[0] + '_' + label[1] + label[2] + \
               '_'+ yearspan + '.dat'
    print('Writing to file ', filename)

    # names of TEC intensification parameters
    if label[2] == 'daily':
        params = ['num_max','num_avg','totalREC_max','totalREC_avg', \
                  'totalsize_max', 'totalsize_avg']
    elif label[2] == '15min':
        params = ['num_of_intensfn', 'total_REC', 'total_relative_size']
    else:
        print('label not recognized by write_info: ', label[2])
    with open(filename,'w') as outfile:
        if label[0] == 'MI':
            outfile.write('#Mutual information between '+label[2]+label[1]+' and ' +
                          'time-shifted TEC intensification parameters for '+yearspan+'\n')
        elif label[0] == 'TE':
            outfile.write('#Transfer entropy from '+label[2]+label[1]+' to ' +
                          'time-shifted TEC intensification parameters for '+yearspan+'\n')
        outfile.write('#Array shape (ntau, ndrivers, ntec) = (' + \
                      str(ntau)+','+str(ndrivers)+','+str(ntec)+')\n')
        if label[1] == 'solarwind':
            outfile.write('#Bmag Bx By Bz Vmag Vx Vy Vz Density ' +
                          'Temperature Flow_pressure Ey\n')
        elif label[1] == 'f107':
            outfile.write('#F107\n')
        else:
            print('label not recognized by write_info: ', label[1])

        for itec in range(0,ntec):
            outfile.write('For ' + params[itec] + '\n')
            for itau in range(0,ntau):
                # this is not as efficient as datain[:,isw,itec], but more friendly
                # for human reading the data in the file
                for idriver in range(0,ndrivers):
                    outfile.write('{:12.5f}'.format(datain[itau,idriver,itec]))
                outfile.write('\n')

# --------------------------------------------------------------------------------

def write_info_shuffle(datain1, datain2, label, yearspan, outdir):
    """
    Write mean and 3 sigma for shuffled TE into ASCII data files
    Input: datain1 = mean, datain2 = 3sigma
           label = [A, B, C] where A = "TEshuffle", 
               B = "solarwind" or "f107",
               C = "15min" or "daily"
           yearspan: a string containg the years coverd by datain
           outdir: directory to save the output file

    """
    (ntau, ndrivers, ntec) = datain1.shape
    assert(datain1.shape == datain2.shape)

    filename = outdir + '/' + label[0] + '_' + label[1] + label[2] + \
               '_'+ yearspan + '.dat'
    print('Writing to file ', filename)

    # names of TEC intensfn parameters
    if label[2] == 'daily':
        params = ['num_max','num_avg','totalREC_max','totalREC_avg', \
                  'totalsize_max', 'totalsize_avg']
    elif label[2] == '15min':
        params = ['num_of_intensfn', 'total_REC', 'total_relative_size']
    else:
        print('label not recognized by write_info: ', label[2])
    with open(filename,'w') as outfile:
        if label[0] == 'TEshuffle':
            outfile.write('#Mean and 3 Sigma of transfer entropy from shuffled ' + \
                          label[2]+label[1]+' to ' +
                          'time-shifted TEC intensification parameters for '+yearspan+'\n')
        outfile.write('#Array shape (ntau, ndrivers, ntec) = (' + \
                      str(ntau)+','+str(ndrivers)+','+str(ntec)+') ' + \
                      'for both mean and 3sigma\n')        
        if label[1] == 'solarwind':
            outfile.write('#Bmag(mean 3sigma) Bx By Bz Vmag Vx Vy Vz Density ' +
                          'Temperature Flow_pressure Ey\n')
        elif label[1] == 'f107':
            outfile.write('#F107(mean 3sigma)\n')
        else:
            print('label not recognized by write_info: ', label[1])
        for itec in range(0,ntec):
            outfile.write('For ' + params[itec] + '\n')
            for itau in range(0,ntau):
                # this is not as efficient as datain[:,isw,itec], but more friendly
                # for human reading the data in the file
                for idriver in range(0,ndrivers):
                    outfile.write('{:12.5f}{:12.5f}'. \
                                  format(datain1[itau,idriver,itec], \
                                         datain2[itau,idriver,itec]))
                outfile.write('\n')

# --------------------------------------------------------------------------------

def write_entropy_rate(datain, label, yearspan, outdir):
    """
    Write ER into ASCII data files
    Input: datain is ER
           label = "15min" or "daily"
           yearspan: a string containg the years coverd by datain
           outdir: directory to save the output file

    """
    (ntau, ntec) = datain.shape
    filename = outdir + '/ER_' + label + '_'+ yearspan + '.dat'
    print('Writing to file ', filename)

    # names of TEC intensfn parameters
    if label == 'daily':
        params = ['num_max','num_avg','totalREC_max','totalREC_avg', \
                  'totalsize_max', 'totalsize_avg']
    elif label == '15min':
        params = ['num_of_intensfn', 'total_REC', 'total_relative_size']
    with open(filename,'w') as outfile:
        outfile.write('#Entropy rate for '+label + \
                      ' TEC intensification parameters for '+yearspan+'\n')
        outfile.write('#Array shape (ntau,ntec) = (' + \
                      str(ntau)+','+str(ntec)+')\n')
        for itec in range(0,ntec):
            outfile.write('For ' + params[itec] + '\n')
            for itau in range(0,ntau):
                outfile.write('{:12.5f}\n'.format(datain[itau,itec]))

# --------------------------------------------------------------------------------
                
def read_intensfn(years, indir):
    """
    Read 15-min tec intensfn parameters for years    
    Input: years -- list of years; indir -- data file directory
    Output: data_intensfn (15-min tec parameters)

    """

    # 15-min TEC intensification parameters, selected
    # use lists first, list append is faster than np.append
    # https://stackoverflow.com/questions/70144878/list-append-faster-than-np-append
    time_allyears = []   # t sequence every 15 mins, no repeat
    num_allyears = []
    totalrec_allyears = []
    totalsize_allyears = []
    
    t_prior = datetime(2000,1,1)
    for iyear, year in enumerate(years):
        # arrays to fill for this year only
        time = []
        num = []
        totalrec = []
        totalsize = []
        print('Reading TEC intensification data for year '+str(year))
        f1 = open(indir + '/intensifications_'+str(year)+'.dat','r')
        for line in f1:
            if line[0] == '#':
                continue
            record = line.split()
            t = datetime.strptime(' '.join(record[0:6]),"%Y %m %d %H %M %S")
            num.append(int(record[6]))
            rec = float(record[12])
            size = float(record[13])

            if t == t_prior:
                num.pop(-2)
                totalrec[-1] = totalrec[-1] + rec
                totalsize[-1] = totalsize[-1] + size
            else:
                time.append(t)
                totalrec.append(rec)
                totalsize.append(size)
            t_prior = t
        f1.close()

        assert(len(time) == len(num))
        assert(len(time) == len(totalrec))
        assert(len(time) == len(totalsize))
        print('len(time)=', len(time))
        
        # append this year to all years of data
        time_allyears.extend(time)
        num_allyears.extend(num)
        totalrec_allyears.extend(totalrec)
        totalsize_allyears.extend(totalsize)
        
    data_intensfn = np.column_stack((num_allyears, \
                                     totalrec_allyears, \
                                     totalsize_allyears))
    print('data_intensfn.shape = ', data_intensfn.shape)

    return data_intensfn
        
# --------------------------------------------------------------------------------

def read_intensfn_daily(years, yearspan, indir):
    """
    Read daily tec intensification parameters for years
    Input: years -- list of years; yearspan -- yyyytoyyyy; 
           indir -- data file directory
    Output: data_intensfn
 
    """

    # daily TEC intensification parameters
    time_daily = []
    num_max_daily = []
    num_avg_daily = []
    rectotal_max_daily = []
    rectotal_avg_daily = []
    sizetotal_max_daily = []
    sizetotal_avg_daily = []
    
    f1 = open(indir + '/intensifications_daily_' + yearspan + '.dat','r')
    for line in f1:
        if line[0] == '#':
            continue
        record = line.split()
        t = datetime.strptime(' '.join(record[0:3]),"%Y %m %d")
        if t.year in years:
            time_daily.append(t)
            num_max_daily.append(int(record[3]))
            num_avg_daily.append(float(record[4]))
            rectotal_max_daily.append(float(record[5]))
            rectotal_avg_daily.append(float(record[6]))
            sizetotal_max_daily.append(float(record[7]))
            sizetotal_avg_daily.append(float(record[8]))
    f1.close()

    # fill 2D arrays
    data_intensfn = np.column_stack((num_max_daily,num_avg_daily, \
                                     rectotal_max_daily,rectotal_avg_daily, \
                                     sizetotal_max_daily,sizetotal_avg_daily))
    print('data_intensfn.shape = ', data_intensfn.shape)

    return data_intensfn

# --------------------------------------------------------------------------------

def read_driver(years, indir):
    """
    Read 15-min solar wind data for years
    Input: years, indir
    Output: data_driver

    """

    time_sw_allyears = []
    Bmag_allyears = []
    Bx_allyears = []
    By_allyears = []
    Bz_allyears = []
    Vmag_allyears = []
    Vx_allyears = []
    Vy_allyears = []
    Vz_allyears = []
    den_allyears = []
    tem_allyears = []
    pressure_allyears = []
    Ey_allyears = []

    for iyear, year in enumerate(years):        
        # arrays to fill for this year, significantly speeds up from
        # filling allyears to the same array
        time_sw = []
        Bmag = []
        Bx = []
        By = []
        Bz = []
        Vmag = []
        Vx = []
        Vy = []
        Vz = []
        den = []
        tem = []
        pressure = []
        Ey = []
        print('Reading solar wind data for year '+str(year))
        f2 = open(indir + '/solarwind_avg15min_'+str(year)+'.dat','r')
        for line in f2:
            if line[0] == '#':
                continue
            record = line.split()
            t = datetime.strptime(' '.join(record[0:5]),'%Y %m %d %H %M')
            # manually remove doy303/year2003 data to be consistent with JPLD TEC data
            if t.year == 2003 and t.month == 10 and t.day == 31:
                continue
            time_sw.append(t)
            Bmag.append(float(record[5]))
            Bx.append(float(record[6]))
            By.append(float(record[7]))
            Bz.append(float(record[8]))
            Vmag.append(float(record[9]))
            Vx.append(float(record[10]))
            Vy.append(float(record[11]))
            Vz.append(float(record[12]))
            den.append(float(record[13]))
            tem.append(float(record[14]))
            pressure.append(float(record[15]))
            # Electric field = -V(km/s) * Bz (nT; GSM) * 10**-3
            Ey.append(float(record[16]))
        f2.close()
        # append this year to all years of data
        time_sw_allyears.extend(time_sw)
        Bmag_allyears.extend(Bmag)
        Bx_allyears.extend(Bx)
        By_allyears.extend(By)
        Bz_allyears.extend(Bz)
        Vmag_allyears.extend(Vmag)
        Vx_allyears.extend(Vx)
        Vy_allyears.extend(Vy)
        Vz_allyears.extend(Vz)
        den_allyears.extend(den)
        tem_allyears.extend(tem)
        pressure_allyears.extend(pressure)
        Ey_allyears.extend(Ey)
            
    data_driver = np.column_stack((Bmag_allyears,Bx_allyears,By_allyears, \
                                   Bz_allyears,Vmag_allyears, \
                                   Vx_allyears,Vy_allyears,Vz_allyears, \
                                   den_allyears,tem_allyears,pressure_allyears, \
                                   Ey_allyears))
    print('data_driver.shape = ', data_driver.shape)

    return data_driver

# --------------------------------------------------------------------------------

def read_driver_daily(years, indir):
    """
    Read daily F10.7 data
    Input: years -- list of years; 
           indir -- data file directory
    Output: data_driver

    """

    time_f107 = []
    f107 = []

    f2 = open(indir + '/cls_radio_flux_f107.csv','r')
    for line in f2:
        if line[0] == 'D' or line[0] == 't':
            continue
        record = line.split(',')
        date_f107 = datetime.strptime(record[0][0:19],'%Y-%m-%dT%H:%M:%S')
        if date_f107 == datetime(2003,10,31):
            continue  # no intensification data on this day
        yr = date_f107.year
        if yr in years:
            time_f107.append(date_f107)
            f107.append(float(record[2]))
    f2.close()
    ntimes_f107 = len(time_f107)
    print('ntimes_f107=',ntimes_f107) 

    data_driver = np.column_stack((f107,))
    
    print('data_driver.shape = ', data_driver.shape)

    return data_driver


# --------------------------------------------------------------------------------
# End of defined functions
# --------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--beginyear', dest='beginyear', required=True, \
                        help='begin year of data', type=int)
    parser.add_argument('--endyear', dest='endyear', required=True, \
                        help='end year of data', type=int)
    parser.add_argument('-i', '--indir', dest='indir', \
                        help='directory of input F10.7, solar wind, and ' + \
                        'TEC intensification data', \
                        default='../sample_datain')
    parser.add_argument('-o', '--outdir', dest='outdir', \
                        help='directory to store output MI, TE, ER data files', \
                        default='../sample_dataout')
    parser.add_argument('--datafreq', dest='datafreq', \
                        help='frequency of input data, can be daily ' + \
                        '(default) or 15min', \
                        default='daily')
    parser.add_argument('--driver', dest='driver', \
                        help='f107 (default) or solarwind', \
                        default='f107')
    parser.add_argument('--maxdelay', dest='maxdelay', type=int, required=True, \
                        help='Maximum delay time tau in number of data points, ' + \
                        'recommend 365 (365 days) for daily data and ' + \
                        '480 (5 days) for 15-min data.')
    parser.add_argument('--npermu', dest='npermu', type=int, \
                        help='number of times to shuffle driver data for ' + \
                        'computing TEshuffle, default 100', \
                        default=100)

    args = parser.parse_args()

    beginyear = args.beginyear
    endyear = args.endyear
    years = list(range(beginyear,endyear+1))
    stryearspan = str(beginyear) + 'to' + str(endyear)
    indir = args.indir
    outdir = args.outdir
    if not path.exists(outdir):
        makedirs(outdir)
    label = [args.driver, args.datafreq]
    ntaus = args.maxdelay
    npermutations = args.npermu
    
    if label[1] == '15min':
        data_intensfn = read_intensfn(years, indir)
        data_driver = read_driver(years, indir)
    elif label[1] == 'daily':
        data_intensfn = read_intensfn_daily(years, stryearspan, indir)
        data_driver = read_driver_daily(years, indir)
    #print(data_intensfn.shape, data_driver.shape)
    
    # driver data thresholds, data above/below threshold will be ignored
    if label[0] == 'solarwind':
        #threshold_sw = [9999]*4 + [99999]*4 + [999, 9999990, 99, 999] # unlimited range
        threshold = [25,15,20,20,99999,99999,100,100,30,7e5,15,10]
        threshold_dir = [1,2,2,2,1,1,2,2,1,1,1,2]
    elif label[0] == 'f107':
        threshold = [300]
        threshold_dir = [1]

    # Calculate MI and TE between upper drivers and time-shifted TEC intensification parameters
    ndrivers = data_driver.shape[1]
    ntec = data_intensfn.shape[1] #number of tec intensfn parameters        
    MI = np.zeros((ntaus,ndrivers,ntec))
    TE = np.zeros((ntaus,ndrivers,ntec))
    TE_shuffle_mean = np.zeros((ntaus,ndrivers,ntec))
    TE_shuffle_3sigma = np.zeros((ntaus,ndrivers,ntec))
    for itec in range(0,ntec):
        print('... Working on TEC intensification parameter ', itec)
        for idriver in range(0,ndrivers):
            print('...... Working on driver ', idriver)
            for itau in range(0,ntaus): 
                #print('... Working on tau = ', itau)
                MI[itau,idriver,itec] = calc_mutual_info(data_driver[:,idriver], \
                                                         threshold[idriver], \
                                                         threshold_dir[idriver], \
                                                         data_intensfn[:,itec], itau)
                TE[itau,idriver,itec] = calc_trans_entropy(data_driver[:,idriver], \
                                                           threshold[idriver], \
                                                           threshold_dir[idriver], \
                                                           data_intensfn[:,itec], itau, 1)
                TE_shuffle_mean[itau,idriver,itec], \
                    TE_shuffle_3sigma[itau,idriver,itec] = calc_te_shuffle(npermutations, \
                                                                           data_driver[:,idriver], \
                                                                           threshold[idriver], \
                                                                           threshold_dir[idriver], \
                                                                           data_intensfn[:,itec], \
                                                                           itau)
                                
            
    # write MI and TE into ascii files
    write_info(MI, ['MI'] + label, stryearspan, outdir)
    write_info(TE, ['TE'] + label, stryearspan, outdir)
    write_info_shuffle(TE_shuffle_mean, TE_shuffle_3sigma, \
                       ['TEshuffle'] + label, stryearspan, outdir)


    # Calculate entropy rate for TEC intensification parameters
    ER = np.zeros((ntaus,ntec))
    for itec in range(0,ntec):
        print('... Computing ER for TEC intensification parameter ', itec)
        for itau in range(0,ntaus):
            ER[itau,itec] = calc_entropy_rate(data_intensfn[:,itec], itau)
            
    # write ER into ascii files                                                                            
    write_entropy_rate(ER, label[1], stryearspan, outdir)


