#!/usr/bin/env python3
'''
PURPOSE:
    This is companion plotting script for calc_information.py
    This script plots MI, TE, TEshuffle, ER, NTE as functions of 
    the delay time

AUTHOR:
    Xing Meng

LICENSE:
    Copyright 2023 California Institute of Technology                                             
    Licensed under the Apache License, Version 2.0 (the "License");                               
    you may not use this code except in compliance with the License.                              
    You may obtain a copy of the License at                                                       
    http://www.apache.org/licenses/LICENSE-2.0
             
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# --------------------------------------------------------------------------------
# Defined functions
# --------------------------------------------------------------------------------

def read_info(filename):
    """
    Read MI or TE or TEshuffle or ER values from a file to an array
    Input: filename (with directory included)
    Output: data (containing MI or TE values)
    Note: for TEshuffle files, output data includes both mean and 3 sigma of TE
          shuffle. ndriver is twice of the actual number of driver parameters.
    
    """
    ntec = 0
    ndriver = 0
    ntau = 0
    IsFirstRecord = 1
    # first loop through the file to find out ntau, ndriver, ntec
    f = open(filename,'r')    
    for line in f:
        if line[0] == '#':
            continue
        if line[0:3] == 'For':
            ntec = ntec + 1
            continue
        if IsFirstRecord:
            record = line.split()
            ndriver = len(record)
            IsFirstRecord = 0
        if ntec == 1:
            ntau = ntau + 1
    f.close()
    print('ntau, ndriver, ntec = ', ntau, ndriver, ntec)

    # now read the data
    data = np.zeros((ntau,ndriver,ntec))
    itau = 0
    itec = 0
    f = open(filename,'r')    
    for line in f:
        if line[0] == '#':
            continue
        if line[0:3] == 'For':
            itec = itec + 1
            itau = 0
            continue
        record = line.split()
        data[itau,:,itec-1] = [float(i) for i in record]
        itau = itau + 1
    f.close()
    print('itec, itau = ', itec, itau)

    return data
    
# --------------------------------------------------------------------------------

def plot_info_lines(datain, datain2, driver, label, yearspan, timestep, xmax):
    """
    Make line plots of datain, a 3D array of size (ntau,ndriver,ntec)
    Input: datain can be MI or TE or ER 3D arrays,
           datain2: mean and 3sigma of TEshuffle, only used when label='TECwshuffle'
           driver = 'f107daily' or 'solarwind15min'
           label = "MI" or "TE" or "TEwshuffle" or "ER"
           yearspan: a string containg the years coverd by datain
           timestep: 0.25hr for 15-min data, 1day for daily data
           xmax: the maximum delay time to plot, in unit hr or day; if
                 -1, then plot all delay times in datain

    """
    print('Ploting ' + label)

    (ntau, ndriver, ntec) = datain.shape
    print(ntau, ndriver, ntec)
        
    # the horizontal axis is delay time
    tau = np.array(range(0,ntau))*timestep  # in unit of hour or day

    # names of tec intensfn and driver parameters
    if driver == 'solarwind15min':
        driverparam = ['|B|','Bx','By','Bz','|V|','Vx','Vy','Vz','Density', \
                       'Temperature','Flow pressure','Ey']
        tecparam = ['num_of_intensfn', 'total_REC', 'total_relative_size']
    elif driver == 'f107daily':
        driverparam =['F107']
        tecparam = ['num_max','num_avg','totalREC_max','totalREC_avg', \
                    'totalsize_max', 'totalsize_avg']
        
    # colors representing different driver parameters, sufficient for 13 params
    colors = ['C'+str(i) for i in range(0,10)] + ['k', 'tan','gold']

    # determine the figure title based on label
    if label == 'MI':
        title = 'Mutual Information between '+driver+ \
            ' and TEC Intensification Parameters for '
        figname = label+'_'+driver+'_'+yearspan
    if label == 'TE' or label == 'TEwshuffle':
        title = 'Transfer Entropy from '+driver+ \
            ' to TEC Intensification Parameters for '
        figname = label+'_'+driver+'_'+yearspan
    if label == 'ER':
        title = 'Entropy Rate for '+ driver[-5:] + \
            ' TEC Intensification Parameters for '
        figname = label+'_'+driver[-5:]+'_'+yearspan
        
    title = title + yearspan

    # determine the timeunit and panel layout for horizontal axis
    if driver[-5:] == 'daily':
        unit = 'day'
        ncols = 2
        nrows = 3
    else:
        unit = 'hour'
        ncols = 2
        nrows = 2

    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7    
    plt.figure(figsize=(8,8))
    for itec in range(0,ntec):
        plt.subplot(nrows,ncols,itec+1)
        for idriver in range(0,ndriver):
            plt.plot(tau, datain[:,idriver,itec], color=colors[idriver], \
                     label=driverparam[idriver])
        if itec == 0:
            plt.legend()

        if label == 'TEwshuffle':
            # now datain2 should have doubled sized of nparam for 2nd dimension
            # containing the mean and 3 sigma of TE shuffle
            for idriver in range(0,ndriver):
                # mean
                plt.plot(tau, datain2[:,2*idriver,itec], '--', color=colors[idriver])
                # mean + 3sigma
                plt.plot(tau, datain2[:,2*idriver,itec]+datain2[:,2*idriver+1,itec],\
                         ':', color=colors[idriver])
                # mean - 3sigma
                plt.plot(tau, datain2[:,2*idriver,itec]-datain2[:,2*idriver+1,itec],\
                         ':', color=colors[idriver])
                            
        if xmax != -1:
            # user specified delay time range to plot
            plt.xlim([0,xmax])

        plt.xlabel('Delay Time ['+unit+']')
        plt.ylabel('Information [bits]')
        plt.title(tecparam[itec])
    plt.figtext(0.5,0.95,title,horizontalalignment='center')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if xmax == -1:
        plt.savefig(figname + '.pdf')
    else:
        # add to filename max xlim if not all dely times are plotted
        plt.savefig(figname + '_xlim'+str(xmax)+'.pdf')
    plt.close()

# --------------------------------------------------------------------------------

def plot_nte(datain1, datain2, datain3, driver, yearspan, timestep, xmax):
    
    """
    Make line plots of normalized TE for 15-min data
    along with the 3-sigma from TE of shuffled driver data
    normalized TE = (TE - mean TE of shuffled driver)/ER,                                       
    Input: datain1: TE 3D arrays, (ntau,nparam,ntec)
           datain2: mean and 3sigma of TEshuffle
           datain3: entropy rate arrays (ntau,1,ntec) 
           driver = 'f107daily' or 'solarwind15min'
           yearspan: a string containg the years coverd by datain
           timestep: 0.25hr for 15-min data, 1day for daily data
           xmax: the maximum delay time to plot, in unit hr or day; if
                 -1, then plot all delay times in datain1
    """

    (ntau, ndriver, ntec) = datain1.shape
    print(ntau, ndriver, ntec)

    tau = np.array(range(0,ntau))*timestep

    # names of tec and driver parameters
    if driver == 'solarwind15min':
        driverparam = ['|B|','Bx','By','Bz','|V|','Vx','Vy','Vz','Density', \
                       'Temperature','Flow pressure','Ey']
        tecparam = ['num_of_intensfn', 'total_REC', 'total_relative_size']
    elif driver == 'f107daily':
        driverparam =['F107']
        tecparam = ['num_max','num_avg','totalREC_max','totalREC_avg', \
                    'totalsize_max', 'totalsize_avg']
        
    # colors representing different driver parameters
    colors = ['C'+str(i) for i in range(0,10)] + ['k', 'tan','gold']

    # plot title
    title = 'Normalized Transfer Entropy from ' + driver + ' to ' + \
        'TEC Intensifications for ' + yearspan

    # determine the timeunit and panel layout for horizontal axis
    if driver[-5:] == 'daily':
        unit = 'day'
        ncols = 2
        nrows = 3
    else:
        unit = 'hour'
        ncols = 2
        nrows = 2

    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7    
    plt.figure(figsize=(8,8))

    # number of data points to plot along x (num of time steps)
    if xmax != -1:
        ntaumax = xmax/timestep
    else:
        ntaumax = ntau 

    for itec in range(0,ntec):
        nte_selected = np.zeros([2,ndriver])
        delta_nte_selected = np.zeros([2,ndriver])
        nte_max = np.zeros([ndriver])
        tau_nte_max = np.zeros([ndriver])
        delta_nte_max = np.zeros([ndriver])

        plt.subplot(nrows,ncols,itec+1)
        for idriver in range(0,ndriver):
            # for TE formula 1  omega = tau
            # set nte[0] and delta_nte[0] to 0 because datain1, datain2 and
            # datain3 (ER) are zero for tau = 0
            nte = np.zeros(ntaumax)
            delta_nte = np.zeros(ntaumax)
            nte[1:] = (datain1[1:ntaumax,idriver,itec]-datain2[1:ntaumax,2*idriver,itec]) \
                      /datain3[1:ntaumax,0,itec]            
            delta_nte[1:] = datain2[1:ntaumax,2*idriver+1,itec]/datain3[1:ntaumax,0,itec]

            nte_selected[0,idriver] = nte[2]
            delta_nte_selected[0,idriver] = delta_nte[2]
            nte_selected[1,idriver] = nte[90]
            delta_nte_selected[1,idriver] = delta_nte[90]            
            if idriver < 6 and idriver > 0:
                # for sw drivers, nte is tiny with relatively large errors
                # after 5 days. If not restricting to less than 5 days,
                # the nte_max will end up at tau > 50 for TECmax and GEC
                nte_max[idriver] = np.amax(nte[:5])
                tau_nte_max[idriver] = np.argmax(nte[:5])
                delta_nte_max[idriver] = delta_nte[np.argmax(nte[:5])]
            else:
                nte_max[idriver] = np.amax(nte[:])
                tau_nte_max[idriver] = np.argmax(nte[:])
                delta_nte_max[idriver] = delta_nte[np.argmax(nte[:])]
            
            plt.plot(tau[:ntaumax], nte, linewidth=1.2, \
                     color=colors[idriver], label=driverparam[idriver])
            # shade the +/- 3sigma region
            # linewidth=0.0 ensures no lines are drawn
            plt.fill_between(tau[:ntaumax], nte-delta_nte, nte+delta_nte, \
                             color=colors[idriver], alpha=0.4, linewidth=0.0)

            
        if xmax != -1:
            # user specified delay time range to plot
            plt.xlim([0,xmax])
        else:
            plt.xlim([0,ntau*timestep])
            
        # display legend
        if itec == 0:
            plt.legend(fontsize=6,ncol=2, \
                       labelspacing=0.02,borderpad=0.3)
        plt.xlabel(r'Time Lag $\tau$ [hour]')
        plt.ylabel('Normalized\nTransfer Entropy')
        plt.title(tecparam[itec])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.figtext(0.5,0.95,title,horizontalalignment='center')
    plt.savefig('NTE_'+driver+'_'+yearspan+'.pdf')
    plt.close()


# --------------------------------------------------------------------------------
# End of defined functions
# --------------------------------------------------------------------------------

years = '2003to2022'

#driver = 'f107daily'
#timestep = 1 #1 day

### Another Option
driver = 'solarwind15min'
timestep = 0.25 #0.25 hr
###

# read MI, TE, TEshuffle and ER from files
indir = '../sample_dataout/'
data_MI = read_info(indir+'MI_'+driver+'_'+years+'.dat')
data_TE = read_info(indir+'TE_'+driver+'_'+years+'.dat')
data_TEshuffle = read_info(indir+'TEshuffle_'+driver+'_'+years+'.dat')
data_ER = read_info(indir+'ER_'+driver[-5:]+'_'+years+'.dat')

print('data_ER.shape = ', data_ER.shape, \
      'data_MI.shape = ', data_MI.shape)
print('data_TE.shape = ', data_TE.shape, \
      'data_TEshuffle.shape = ', data_TEshuffle.shape)

# plot MI and TE
plot_info_lines(data_MI, data_MI, driver, 'MI', years, timestep, -1)
plot_info_lines(data_TE, data_TE, driver, 'TE', years, timestep, -1)
plot_info_lines(data_TE, data_TEshuffle, driver, 'TEwshuffle', years, timestep, -1)

# plot ER
plot_info_lines(data_ER, data_ER, driver, 'ER', years, timestep, -1)

# plot normalized transfer entropy (NTE)
plot_nte(data_TE, data_TEshuffle, data_ER, driver, years, timestep, -1)

