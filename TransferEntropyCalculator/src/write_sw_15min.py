#!/usr/bin/env python3
'''
PURPOSE:
    Write 15-min averaged solar wind and geomagnetic parameters including 
    n, v, B, T, as well as AE and SYM-H from 5-min OMNI data to 
    ascii files. One file per year for a user-specified year range.
    Plot the data for each year.

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
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--beginyear', dest='beginyear', required=True, \
                        help='begin year of data to fetch', type=int)
    parser.add_argument('--endyear', dest='endyear', required=True, \
                        help='end year of data to fetch', type=int)
    parser.add_argument('-i', '--indir', dest='indir', required=True, \
                        help='directory of input 5-min OMNI data')
    parser.add_argument('-o', '--outdir', dest='outdir', \
                        help='directory to store output 15-min OMNI data', \
                        default='../sample_datain')
    parser.add_argument('--plotdir', dest='plotdir', \
                        help='directory to save plots, default is ./plots', \
                        default='plots')
    
    args = parser.parse_args()
    beginyear = args.beginyear
    endyear = args.endyear
    indir = args.indir
    outdir = args.outdir
    if not path.exists(outdir):
        makedirs(outdir)
    plotdir = args.plotdir
    if not path.exists(plotdir):
        makedirs(plotdir)

    years = list(range(beginyear,endyear+1))    
    for iyear, year in enumerate(years):

        print('Reading OMNI data for year ', year)
        data = np.loadtxt(indir + '/omni_5min'+str(year)+'.asc')

        # compute 15-min average
        ntimes = (data.shape[0]//3)*3
        yr = data[::3,0]
        doy = data[::3,1]
        hr = data[::3,2]
        mn = data[::3,3]
        Bmag = np.mean(data[:,13][:ntimes].reshape(-1,3),axis=1)
        Bx = np.mean(data[:,14][:ntimes].reshape(-1,3),axis=1)
        By = np.mean(data[:,17][:ntimes].reshape(-1,3),axis=1) # in GSM
        Bz = np.mean(data[:,18][:ntimes].reshape(-1,3),axis=1) # in GSM
        V = np.mean(data[:,21][:ntimes].reshape(-1,3),axis=1)
        Vx = np.mean(data[:,22][:ntimes].reshape(-1,3),axis=1)
        Vy = np.mean(data[:,23][:ntimes].reshape(-1,3),axis=1) # in GSE
        Vz = np.mean(data[:,24][:ntimes].reshape(-1,3),axis=1) # in GSE
        den = np.mean(data[:,25][:ntimes].reshape(-1,3),axis=1)
        tem = np.mean(data[:,26][:ntimes].reshape(-1,3),axis=1)
        pressure = np.mean(data[:,27][:ntimes].reshape(-1,3),axis=1) # flow pressure
        Efield = np.mean(data[:,28][:ntimes].reshape(-1,3),axis=1)
        AE = np.mean(data[:,37][:ntimes].reshape(-1,3),axis=1)
        SYMH = np.mean(data[:,41][:ntimes].reshape(-1,3),axis=1)
        
        # replace large numbers with the "missing data value" same as OMNI raw data
        # the large numbers are resulted from at least 1 missing data within three
        # neighboring data points that are being averaged.
        Bmag[Bmag>3333.3] = 9999.99
        Bx[Bx>3000] = 9999.99
        By[By>3000] = 9999.99
        Bz[Bz>3000] = 9999.99
        V[V>33333.3] = 99999.9
        Vx[Vx>30000] = 99999.9
        Vy[Vy>30000] = 99999.9
        Vz[Vz>30000] = 99999.9
        den[den>333.3] = 999.99
        tem[tem>3333333] = 9999999.0
        pressure[pressure>33.3] = 99.99
        Efield[Efield>300] = 999.99
        AE[AE>33333] = 99999
        SYMH[SYMH>33333] = 99999
        
        # check array size
        assert(yr.shape == Bmag.shape)
        print('... yr.shape = ', yr.shape)
        
        # write to file
        swfile = outdir + '/solarwind_avg15min_'+str(year)+'.dat'
        with open(swfile,'w') as outfile:
            outfile.write('# 15-min average solar wind parameters and geomagnetic indices \n')
            outfile.write('# year month day hour minute Bmag[nT], Bx[nT] By_gsm[nT] ' + \
                          'Bz_gsm[nT] V[km/s] Vx[km/s] Vy_gse[km/s] Vz_gse[km/s] ' + \
                          'proton_density[n/cc] temperature[K] flow_pressure[nPa] ' + \
                          'Efield[mV/m] AE[nT] SYM-H[nT]\n')
            for itime in range(0,len(yr)):
                # convert doy to date
                date = datetime(int(yr[itime]),1,1) + timedelta(int(doy[itime])-1)
                outfile.write(('{:4d}'*5 + '{:8.2f}'*4 + '{:8.1f}'*4 + \
                               '{:7.2f}{:9.0f}{:6.2f}{:7.2f}{:6d}{:6d}\n').\
                              format(date.year,date.month,date.day,int(hr[itime]),int(mn[itime]), \
                                     Bmag[itime], Bx[itime], By[itime], Bz[itime], \
                                     V[itime], Vx[itime], Vy[itime], Vz[itime], den[itime], \
                                     tem[itime], pressure[itime], Efield[itime], \
                                     int(round(AE[itime])), int(round(SYMH[itime])))) #make AE and SYMH integers
                
        
        # make a timeseries plot
        # get the time array
        print('... plotting')
        t = []
        for itime in range(0,len(yr)):
            t = t + [datetime(int(yr[itime]),1,1,int(hr[itime]),int(mn[itime])) + \
                timedelta(int(doy[itime])-1)]
        plt.figure(1,figsize=(8,11))
        plt.subplot(811)
        # exclude missing data points, minimum 9999.99/3 (1 missing data point
        # and two neighbouring valid data points in 5-minute OMNI data)
        plt.plot(t,np.ma.masked_where(Bmag>9999, Bmag),'k', label='B')
        plt.plot(t,np.ma.masked_where(Bx>9999, Bx),'b', label='Bx')
        plt.plot(t,np.ma.masked_where(By>9999, By),'g', label='By')
        plt.plot(t,np.ma.masked_where(Bz>9999, Bz),'r', label='Bz')
        plt.legend()
        plt.xlim([t[0],t[-1]])
        plt.ylabel('B [nT]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.title('15-min average solar wind and geomagentic indices for year ' + str(year))
        
        plt.subplot(812)
        plt.plot(t,np.ma.masked_where(V>99999, V),'k', label='V')
        plt.plot(t,np.ma.masked_where(Vx>99999, Vx),'b', label='Vx')
        plt.plot(t,np.ma.masked_where(Vy>99999, Vy),'g', label='Vy')
        plt.plot(t,np.ma.masked_where(Vz>99999, Vz),'r', label='Vz')
        plt.legend()
        plt.xlim([t[0],t[-1]])
        plt.ylabel('V [km/s]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(813)
        plt.plot(t,np.ma.masked_where(den>999, den),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('n [/cc]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(814)
        plt.plot(t,np.ma.masked_where(tem>9999990, tem),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('T [K]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(815)
        plt.plot(t,np.ma.masked_where(pressure>99, pressure),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('P [nPa]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(816)
        plt.plot(t,np.ma.masked_where(Efield>999, Efield),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('E [mV/m]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(817)
        plt.plot(t,np.ma.masked_where(AE>99990, AE),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('AE [nT]',fontsize='small')
        plt.gca().axes.xaxis.set_ticklabels([])
        
        plt.subplot(818)
        plt.plot(t,np.ma.masked_where(SYMH>99990, SYMH),'k')
        plt.xlim([t[0],t[-1]])
        plt.ylabel('SYM-H [nT]',fontsize='small')
        plt.xlabel('Month in Year ' + str(year),fontsize='small')
        locs, labels = plt.xticks()
        assert(len(locs) == 12)
        mylabels = list(range(1,13))
        plt.gca().axes.xaxis.set_ticklabels(mylabels)
            
        plt.savefig(plotdir+'/solarwind_'+str(year)+'.pdf')
        plt.close()
