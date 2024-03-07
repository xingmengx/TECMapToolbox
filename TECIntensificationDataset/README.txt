This README.txt file was generated on 31 August 2023 by Xing Meng
Copyright 2023. California Institute of Technology. All Rights Reserved.

================================================================================
                                GENERAL INFORMATION
================================================================================
Title of Dataset: Total Electron Content Intensification Dataset for 
                  Years 2003 to 2022

Author information: X. Meng and O. P. Verkhoglyadova                  


Contact information:
Primary Contact: Xing Meng
		 Jet Propulsion Laboratory
                 California Institute of Technology
		 Pasadena, CA, USA
                 xing.meng@jpl.nasa.gov
                 
Alternative Contacts: Olga Verkhoglyadova
		      Jet Propulsion Laboratory
		      California Institute of Technology
		      Pasadena, CA, USA
                      olga.verkhoglyadova@jpl.nasa.gov

Funding sources and sponsorship:
This work was sponsored by NASA ROSES Living With a Star Tools and Methods 
Program (NNH21ZDA001N-LWSTM) and performed at the Jet Propulsion Laboratory, 
California Institute of Technology, under a contract with the NASA.


================================================================================
                                REPOSITORY OVERVIEW
================================================================================
Repository structure is as follows.

Dataset
    intensifications_[yyyy].dat 
	Total Electron Content (TEC) intensification data for yyyy, where yyyy 
        spans from 2003 to 2022.

    README.txt


================================================================================
                            DATA-SPECIFIC INFORMATION
================================================================================ 
The TEC intensification data file intensifications_[yyyy].dat contains the
TEC intensification characteristics identified from JPLD TEC maps
(https://sideshow.jpl.nasa.gov/pub/iono_daily/gim_for_research/jpld/) every 15 
minutes for year yyyy using feature extraction software. The file is in ASCII
format and contains a header and following columns: year, month, day, hour,
minute, second, index of intensification, TEC maximum within the intensification
in TECU, geographic latitude of the TEC maximum in degree, geographic longitude 
of the TEC maximum in degree, median TEC within the intensification region in TECU,
minimum TEC within the intensification region in TECU, Regional Electron Content of
the intensification in GECU, relative size of the intensification, number of GNSS
stations within/nearby the intensification.

