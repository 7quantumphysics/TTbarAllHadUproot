#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from coffea import util
import itertools
import os, sys
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import hist
import gc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
hep.style.use("CMS")

sys.path.append('../python/')
from functions import loadCoffeaFile, getLabelMap, getCoffeaFilenames, plotBackgroundEstimate, getHist


# ## Scale factors and IOV

useOldHTcut = False
useBlinding = True

lumi = {
    "2016APV": 19800.,
    "2016": 16120., #35920 - 19800
    "2016all": 35920,
    "2017": 41530./10.,
    "2018": 59800./10., #59740./10., #Blinding
    "Full": 46053. # 137190. Blinded
}

ttbar_xs1 = 831.76 * (0.09210) #pb For ttbar mass from 700 to 1000 | XSDB: 65.49 --> 14.5% difference
ttbar_xs2 = 831.76 * (0.02474) #pb For ttbar mass from 1000 to Inf | XSDB: 16.36 --> 20.5% difference

qcd_xs = {
    "300to470": 6806.0,
    "470to600": 551.2,
    "600to800": 156.7,
    "800to1000": 26.25,
    "1000to1400": 7.465,
    "1400to1800": 0.6487,
    "1800to2400": 0.08734,
    "2400to3200": 0.005237,
    "3200toInf": 0.0001352
}

systematics = [
        'jes',
        'jer',
        'pileup',
        'pdf',
        'q2',
        'btag',
        'toptagsf',
        'toptagxs',
        'lumi',
        'prefiring'
    ]


oldHTstr = ''
if useOldHTcut:
    oldHTstr = '_oldHTcut_oldBTag'
    

directories = [
    'images/png/closureTest/2016all',
    'images/png/closureTest/2016APV',
    'images/png/closureTest/2016',
    'images/png/closureTest/2017',
    'images/png/closureTest/2018',
    'images/png/closureTest/Full',
    'images/pdf/closureTest/2016all',
    'images/pdf/closureTest/2016APV',
    'images/pdf/closureTest/2016',
    'images/pdf/closureTest/2017',
    'images/pdf/closureTest/2018',
    'images/pdf/closureTest/Full',
    'images/png/kinematics/2016all',
    'images/png/kinematics/2016APV',
    'images/png/kinematics/2016',
    'images/png/kinematics/2017',
    'images/png/kinematics/2018',
    'images/png/kinematics/Full',
    'images/pdf/kinematics/2016all',
    'images/pdf/kinematics/2016APV',
    'images/pdf/kinematics/2016',
    'images/pdf/kinematics/2017',
    'images/pdf/kinematics/2018',
    'images/pdf/kinematics/Full',
    'images/png/massmodCompare/2016all',
    'images/png/massmodCompare/2016APV',
    'images/png/massmodCompare/2016',
    'images/png/massmodCompare/2017',
    'images/png/massmodCompare/2018',
    'images/png/massmodCompare/Full',
    'images/pdf/massmodCompare/2016all',
    'images/pdf/massmodCompare/2016APV',
    'images/pdf/massmodCompare/2016',
    'images/pdf/massmodCompare/2017',
    'images/pdf/massmodCompare/2018',
    'images/pdf/massmodCompare/Full'
]


for path in directories:
    if not os.path.exists(path):
        os.makedirs(path)


# load histograms and get scale factors
coffeaFiles = getCoffeaFilenames(False, useOldHTcut, useBlinding)

LoadedFiles = {
    'TTbar': {
        'unweighted':{
            '2016APV': {'700to1000':None, '1000toInf':None},
            '2016': {'700to1000':None, '1000toInf':None},
            '2017': {'700to1000':None, '1000toInf':None},
            '2018': {'700to1000':None, '1000toInf':None}
        },
        'weighted':{
            '2016APV': {'700to1000':None, '1000toInf':None},
            '2016': {'700to1000':None, '1000toInf':None},
            '2017': {'700to1000':None, '1000toInf':None},
            '2018': {'700to1000':None, '1000toInf':None}
        },
        'noMassMod':{
            '2016APV': {'700to1000':None, '1000toInf':None},
            '2016': {'700to1000':None, '1000toInf':None},
            '2017': {'700to1000':None, '1000toInf':None},
            '2018': {'700to1000':None, '1000toInf':None}
        }
    },
    'JetHT': {
        'unweighted':{
            '2016APV': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2016': {'F':None, 'G':None, 'H':None},
            '2017': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2018': {'A':None, 'B':None, 'C':None, 'D':None}
        },
        'weighted':{
            '2016APV': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2016': {'F':None, 'G':None, 'H':None},
            '2017': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2018': {'A':None, 'B':None, 'C':None, 'D':None}
        },
        'noMassMod':{
            '2016APV': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2016': {'F':None, 'G':None, 'H':None},
            '2017': {'B':None, 'C':None, 'D':None, 'E':None, 'F':None},
            '2018': {'A':None, 'B':None, 'C':None, 'D':None}
        }
    },
    'QCD': {
        'unweighted':{
            '2016APV': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2016': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2017': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2018': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None}
        },
        'weighted':{
            '2016APV': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2016': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2017': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None},
            '2018': {'300to470':None, '470to600':None, '600to800':None, '800to1400':None, '1400to1800':None, '1800to2400':None, '2400to3200':None, '3200toInf':None}
        }
    },
    'RSGluon':{
        'unweighted':{
            '2016APV': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2016': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2017': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2018': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None}
        }
    },
    'ZPrime1':{
        'unweighted':{
            '2016APV': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None},
            '2016': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None},
            '2017': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None},
            '2018': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None}
        }
    },
    'ZPrime10':{
        'unweighted':{
            '2016APV': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2016': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2017': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2018': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None}
        }
    },
    'ZPrime30':{
        'unweighted':{
            '2016APV': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2016': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2017': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2018': {'1000':None, '1200':None, '1400':None, '1600':None, '1800':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None}
        }
    },
    'ZPrimeDM':{
        'unweighted':{
            '2016APV': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2016': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2017': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None},
            '2018': {'1000':None, '1500':None, '2000':None, '2500':None, '3000':None, '3500':None, '4000':None, '4500':None, '5000':None}
        }
    }
}


ds = 'TTbar'
for bkgest_str in ['unweighted', 'weighted', 'noMassMod']:
    for year in ['2016APV', '2016', '2017', '2018']:
        for key, file in coffeaFiles[ds][bkgest_str][year].items():
            LoadedFiles[ds][bkgest_str][year][key] = (util.load(file))
            print(file + ' loaded')
            print(ds + ' ' + year + ' ' + key + ' ' + bkgest_str)


ds = 'QCD'
for bkgest_str in ['unweighted', 'weighted']:
    for year in ['2016APV', '2016', '2017', '2018']:
        for key, file in coffeaFiles[ds][bkgest_str][year].items():
            LoadedFiles[ds][bkgest_str][year][key] = (util.load(file))
            print(file + ' loaded')


ds = 'JetHT'
for bkgest_str in ['unweighted', 'weighted', 'noMassMod']:
    for year in ['2016APV', '2016', '2017', '2018']:
        for key, file in coffeaFiles[ds][bkgest_str][year].items():
            LoadedFiles[ds][bkgest_str][year][key] = (util.load(file))
            print(file + ' loaded')


bkgest_str = 'unweighted'
for year in ['2016APV', '2016', '2017', '2018']:
    for ds in ['ZPrime10']:#['RSGluon', 'ZPrime1', 'ZPrime10', 'ZPrime30', 'ZPrimeDM']:
        for key, file in coffeaFiles[ds][bkgest_str][year].items():
            LoadedFiles[ds][bkgest_str][year][key] = (util.load(file))
            print(file + ' loaded') # for masspoint ' + key)


# analysis categories #
label_dict = LoadedFiles['QCD']['unweighted']['2018']['3200toInf']['analysisCategories']
label_to_int_dict = {label: i for i, label in label_dict.items()}
print(label_dict)


def getHist_NoWeight(hname, ds, bkgest, year, sum_axes=[], integrate_axes={}, masspoint=''):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # bkgest = boolean, True if bkg estimate applied                                     #
    # year = '2016APV' or '2016' or '2017' or '2018'                                     #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    ######################################################################################    

    
    # load histograms and get scale factors
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut)
    
    cfiles = []
    sf = []
    bkgest_str = np.where([bkgest], 'weighted', 'unweighted')[0]    
     
    for key, file in coffeaFiles[ds][bkgest_str][year].items():
        if masspoint != '':
            if masspoint in key:
                loaded_file = LoadedFiles[ds][bkgest_str][year][masspoint]
                sum_axes_dict = {ax:sum for ax in sum_axes}
                histo = loaded_file[hname][integrate_axes][sum_axes_dict]
                histo = histo * (lumi[year] * 1.0 / loaded_file['cutflow']['sumw'])
                return histo
       
        loaded_file = LoadedFiles[ds][bkgest_str][year][key]
        cfiles.append(loaded_file)
        
    # sum or integrate axes for all hists from dataset eras or pt bins
    sum_axes_dict = {ax:sum for ax in sum_axes}
    
    hists = []
    
    for cfile in cfiles:
        hists.append(cfile[hname][integrate_axes][sum_axes_dict])   
            
    return hists

def getHist(hname, ds, bkgest, year, sum_axes=[], integrate_axes={}, masspoint=''):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # bkgest = boolean, True if bkg estimate applied                                     #
    # year = '2016APV' or '2016' or '2017' or '2018'                                     #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    ######################################################################################    

    
    # load histograms and get scale factors
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut)
    
    cfiles = []
    sf = []
    bkgest_str = np.where([bkgest], 'weighted', 'unweighted')[0]    
     
    for key, file in coffeaFiles[ds][bkgest_str][year].items():
        if masspoint != '':
            if masspoint in key:
                loaded_file = LoadedFiles[ds][bkgest_str][year][masspoint]
                sum_axes_dict = {ax:sum for ax in sum_axes}
                histo = loaded_file[hname][integrate_axes][sum_axes_dict]
                histo = histo * (lumi[year] * 1.0 / loaded_file['cutflow']['sumw'])
                return histo
        
        loaded_file = LoadedFiles[ds][bkgest_str][year][key]
        cfiles.append(loaded_file)


        if 'TTbar' in ds and '700to1000' in key:
            sf.append(lumi[year] * ttbar_xs1 / (loaded_file['cutflow']['sumw']))
        elif 'TTbar' in ds and '1000toInf' in key:
            sf.append(lumi[year] * ttbar_xs2 / (loaded_file['cutflow']['sumw']))   
        elif 'QCD' in ds:
            sf.append(lumi[year] * qcd_xs[key] / (loaded_file['cutflow']['sumw']))
        else:
            sf.append(1.)
        
    # sum or integrate axes for all hists from dataset eras or pt bins
    sum_axes_dict = {ax:sum for ax in sum_axes}
    
    hists = []
    
    for cfile in cfiles:
        hists.append(cfile[hname][integrate_axes][sum_axes_dict])   
        
    histo = hists[0]*sf[0]
    if len(hists) > 1:
        for i in range(len(hists) - 1): 
            histo = histo + hists[i+1]*sf[i+1]
        
            
    return histo

def getHistNoMassMod(hname, ds, year, sum_axes=[], integrate_axes={}):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # bkgest = boolean, True if bkg estimate applied                                     #
    # year = '2016APV' or '2016' or '2017' or '2018'                                     #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    ######################################################################################    

    
    # load histograms and get scale factors
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut)
    
    cfiles = []
    sf = []
    noMassMod_str = 'noMassMod'
    
    for key, file in coffeaFiles[ds][noMassMod_str][year].items():
            
        loaded_file = LoadedFiles[ds][noMassMod_str][year][key]
        cfiles.append(loaded_file)
        
        
        if 'TTbar' in ds and '700to1000' in key:
            sf.append(lumi[year] * ttbar_xs1 / loaded_file['cutflow']['sumw'])
        elif 'TTbar' in ds and '1000toInf' in key:
            sf.append(lumi[year] * ttbar_xs2 / loaded_file['cutflow']['sumw'])  
        else:
            sf.append(1.)
        
    # sum or integrate axes for all hists from dataset eras or pt bins
    sum_axes_dict = {ax:sum for ax in sum_axes}
    
    hists = []
    for cfile in cfiles:
        hists.append(cfile[hname][integrate_axes][sum_axes_dict])    
    
    # sum all hists from dataset eras or pt bins
    histo = hists[0]*sf[0]
    if len(hists) > 1:
        for i in range(len(hists) - 1): 
            histo = histo + hists[i+1]*sf[i+1]
            
            
    return histo


def plotBackgroundEstimate(HistDict, SystUnc, Text='', SaveFileName='', isInclusive=True, category='', signame='', xaxis='', linear=False, Pull=False, SignalPlot=False):
    
    Hbkg = HistDict['ntmj'] + HistDict['ttbar']
    Hbkg_fixed = HistDict['ntmj_fixed'] + HistDict['ttbar']
    
    Ndenom = HistDict['antitag_data'].values() - HistDict['antitag_ttbar'].values()
    
    mistag_data = np.where(HistDict['pretag'].values()>0., HistDict['ntmj_fixed'].values() / HistDict['pretag'].values(), 0.)
    term1 = np.where(HistDict['pretag'].values()>0., 1. / HistDict['pretag'].values(), 0.)
    term2 = np.where( (Ndenom*mistag_data)>0., (np.ones(len(mistag_data))-mistag_data)/(Ndenom*mistag_data), 0. ) 
    mistagErrProp =  HistDict['ntmj_fixed'].values()*np.sqrt( term1 + term2 )# 1 + 2 = mistag error & stat. error of NTMJ
    
    mmTerm1 = (HistDict['ntmj_fixed_noMM'].values() - HistDict['ntmj_fixed'].values()) # "source" of Mass Mod error
    mmTerm2 = (HistDict['ntmj_fixed_noMM'].values() + HistDict['ntmj_fixed'].values()) / 2.
    mmPercentErr = np.where( mmTerm2>0., mmTerm1/mmTerm2, 0. ) # Calculated percent error of the difference
    mmErrProp = HistDict['ntmj_fixed'].values()*mmPercentErr # Mass Mod error
    
    qcdTerm1 = (HistDict['QCDbkgest'].values() - HistDict['QCDsignal'].values()) # "source" of QCD closure error
    qcdTerm2 = (HistDict['QCDbkgest'].values() + HistDict['QCDsignal'].values()) / 2.
    qcdPercentErr = np.where((qcdTerm2 > 10**(-4)), qcdTerm1/qcdTerm2, 0.) # Calculated percent error of the difference
    qcdErrProp = HistDict['ntmj_fixed'].values()*qcdPercentErr #QCD closure error
    
    errNTMJ = np.sqrt(mistagErrProp**2 + mmErrProp**2 + qcdErrProp**2) # +- dN
    # print('qcdClosure Percent Difference = ', qcdPercentErr)
    # print('Mass Mod Percent Difference = ', mmPercentErr)
    # print('mistagErr = ', mistagErrProp)
    # print('massmodErr = ', mmErrProp)
    # print('qcdclosureErr = ', qcdErrProp)

    errsUpTTbar = SystUnc['plusErrs'] # + dT
    errsDownTTbar = SystUnc['minusErrs'] # - dT
    
    errsUp = np.sqrt( errNTMJ**2 + errsUpTTbar**2 ) # + dB
    errsDown = np.sqrt( errNTMJ**2 + errsDownTTbar**2 ) # - dB
    
    height = errsUp + errsDown
    bottom = Hbkg_fixed.values() - errsDown
        
        
    # ---- filling text file with pre-fit yield information ---- #
    NTMJnum = np.sum(HistDict['ntmj_fixed'].values())
    TTbarnum = np.sum(HistDict['ttbar'].values())
    NTMJerr = np.sqrt(np.sum(np.where(errNTMJ>0,errNTMJ**2,0)))
    TTbarerrUp = np.sqrt(np.sum(np.where(errsUpTTbar>0,errsUpTTbar**2,0)))
    TTbarerrDown = np.sqrt(np.sum(np.where(errsDownTTbar>0,errsDownTTbar**2,0)))
    ObsStatErr = np.sqrt(np.sum(HistDict['data'].values()))
    # if isInclusive and SaveFileName != '':
    #     with open(SaveFileName, 'a') as f:
    #         print('Obs.      = ',  '%10i'% np.sum(HistDict['data'].values()), ' +- ', '%1i'% ObsStatErr, file=f)
    #         print('NTMJ      = ',  '%10i'% NTMJnum, ' +- ', '%1i'% TTbarerrDown, file=f)
    #         print('TTbar     = ',  '%10i'% TTbarnum, ' + ', '%1i'% TTbarerrUp, ' - ', '%1i'% TTbarerrDown, file=f)
    #         print('Tot. Bkg. = ',  '%10i'% (NTMJnum+TTbarnum), ' + ', '%1i'% np.sqrt((NTMJerr**2+TTbarerrUp**2)), ' - ', '%1i'% np.sqrt((NTMJerr**2+TTbarerrDown**2)), file=f)
    # elif not isInclusive and SaveFileName != '':
    #     with open(SaveFileName, 'a') as f:
    #         print(category+'\n===================================================', file=f)
    #         print('Obs.      = ',  '%10i'% np.sum(HistDict['data'].values()), ' +- ', '%1i'% ObsStatErr, file=f)
    #         print('NTMJ      = ',  '%10i'% NTMJnum, ' +- ', '%1i'% TTbarerrDown, file=f)
    #         print('TTbar     = ',  '%10i'% TTbarnum, ' + ', '%1i'% TTbarerrUp, ' - ', '%1i'% TTbarerrDown, file=f)
    #         print('Tot. Bkg. = ',  '%10i'% (NTMJnum+TTbarnum), ' + ', '%1i'% np.sqrt((NTMJerr**2+TTbarerrUp**2)), ' - ', '%1i'% np.sqrt((NTMJerr**2+TTbarerrDown**2)), file=f)
    #         print('\n', file=f)
    
        
    if 'ttbarmass' in xaxis:
        edges = HistDict['ntmj_fixed'].axes['ttbarmass'].edges
    else:
        edges = HistDict['ntmj_fixed'].axes[xaxis].edges
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])
    
    if '2016' in HistDict['IOV']:
        Year = '2016'
    else:
        Year = HistDict['IOV']

    if HistDict['IOV'] == 'Full':
        if linear:
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=0, fontsize=15, ax=ax1)
        else:
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=2, fontsize=20, ax=ax1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=ax1)
    else:
        if linear:
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=0, fontsize=15, ax=ax1)
        else:
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=2, fontsize=20, ax=ax1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=ax1)
            
    
    S = hist.Stack(HistDict['ttbar'], HistDict['ntmj_fixed'])
    S.plot(ax=ax1, stack=True, histtype="fill", color=['xkcd:deep red', 'xkcd:pale gold'], label=['TTbar', 'NTMJ'])
    hep.histplot(HistDict['data'],  ax=ax1, histtype='errorbar', color='black', label='Data')  
    
    
    ax1.bar(x = edges[:-1],
           height=height,
           bottom=bottom,
           width = np.diff(edges), align='edge', hatch='///', edgecolor='gray',
           linewidth=0, facecolor='none', alpha=0.8,
           zorder=10, label='Syst. Unc.')

    
    print('TTbar Unc Up:', errsUpTTbar)
    print('TTbar Unc Down:', errsDownTTbar)
    print('NTMJ Up or Down:', errNTMJ)
    print('Error Up = ', errsUp)
    print('Error Down = ', errsDown)
    print('Height of error bar = ', height)
        
    if SignalPlot:
        Signal = {
            'RSGluon' : r'RS$_{KK}$ Gluon',
            'ZPrime1' : r'Z ` $1\%$',
            'ZPrime10': r'Z ` $10\%$',
            'ZPrime30': r'Z ` $30\%$',
            'ZPrimeDM': r'Z ` DM'
        }
        hep.histplot(HistDict['sig1'], ax=ax1, histtype='step', label=Signal[signame]+' 1 TeV')
        hep.histplot(HistDict['sig3'], ax=ax1, histtype='step', ls='--', lw=3, label=Signal[signame]+' 3 TeV')
        
            
    if Pull != True:
        ratio_plot =  HistDict['data'] / Hbkg_fixed.values()
        AvgUnc = 0.5*(errsUp + errsDown)
        errs = ratio_plot.values() * np.sqrt( 1./(HistDict['data'].values()) + (AvgUnc**2/Hbkg_fixed.values()**2) ) 
        errs_stat = 1./np.sqrt(Hbkg_fixed.values())
        
        ax2.bar(x = edges[:-1],
           height=(2.*(errs+errs_stat)),
           bottom=(np.ones_like(ratio_plot.values()) - errs - errs_stat),
           width = np.diff(edges), align='edge', edgecolor='black',
           linewidth=0, facecolor='red', alpha=0.3,
           zorder=10, label='Tot. Unc.')

        ax2.bar(x = edges[:-1],
               height=(2.*errs_stat),
               bottom=(np.ones_like(ratio_plot.values()) - errs_stat),
               width = np.diff(edges), align='edge', hatch='\\\\', edgecolor='black',
               linewidth=0, facecolor='blue', alpha=0.3,
               zorder=10, label='Stat. Unc.')
        
        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        
        legend2 = plt.legend(bbox_to_anchor =(0.1,-0.85), loc='lower center')
        ax2.set_ylim(0,2)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')
        
    elif Pull == True:
        
        DminusB = (HistDict['data'] + -1*Hbkg_fixed)
        pull_plot = DminusB / np.sqrt( HistDict['data'].values() + np.where(height>0.,height**2,0.) ) 
        pull_plot_arr = np.where(HistDict['data'].values()>0., pull_plot.values(), 0.)
        
        if np.all(np.abs(pull_plot_arr) < 2.): # check to see if pulls are below 2 sigma
            hist.plot.plot_pull_array(pull_plot, pull_plot_arr, ax=ax2, bar_kwargs={'color':'steelblue'}, pp_kwargs={'alpha':0.5})  
            ax2.set_ylim(-2.5,2.5)
            ax2.set_yticks([-2,-1,0,1,2])
        else:
            print('Pull:', pull_plot_arr)
            hist.plot.plot_pull_array(pull_plot, pull_plot_arr, ax=ax2, bar_kwargs={'color':'lightcoral'}, pp_kwargs={'alpha':0.5})  
            ax2.set_ylim(-3.5,3.5)
            ax2.set_yticks([-3,-2,-1,0,1,2,3])
        ax2.axhline(0, color='black', ls='--')
        ax2.set_ylabel(r'(D-B)/$\sigma_{tot.}$')
            

    ax1.legend(fontsize='xx-small', loc=1)
    ax1.set_yscale('symlog')
    ax1.set_ylabel('Events')
    ax1.set_xlabel('')
    ax1.set_ylim(0, 1e5)
    ax1.set_xlim(800, 8000)
    ax2.set_xlim(800, 8000)    
    if linear:
        ax1.set_yscale('linear')
        ax1.autoscale('y')
        ax1.set_xlim(800, 8000)
        ax2.set_xlim(800, 8000) 
        ax1.set_ylim(bottom=0.)
        
    histname = HistDict['varname']
    if histname == 'jetpt':
        ax1.set_xlim(400, 2000)
        ax2.set_xlim(400, 2000)
    elif histname == 'jeteta':
        ax1.set_xlim(-2.4, 2.4)
        ax2.set_xlim(-2.4, 2.4)
    elif histname == 'jetphi':
        ax1.set_xlim(-np.pi, np.pi)
        ax2.set_xlim(-np.pi, np.pi)
    elif histname == 'sdjetmass':
        ax1.set_xlim(0, 500)
        ax2.set_xlim(0, 500)
    elif histname == 'jetp':
        ax1.set_xlim(400, 3600)
        ax2.set_xlim(400, 3600)
        
    if isInclusive:
        leg2 = plt.text(0.75, 0.45, 'b-tag Inclusive\n$|\Delta y|$ Inclusive',#\n$H_T>950$ GeV', #\nCSVv2 b-tagger',
                    fontsize=16,
                    weight='bold',
                    transform=ax1.transAxes
                   )
    

def plotBackgroundEstimateComparison(HistDict, SystUnc, Text='', isInclusive=True, signame='', xaxis='', linear=False, Pull=False, SignalPlot=False):
    
    Hbkg = HistDict['ntmj'] + HistDict['ttbar']
    Hbkg_fixed = HistDict['ntmj_fixed'] + HistDict['ttbar']
    Ndenom = HistDict['antitag_data'].values() - HistDict['antitag_ttbar'].values()
    mistag_data = np.where(HistDict['pretag'].values()>0., HistDict['ntmj_fixed'].values() / HistDict['pretag'].values(), 0.)
    term1 = np.where(HistDict['pretag'].values()>0., 1. / HistDict['pretag'].values(), 0.)
    term2 = np.where( (Ndenom*mistag_data)>0., (np.ones(len(mistag_data))-mistag_data)/(Ndenom*mistag_data), 0. ) 
    mistagErrProp =  HistDict['ntmj_fixed'].values()*np.sqrt( term1 + term2 )# 1 + 2 = mistag error & stat. error of NTMJ
    qcdTerm1 = (HistDict['QCDbkgest'].values() - HistDict['QCDsignal'].values()) # "source" of QCD closure error
    qcdTerm2 = (HistDict['QCDbkgest'].values() + HistDict['QCDsignal'].values()) / 2.
    qcdPercentErr = np.where((qcdTerm2 > 10**(-4)), qcdTerm1/qcdTerm2, 0.) # Calculated percent error of the difference
    qcdErrProp = HistDict['ntmj_fixed'].values()*qcdPercentErr #QCD closure error
    errNTMJ = np.sqrt(mistagErrProp**2 + qcdErrProp**2) # +- dN
    errsUpTTbar = SystUnc['plusErrs'] # + dT
    errsDownTTbar = SystUnc['minusErrs'] # - dT
    errsUp = np.sqrt( errNTMJ**2 + errsUpTTbar**2 ) # + dB
    errsDown = np.sqrt( errNTMJ**2 + errsDownTTbar**2 ) # - dB
    height = errsUp + errsDown
    bottom = Hbkg_fixed.values() - errsDown
    
    Hbkg_noMM = HistDict['ntmj_noMM'] + HistDict['ttbar']
    Hbkg_fixed_noMM = HistDict['ntmj_fixed_noMM'] + HistDict['ttbar']
    height_noMM = errsUp + errsDown
    bottom_noMM = Hbkg_fixed_noMM.values() - errsDown
    
    
    if 'ttbarmass' in xaxis:
        edges = HistDict['ntmj_fixed'].axes['ttbarmass'].edges
    else:
        edges = HistDict['ntmj_fixed'].axes[xaxis].edges
    
    fig, ([ax1, bx1], [ax2, bx2]) = plt.subplots(
        nrows=2, 
        ncols=2, 
        figsize=(20,10),
        height_ratios=[3, 1])
    
    if '2016' in HistDict['IOV']:
        Year = '2016'
    else:
        Year = HistDict['IOV']

    if HistDict['IOV'] == 'Full':
        if linear:
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=0, fontsize=15, ax=ax1)
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=0, fontsize=15, ax=bx1)
        else:
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=2, fontsize=20, ax=ax1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=ax1)
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), loc=2, fontsize=20, ax=bx1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=bx1)
    else:
        if linear:
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=0, fontsize=15, ax=ax1)
            hep.cms.label(Text, data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=0, fontsize=15, ax=bx1)
        else:
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=2, fontsize=20, ax=ax1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=ax1)
            hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[HistDict['IOV']]/1000.), year=Year, loc=2, fontsize=20, ax=bx1)
            hep.cms.text(Text, loc=2, fontsize=20, ax=bx1)
            
    S = hist.Stack(HistDict['ttbar'], HistDict['ntmj_fixed'])
    S_noMM = hist.Stack(HistDict['ttbar'], HistDict['ntmj_fixed_noMM'])
    S.plot(ax=ax1, stack=True, histtype="fill", color=['xkcd:deep red', 'xkcd:pale gold'], label=['TTbar', 'NTMJ'])
    S_noMM.plot(ax=bx1, stack=True, histtype="fill", color=['xkcd:deep red', 'xkcd:pale gold'], label=['TTbar', 'NTMJ'])
    hep.histplot(HistDict['data'],  ax=ax1, histtype='errorbar', color='black', label='Data')   
    hep.histplot(HistDict['data'],  ax=bx1, histtype='errorbar', color='black', label='Data')   
    ax1.bar(x = edges[:-1],
           height=height,
           bottom=bottom,
           width = np.diff(edges), align='edge', hatch='///', edgecolor='gray',
           linewidth=0, facecolor='none', alpha=0.8,
           zorder=10, label='Unc.')
    bx1.bar(x = edges[:-1],
           height=height_noMM,
           bottom=bottom_noMM,
           width = np.diff(edges), align='edge', hatch='///', edgecolor='gray',
           linewidth=0, facecolor='none', alpha=0.8,
           zorder=10, label='Unc.')
    
    if SignalPlot:
        Signal = {
            'RSGluon' : r'RS$_{KK}$ Gluon',
            'ZPrime1' : r'Z ` $1\%$',
            'ZPrime10': r'Z ` $10\%$',
            'ZPrime30': r'Z ` $30\%$',
            'ZPrimeDM': r'Z ` DM'
        }
        hep.histplot(HistDict['sig1'], ax=ax1, histtype='step', label=Signal[signame]+' 1 TeV')
        hep.histplot(HistDict['sig4'], ax=ax1, histtype='step', ls='--', lw=3, label=Signal[signame]+' 4 TeV')
        hep.histplot(HistDict['sig1'], ax=bx1, histtype='step', label=Signal[signame]+' 1 TeV')
        hep.histplot(HistDict['sig4'], ax=bx1, histtype='step', ls='--', lw=3, label=Signal[signame]+' 4 TeV')
        
            
    if Pull != True:
        
        ratio_plot =  HistDict['data'] / Hbkg_fixed.values()
        AvgUnc = 0.5*(errsUp + errsDown)
        errs = ratio_plot.values() * np.sqrt( 1./(HistDict['data'].values()) + (AvgUnc**2/Hbkg_fixed.values()**2) ) 
        errs_stat = 1./np.sqrt(Hbkg_fixed.values())
        
        ratio_plot_noMM =  HistDict['data'] / Hbkg_fixed_noMM.values()
        AvgUnc_noMM = 0.5*(errsUp + errsDown)
        errs_noMM = ratio_plot.values() * np.sqrt( 1./(HistDict['data'].values()) + (AvgUnc_noMM**2/Hbkg_fixed_noMM.values()**2) ) 
        errs_stat_noMM = 1./np.sqrt(Hbkg_fixed_noMM.values())
        
        ax2.bar(x = edges[:-1],
           height=(2.*(errs+errs_stat)),
           bottom=(np.ones_like(ratio_plot.values()) - errs - errs_stat),
           width = np.diff(edges), align='edge', edgecolor='black',
           linewidth=0, facecolor='red', alpha=0.3,
           zorder=10, label='Tot. Unc.')

        ax2.bar(x = edges[:-1],
               height=(2.*errs_stat),
               bottom=(np.ones_like(ratio_plot.values()) - errs_stat),
               width = np.diff(edges), align='edge', hatch='\\\\', edgecolor='black',
               linewidth=0, facecolor='blue', alpha=0.3,
               zorder=10, label='Stat. Unc.')
        
        bx2.bar(x = edges[:-1],
           height=(2.*(errs_noMM+errs_stat_noMM)),
           bottom=(np.ones_like(ratio_plot_noMM.values()) - errs_noMM - errs_stat_noMM),
           width = np.diff(edges), align='edge', edgecolor='black',
           linewidth=0, facecolor='red', alpha=0.3,
           zorder=10, label='Tot. Unc.')

        bx2.bar(x = edges[:-1],
               height=(2.*errs_stat_noMM),
               bottom=(np.ones_like(ratio_plot_noMM.values()) - errs_stat_noMM),
               width = np.diff(edges), align='edge', hatch='\\\\', edgecolor='black',
               linewidth=0, facecolor='blue', alpha=0.3,
               zorder=10, label='Stat. Unc.')
        
        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        hep.histplot(ratio_plot_noMM, ax=bx2, histtype='errorbar', color='black')
        
        legend2 = plt.legend(bbox_to_anchor =(-1.00,-0.85), loc='lower center')
        ax2.set_ylim(0,2)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')
        bx2.set_ylim(0,2)
        bx2.axhline(1, color='black', ls='--')
        bx2.set_ylabel('Data/Bkg')
        
    elif Pull == True:
        
        DminusB = (HistDict['data'] + -1*Hbkg_fixed)
        pull_plot = DminusB / np.sqrt( HistDict['data'].values() + np.where(height>0.,height**2,0.) ) 
        pull_plot_arr = np.where(HistDict['data'].values()>0., pull_plot.values(), 0.)
        
        DminusB_noMM = (HistDict['data'] + -1*Hbkg_fixed_noMM)
        pull_plot_noMM = DminusB_noMM / np.sqrt( HistDict['data'].values() + np.where(height>0.,height**2,0.) ) 
        pull_plot_arr_noMM = np.where(HistDict['data'].values()>0., pull_plot_noMM.values(), 0.)
        
        if np.all(np.abs(pull_plot_arr) < 2.): # check to see if pulls are below 2 sigma
            hist.plot.plot_pull_array(pull_plot, pull_plot_arr, ax=ax2, bar_kwargs={'color':'steelblue'}, pp_kwargs={'alpha':0.5})
            hist.plot.plot_pull_array(pull_plot_noMM, pull_plot_arr_noMM, ax=bx2, bar_kwargs={'color':'steelblue'}, pp_kwargs={'alpha':0.5})
            ax2.set_ylim(-2.5,2.5)
            ax2.set_yticks([-2,-1,0,1,2])
            bx2.set_ylim(-2.5,2.5)
            bx2.set_yticks([-2,-1,0,1,2])
        else:
            print('Pull:', pull_plot_arr)
            hist.plot.plot_pull_array(pull_plot, pull_plot_arr, ax=ax2, bar_kwargs={'color':'lightcoral'}, pp_kwargs={'alpha':0.5})  
            hist.plot.plot_pull_array(pull_plot_noMM, pull_plot_arr_noMM, ax=bx2, bar_kwargs={'color':'lightcoral'}, pp_kwargs={'alpha':0.5})  
            ax2.set_ylim(-3.5,3.5)
            ax2.set_yticks([-3,-2,-1,0,1,2,3])
            bx2.set_ylim(-3.5,3.5)
            bx2.set_yticks([-3,-2,-1,0,1,2,3])
            
        ax2.axhline(0, color='black', ls='--')
        ax2.set_ylabel(r'(D-B)/$\sigma_{tot.}$')
        bx2.axhline(0, color='black', ls='--')
        bx2.set_ylabel(r'(D-B)/$\sigma_{tot.}$')

    ax1.legend(fontsize='xx-small', loc=1)
    ax1.set_yscale('log')
    ax1.set_ylabel('Events')
    ax1.set_xlabel('')
    ax1.set_ylim(1e-2, 1e7)
    ax1.set_xlim(900, 8000)
    ax2.set_xlim(900, 8000)   
    
    bx1.legend(fontsize='xx-small', loc=1)
    bx1.set_yscale('log')
    bx1.set_ylabel('Events')
    bx1.set_xlabel('')
    bx1.set_ylim(1e-2, 1e7)
    bx1.set_xlim(900, 8000)
    bx2.set_xlim(900, 8000)   
    if linear:
        ax1.set_yscale('linear')
        ax1.autoscale('y')
        ax1.set_xlim(900, 8000)
        bx1.set_yscale('linear')
        bx1.autoscale('y')
        bx1.set_xlim(900, 8000)
        
    histname = HistDict['varname']
    if histname == 'jetpt':
        ax1.set_xlim(400, 2000)
        ax2.set_xlim(400, 2000)
        bx1.set_xlim(400, 2000)
        bx2.set_xlim(400, 2000)
    elif histname == 'jeteta':
        ax1.set_xlim(-2.4, 2.4)
        ax2.set_xlim(-2.4, 2.4)
        bx1.set_xlim(-2.4, 2.4)
        bx2.set_xlim(-2.4, 2.4)
    elif histname == 'jetphi':
        ax1.set_xlim(-np.pi, np.pi)
        ax2.set_xlim(-np.pi, np.pi)
        bx1.set_xlim(-np.pi, np.pi)
        bx2.set_xlim(-np.pi, np.pi)
    elif histname == 'sdjetmass':
        ax1.set_xlim(0, 500)
        ax2.set_xlim(0, 500)
        bx1.set_xlim(0, 500)
        bx2.set_xlim(0, 500)
    elif histname == 'jetp':
        ax1.set_xlim(400, 3600)
        ax2.set_xlim(400, 3600)
        bx1.set_xlim(400, 3600)
        bx2.set_xlim(400, 3600)
        
    if isInclusive:
        leg2a = plt.text(0.60, 0.60, 'Mass Mod\nb-tag Inclusive\n$|\Delta y|$ Inclusive',
                    fontsize=16,
                    weight='bold',
                    transform=ax1.transAxes
                   )
        leg2b = plt.text(0.60, 0.60, 'Without Mass Mod\nb-tag Inclusive\n$|\Delta y|$ Inclusive',
                    fontsize=16,
                    weight='bold',
                    transform=bx1.transAxes
                   )
    else:
        leg2a = plt.text(0.60, 0.60, 'Mass Mod',
                    fontsize=16,
                    weight='bold',
                    transform=ax1.transAxes
                   )
        leg2b = plt.text(0.60, 0.60, 'Without Mass Mod',
                    fontsize=16,
                    weight='bold',
                    transform=bx1.transAxes
                   )



def NeededHists(Variable, iov, signal, category=''):
    
    if category == '':
        signal_cats = [ i for label, i in label_to_int_dict.items() if '2t' in label]
        pretag_cats = [ i for label, i in label_to_int_dict.items() if 'pre' in label]
        anti_cats   = [ i for label, i in label_to_int_dict.items() if 'at' in label]
        
        if 'ttbarmass' in Variable:
            httbar = getHist(Variable, 'TTbar', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hcontam = getHist(Variable, 'TTbar', True, iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hQCDsignal = getHist(Variable, 'QCD', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hQCDbkgest = getHist(Variable, 'QCD', True, iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hcontam_noMM = getHistNoMassMod(Variable, 'TTbar', iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hntmj = getHist(Variable, 'JetHT', True, iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hntmj_noMM = getHistNoMassMod(Variable, 'JetHT', iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hdata = getHist(Variable, 'JetHT', False, iov,  sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hpretag = getHist(Variable, 'JetHT', False, iov,sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'}) 
            hantitag_data = getHist(Variable, 'JetHT', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':anti_cats, 'systematic':'nominal'})
            hantitag_ttbar = getHist(Variable, 'TTbar', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':anti_cats, 'systematic':'nominal'}) 
            
            hsignal1000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='1000')
            hsignal2000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='2000')
            hsignal3000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='3000')
            hsignal4000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='4000')

        else:
            httbar = getHist(Variable, 'TTbar', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats})
            hcontam = getHist(Variable, 'TTbar', True, iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hQCDsignal = getHist(Variable, 'QCD', False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats})
            hQCDbkgest = getHist(Variable, 'QCD', True, iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hcontam_noMM = getHistNoMassMod(Variable, 'TTbar', iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hntmj = getHist(Variable, 'JetHT', True, iov,   sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hntmj_noMM = getHistNoMassMod(Variable, 'JetHT', iov, sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hdata = getHist(Variable, 'JetHT', False, iov,  sum_axes=['anacat'], integrate_axes={'anacat':signal_cats})
            hpretag = getHist(Variable, 'JetHT', False, iov,sum_axes=['anacat'], integrate_axes={'anacat':pretag_cats})
            hantitag_data = getHist(Variable, 'JetHT', False, iov,  sum_axes=['anacat'], integrate_axes={'anacat':anti_cats})
            hantitag_ttbar = getHist(Variable, 'TTbar', False, iov,  sum_axes=['anacat'], integrate_axes={'anacat':anti_cats}) 
            
            hsignal1000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats}, masspoint='1000')
            hsignal2000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats}, masspoint='2000')
            hsignal3000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats}, masspoint='3000')
            hsignal4000 = getHist(Variable, signal, False, iov, sum_axes=['anacat'], integrate_axes={'anacat':signal_cats}, masspoint='4000')
    else:
        signal_cats = label_to_int_dict['2t'+category]
        pretag_cats = label_to_int_dict['pret'+category]
        anti_cats   = label_to_int_dict['at'+category]
        
        if 'ttbarmass' in Variable:
            httbar = getHist(Variable, 'TTbar', False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hcontam = getHist(Variable, 'TTbar', True, iov, sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hQCDsignal = getHist(Variable, 'QCD', False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hQCDbkgest = getHist(Variable, 'QCD', True, iov, sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hcontam_noMM = getHistNoMassMod(Variable, 'TTbar', iov, sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hntmj = getHist(Variable, 'JetHT', True, iov,   sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hntmj_noMM = getHistNoMassMod(Variable, 'JetHT', iov, sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hdata = getHist(Variable, 'JetHT', False, iov,  sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'})
            hpretag = getHist(Variable, 'JetHT', False, iov,sum_axes=[], integrate_axes={'anacat':pretag_cats, 'systematic':'nominal'})
            hantitag_data = getHist(Variable, 'JetHT', False, iov,  sum_axes=[], integrate_axes={'anacat':anti_cats, 'systematic':'nominal'})
            hantitag_ttbar = getHist(Variable, 'TTbar', False, iov,  sum_axes=[], integrate_axes={'anacat':anti_cats, 'systematic':'nominal'}) 
            
            hsignal1000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='1000')
            hsignal2000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='2000')
            hsignal3000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='3000')
            hsignal4000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats, 'systematic':'nominal'}, masspoint='4000')

        else:
            httbar = getHist(Variable, 'TTbar', False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats})
            hcontam = getHist(Variable, 'TTbar', True, iov, sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hQCDsignal = getHist(Variable, 'QCD', False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats})
            hQCDbkgest = getHist(Variable, 'QCD', True, iov, sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hcontam_noMM = getHistNoMassMod(Variable, 'TTbar', iov, sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hntmj = getHist(Variable, 'JetHT', True, iov,   sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hntmj_noMM = getHistNoMassMod(Variable, 'JetHT', iov, sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hdata = getHist(Variable, 'JetHT', False, iov,  sum_axes=[], integrate_axes={'anacat':signal_cats})
            hpretag = getHist(Variable, 'JetHT', False, iov,sum_axes=[], integrate_axes={'anacat':pretag_cats})
            hantitag_data = getHist(Variable, 'JetHT', False, iov,  sum_axes=[], integrate_axes={'anacat':anti_cats})
            hantitag_ttbar = getHist(Variable, 'TTbar', False, iov,  sum_axes=[], integrate_axes={'anacat':anti_cats}) 
            
            hsignal1000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats}, masspoint='1000')
            hsignal2000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats}, masspoint='2000')
            hsignal3000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats}, masspoint='3000')
            hsignal4000 = getHist(Variable, signal, False, iov, sum_axes=[], integrate_axes={'anacat':signal_cats}, masspoint='4000')
            
    neededHists = {
        'httbar': httbar,
        'hcontam': hcontam,
        'hQCDsignal': hQCDsignal,
        'hQCDbkgest': hQCDbkgest,
        'hntmj': hntmj,
        'hcontam_noMM': hcontam_noMM,
        'hntmj_noMM': hntmj_noMM,
        'hdata': hdata,
        'hpretag': hpretag,
        'hantitag_data': hantitag_data,
        'hantitag_ttbar': hantitag_ttbar,
        'hsignal1000': hsignal1000,
        'hsignal2000': hsignal2000,
        'hsignal3000': hsignal3000,
        'hsignal4000': hsignal4000
    }
    
    return neededHists


def Use2016allIOV(Variable, signal, category=''):
    

    HistsAPV = NeededHists(Variable, '2016APV', signal, category)
    HistsnoAPV = NeededHists(Variable, '2016', signal, category)
    
    httbar  = HistsAPV['httbar'] + HistsnoAPV['httbar'] 
    hcontam = HistsAPV['hcontam'] + HistsnoAPV['hcontam'] 
    hQCDsignal = HistsAPV['hQCDsignal'] + HistsnoAPV['hQCDsignal']
    hQCDbkgest = HistsAPV['hQCDbkgest'] + HistsnoAPV['hQCDbkgest']
    hntmj   = HistsAPV['hntmj'] + HistsnoAPV['hntmj'] 
    hcontam_noMM = HistsAPV['hcontam_noMM'] + HistsnoAPV['hcontam_noMM'] 
    hntmj_noMM   = HistsAPV['hntmj_noMM'] + HistsnoAPV['hntmj_noMM'] 
    hdata   = HistsAPV['hdata'] + HistsnoAPV['hdata'] 
    hpretag = HistsAPV['hpretag'] + HistsnoAPV['hpretag'] 
    hantitag_data = HistsAPV['hantitag_data'] + HistsnoAPV['hantitag_data'] 
    hantitag_ttbar = HistsAPV['hantitag_ttbar'] + HistsnoAPV['hantitag_ttbar'] 
    hsignal1000 = HistsAPV['hsignal1000'] + HistsnoAPV['hsignal1000']
    hsignal2000 = HistsAPV['hsignal2000'] + HistsnoAPV['hsignal2000']
    hsignal3000 = HistsAPV['hsignal3000'] + HistsnoAPV['hsignal3000']
    hsignal4000 = HistsAPV['hsignal4000'] + HistsnoAPV['hsignal4000']
    
    hntmj_fixed = hntmj + -1*hcontam
    hntmj_fixed_noMM = hntmj_noMM + -1*hcontam_noMM
    
    HistDict = {
        'varname': Variable,
        'pretag': hpretag,
        'contam': hcontam,
        'QCDsignal': hQCDsignal,
        'QCDbkgest': hQCDbkgest,
        'antitag_data': hantitag_data,
        'antitag_ttbar': hantitag_ttbar,
        'data': hdata,
        'ntmj': hntmj,
        'ntmj_fixed': hntmj_fixed,
        'ntmj_noMM': hntmj_noMM,
        'ntmj_fixed_noMM': hntmj_fixed_noMM,
        'ttbar': httbar,
        'sig1': hsignal1000,
        'sig2': hsignal2000,
        'sig3': hsignal3000,
        'sig4': hsignal4000,
        'IOV': '2016all'
    }
    return HistDict

def getSystUnc2016all(HistDict, Variable, signal, Syst, category='', contam=False):
    if category == '':
        if contam:
            Cats = [ i for label, i in label_to_int_dict.items() if 'pre' in label]
            nominalHist = HistDict['contam']
        else:
            Cats = [ i for label, i in label_to_int_dict.items() if '2t' in label]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = (Up_apv + Up_noapv) + -1.*nominalHist
                Down = (Down_apv + Down_noapv) + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2
            
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
    else:
        if contam:
            Cats = label_to_int_dict['pret'+category]
            nominalHist = HistDict['contam']
        else:
            Cats = label_to_int_dict['2t'+category]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = (Up_apv + Up_noapv) + -1.*nominalHist
                Down = (Down_apv + Down_noapv) + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2

                               
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
    return {
        'plusErrs': np.sqrt(sumsqrUps),#np.sqrt( np.sum( sqrOfUps ) ),
        'minusErrs': np.sqrt(sumsqrDowns) #np.sqrt( np.sum( sqrOfDowns ) )
    }
    

def UseIOV(Variable, IOV, signal, category=''):
    

    Hists = NeededHists(Variable, IOV, signal, category)
    
    httbar  = Hists['httbar']
    hcontam = Hists['hcontam']
    hQCDsignal = Hists['hQCDsignal']
    hQCDbkgest = Hists['hQCDbkgest']
    hntmj   = Hists['hntmj']
    hcontam_noMM = Hists['hcontam_noMM']
    hntmj_noMM   = Hists['hntmj_noMM']
    hdata   = Hists['hdata']
    hpretag = Hists['hpretag']
    hantitag_data = Hists['hantitag_data']
    hantitag_ttbar = Hists['hantitag_ttbar']
    
    hsignal1000 = Hists['hsignal1000']
    hsignal2000 = Hists['hsignal2000']
    hsignal3000 = Hists['hsignal3000']
    hsignal4000 = Hists['hsignal4000']
    
    hntmj_fixed = hntmj + -1*hcontam
    hntmj_fixed_noMM = hntmj_noMM + -1*hcontam_noMM
    
    HistDict = {
        'varname': Variable,
        'pretag': hpretag,
        'contam': hcontam,
        'QCDsignal': hQCDsignal,
        'QCDbkgest': hQCDbkgest,
        'antitag_data': hantitag_data,
        'antitag_ttbar': hantitag_ttbar,
        'data': hdata,
        'ntmj': hntmj,
        'ntmj_fixed': hntmj_fixed,
        'ntmj_noMM': hntmj_noMM,
        'ntmj_fixed_noMM': hntmj_fixed_noMM,
        'ttbar': httbar,
        'sig1': hsignal1000,
        'sig2': hsignal2000,
        'sig3': hsignal3000,
        'sig4': hsignal4000,
        'IOV': IOV
    }
    return HistDict

def getSystUncIOV(HistDict, Variable, IOV, signal, Syst, category='', contam=False):
    if category == '':
        if contam:
            Cats = [ i for label, i in label_to_int_dict.items() if 'pre' in label]
            nominalHist = HistDict['contam']
        else:
            Cats = [ i for label, i in label_to_int_dict.items() if '2t' in label]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up = getHist(Variable, 'TTbar', contam, IOV, sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down = getHist(Variable, 'TTbar', contam, IOV, sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = Up + -1.*nominalHist
                Down = Down + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2
            
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
    else:
        if contam:
            Cats = label_to_int_dict['pret'+category]
            nominalHist = HistDict['contam']
        else:
            Cats = label_to_int_dict['2t'+category]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up = getHist(Variable, 'TTbar', contam, IOV, sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down = getHist(Variable, 'TTbar', contam, IOV, sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = Up + -1.*nominalHist
                Down = Down + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2

                               
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
    return {
        'plusErrs': np.sqrt(sumsqrUps),#np.sqrt( np.sum( sqrOfUps ) ),
        'minusErrs': np.sqrt(sumsqrDowns) #np.sqrt( np.sum( sqrOfDowns ) )
    }
    

def UseFullIOV(Variable, signal, category=''):
    

    HistsAPV = NeededHists(Variable, '2016APV', signal, category)
    HistsnoAPV = NeededHists(Variable, '2016', signal, category)
    Hists17 = NeededHists(Variable, '2017', signal, category)
    Hists18 = NeededHists(Variable, '2018', signal, category)
    
    httbar  = HistsAPV['httbar'] + HistsnoAPV['httbar'] + Hists17['httbar'] + Hists18['httbar']
    hcontam = HistsAPV['hcontam'] + HistsnoAPV['hcontam'] + Hists17['hcontam'] + Hists18['hcontam']
    hQCDsignal = HistsAPV['hQCDsignal'] + HistsnoAPV['hQCDsignal'] + Hists17['hQCDsignal'] + Hists18['hQCDsignal']
    hQCDbkgest = HistsAPV['hQCDbkgest'] + HistsnoAPV['hQCDbkgest'] + Hists17['hQCDbkgest'] + Hists18['hQCDbkgest']
    hntmj   = HistsAPV['hntmj'] + HistsnoAPV['hntmj'] + Hists17['hntmj'] + Hists18['hntmj']
    hcontam_noMM = HistsAPV['hcontam_noMM'] + HistsnoAPV['hcontam_noMM'] + Hists17['hcontam_noMM'] + Hists18['hcontam_noMM'] 
    hntmj_noMM   = HistsAPV['hntmj_noMM'] + HistsnoAPV['hntmj_noMM'] + Hists17['hntmj_noMM'] + Hists18['hntmj_noMM'] 
    hdata   = HistsAPV['hdata'] + HistsnoAPV['hdata'] + Hists17['hdata'] + Hists18['hdata'] 
    hpretag = HistsAPV['hpretag'] + HistsnoAPV['hpretag'] + Hists17['hpretag'] + Hists18['hpretag'] 
    hantitag_data = HistsAPV['hantitag_data'] + HistsnoAPV['hantitag_data'] + Hists17['hantitag_data'] + Hists18['hantitag_data']
    hantitag_ttbar = HistsAPV['hantitag_ttbar'] + HistsnoAPV['hantitag_ttbar'] + Hists17['hantitag_ttbar'] + Hists18['hantitag_ttbar'] 
    hsignal1000 = HistsAPV['hsignal1000'] + HistsnoAPV['hsignal1000'] + Hists17['hsignal1000'] + Hists18['hsignal1000']
    hsignal2000 = HistsAPV['hsignal2000'] + HistsnoAPV['hsignal2000'] + Hists17['hsignal1000'] + Hists18['hsignal1000']
    hsignal3000 = HistsAPV['hsignal3000'] + HistsnoAPV['hsignal3000'] + Hists17['hsignal1000'] + Hists18['hsignal1000']
    hsignal4000 = HistsAPV['hsignal4000'] + HistsnoAPV['hsignal4000'] + Hists17['hsignal1000'] + Hists18['hsignal1000']
    
    hntmj_fixed = hntmj + -1*hcontam
    hntmj_fixed_noMM = hntmj_noMM + -1*hcontam_noMM
    
    HistDict = {
        'varname': Variable,
        'pretag': hpretag,
        'contam': hcontam,
        'QCDsignal': hQCDsignal,
        'QCDbkgest': hQCDbkgest,
        'antitag_data': hantitag_data,
        'antitag_ttbar': hantitag_ttbar,
        'data': hdata,
        'ntmj': hntmj,
        'ntmj_fixed': hntmj_fixed,
        'ntmj_noMM': hntmj_noMM,
        'ntmj_fixed_noMM': hntmj_fixed_noMM,
        'ttbar': httbar,
        'sig1': hsignal1000,
        'sig2': hsignal2000,
        'sig3': hsignal3000,
        'sig4': hsignal4000,
        'IOV': 'Full'
    }
    return HistDict

def getSystUncFull(HistDict, Variable, signal, Syst, category='', contam=False):
    if category == '':
        if contam:
            Cats = [ i for label, i in label_to_int_dict.items() if 'pre' in label]
            nominalHist = HistDict['contam']
        else:
            Cats = [ i for label, i in label_to_int_dict.items() if '2t' in label]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_17 = getHist(Variable, 'TTbar', contam, '2017', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_17 = getHist(Variable, 'TTbar', contam, '2017', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
            for corr in Syst[:-1]:
                Up_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = (Up_apv + Up_noapv + Up_17 + Up_18) + -1.*nominalHist
                Down = (Down_apv + Down_noapv + Down_17 + Down_18) + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2
            
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
    else:
        if contam:
            Cats = label_to_int_dict['pret'+category]
            nominalHist = HistDict['contam']
        else:
            Cats = label_to_int_dict['2t'+category]
            nominalHist = HistDict['ttbar']
        if 'ttbarmass' in Variable:
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
            for corr in Syst:
                Up_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_apv = getHist(Variable, 'TTbar', contam, '2016APV', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_noapv = getHist(Variable, 'TTbar', contam, '2016', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                Up_17 = getHist(Variable, 'TTbar', contam, '2017', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_17 = getHist(Variable, 'TTbar', contam, '2017', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
            for corr in Syst[:-1]:
                Up_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                Down_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                
                Up = (Up_apv + Up_noapv + Up_17 + Up_18) + -1.*nominalHist
                Down = (Down_apv + Down_noapv + Down_17 + Down_18) + -1.*nominalHist

                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2

                               
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
    return {
        'plusErrs': np.sqrt(sumsqrUps),#np.sqrt( np.sum( sqrOfUps ) ),
        'minusErrs': np.sqrt(sumsqrDowns) #np.sqrt( np.sum( sqrOfDowns ) )
    }
    


# ## plot background estimate (inclusive)


IOVs = ['Full']
variable = 'ttbarmass'  # Change this at will
signal = 'ZPrime10'
useSignal = True
usePull = True
Linear = False
linearStr = ''
sigstr = ''
pullstr = ''
SaveFileName = f'../data/BkgEstEventCounts_Inc{oldHTstr}.txt'

# -- prepare to overwrite event-count txt -- #
# file_to_delete = open(SaveFileName,'w')
# file_to_delete.close()

if Linear:
    linearStr = '_LINEAR'
histDict = {}

if not usePull:
    pullstr = '_noPull'

dirname = 'closureTest'
if 'ttbarmass' not in variable:
    dirname = 'kinematics'
    usePull = False
if useSignal:
    sigstr = '_with_' + signal
elif ('ttbarmass' in variable) and usePull:
    dirname = 'ttbarmass'
    
    
for IOV in IOVs:

    if 'Full' in IOV:
        histDict = UseFullIOV(variable, signal)
        SystemUnc = getSystUncFull(histDict, variable, signal, systematics, '', False)
    elif '2016all' in IOV:
        histDict = Use2016allIOV(variable, signal)
        SystemUnc = getSystUnc2016all(histDict, variable, signal, systematics, '', False)
    else:
        histDict = UseIOV(variable, IOV, signal)
        SystemUnc = getSystUncIOV(histDict, variable, IOV, signal, systematics, '', False)
        
    if Linear:
        text = f'Work in Progress'
    else:
        text = f'Work in Progress\n'

    plotBackgroundEstimate(histDict, SystemUnc, text, SaveFileName, True, '', signal, variable, Linear, usePull, useSignal)
        
    if usePull:
        savefilename = f'images/png/kinematics/{IOV}/{variable}{sigstr}_Inclusive{pullstr}{linearStr}.png'
    else:
        savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_Inclusive{pullstr}{linearStr}.png'

    print(savefilename)
    plt.savefig(savefilename)
    plt.savefig(savefilename.replace('png', 'pdf'))

    plt.show()


# ## plot background estimate (by category)

variable = 'ttbarmass'  # Change this at will
signal = 'ZPrime10'
useSignal = True
usePull = True
Linear = False
linearStr = ''
sigstr = ''

SaveFileName = f'../data/BkgEstEventCounts_Categories{oldHTstr}.txt'

# -- prepare to overwrite event-count txt -- #
# file_to_delete = open(SaveFileName,'w')
# file_to_delete.close()

if Linear:
    linearStr = '_LINEAR'
histDict = {}

if not usePull:
    pullstr = '_noPull'

dirname = 'closureTest'
if 'ttbarmass' not in variable:
    dirname = 'kinematics'
    usePull = False
if useSignal:
    sigstr = '_with_' + signal
elif ('ttbarmass' in variable) and usePull:
    dirname = 'ttbarmass'
    
for IOV in IOVs:
    
    for cat in ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']:

        if 'Full' in IOV:
            histDict = UseFullIOV(variable, signal, cat)
            SystemUnc = getSystUncFull(histDict, variable, signal, systematics, cat, False)
        elif '2016all' in IOV:
            histDict = Use2016allIOV(variable, signal, cat)
            SystemUnc = getSystUnc2016all(histDict, variable, signal, systematics, cat, False)
        else:
            histDict = UseIOV(variable, IOV, signal, cat)
            SystemUnc = getSystUncIOV(histDict, variable, IOV, signal, systematics, cat, False)
            
        dytext = ''
        if 'cen' in cat:
            dytext = r'$\Delta y$ < 1.0'
        elif 'fwd' in cat:
            dytext = r'$\Delta y$ > 1.0'

        btext = ''
        if '0b' in cat:
            btext = '0 b-tags'
        elif '1b' in cat:
            btext = '1 b-tag'
        elif '2b' in cat:
            btext = '2 b-tags'

        if Linear:
            text = f'Work in Progress:    {btext}, {dytext}'
        else:
            text = f'Work in Progress\n{btext}, {dytext} \n'
            

        plotBackgroundEstimate(histDict, SystemUnc, text, SaveFileName, False, cat, signal, variable, Linear, usePull, useSignal)

        if usePull:
            savefilename = f'images/png/kinematics/{IOV}/{variable}{sigstr}_{cat}{pullstr}{linearStr}.png'
        else:
            savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_{cat}{pullstr}{linearStr}.png'

        print(savefilename)
        plt.savefig(savefilename)
        plt.savefig(savefilename.replace('png', 'pdf'))

        plt.show()


# # Comparison without Mass Modification (Inclusive)


variable = 'ttbarmass'  # Change this at will
signal = 'ZPrime10'
useSignal = False
usePull = True
Linear = True
linearStr = ''
sigstr = ''
if Linear:
    linearStr = '_LINEAR'
histDict = {}

dirname = 'massmodCompare'
if variable != 'ttbarmass':
    dirname = 'massmodCompare'
if useSignal:
    sigstr = '_with_' + signal
    
for IOV in IOVs:

    if 'Full' in IOV:
        histDict = UseFullIOV(variable, signal)
        SystemUnc = getSystUncFull(histDict, variable, signal, systematics, '', False)
    elif '2016all' in IOV:
        histDict = Use2016allIOV(variable, signal)
        SystemUnc = getSystUnc2016all(histDict, variable, signal, systematics, '', False)
    else:
        histDict = UseIOV(variable, IOV, signal)
        SystemUnc = getSystUncIOV(histDict, variable, IOV, signal, systematics, '', False)

    if Linear:
        text = f'Work in Progress'
    else:
        text = f'Work in Progress\n'

    plotBackgroundEstimateComparison(histDict, SystemUnc, text, True, signal, variable, Linear, usePull, useSignal)
        
    if usePull:
        savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_Inclusive{linearStr}_pull.png'
    else:
        savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_Inclusive{linearStr}_pull.png'
        

    print(savefilename)
    # plt.savefig(savefilename)
    # plt.savefig(savefilename.replace('png', 'pdf'))

    plt.show()


# # Comparison without Mass Modification (Categories)


variable = 'ttbarmass'  # Change this at will
signal = 'ZPrime10'
useSignal = False
usePull = True
Linear = True
linearStr = ''
sigstr = ''
if Linear:
    linearStr = '_LINEAR'
histDict = {}

dirname = 'massmodCompare'
if variable != 'ttbarmass':
    dirname = 'massmodCompare'
if useSignal:
    sigstr = '_with_' + signal
    
for IOV in IOVs:
    
    for cat in ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']:

        if 'Full' in IOV:
            histDict = UseFullIOV(variable, signal, cat)
            SystemUnc = getSystUncFull(histDict, variable, signal, systematics, cat, False)
        elif '2016all' in IOV:
            histDict = Use2016allIOV(variable, signal, cat)
            SystemUnc = getSystUnc2016all(histDict, variable, signal, systematics, cat, False)
        else:
            histDict = UseIOV(variable, IOV, signal, cat)
            SystemUnc = getSystUncIOV(histDict, variable, IOV, signal, systematics, cat, False)
            
        dytext = ''
        if 'cen' in cat:
            dytext = r'$\Delta y$ < 1.0'
        elif 'fwd' in cat:
            dytext = r'$\Delta y$ > 1.0'

        btext = ''
        if '0b' in cat:
            btext = '0 b-tags'
        elif '1b' in cat:
            btext = '1 b-tag'
        elif '2b' in cat:
            btext = '2 b-tags'

        if Linear:
            text = f'Work in Progress:    {btext}, {dytext}'
        else:
            text = f'Work in Progress\n{btext}, {dytext} \n'

        plotBackgroundEstimateComparison(histDict, SystemUnc, text, False, signal, variable, Linear, usePull, useSignal)

        if usePull:
            savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_Inclusive{linearStr}.png'
        else:
            savefilename = f'images/png/{dirname}/{IOV}/{variable}{sigstr}_Inclusive{linearStr}.png'

        print(savefilename)
        # plt.savefig(savefilename)
        # plt.savefig(savefilename.replace('png', 'pdf'))

        plt.show()
