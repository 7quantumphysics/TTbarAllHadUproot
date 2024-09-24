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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
hep.style.use("CMS")

sys.path.append('../python/')
from functions import loadCoffeaFile, getLabelMap, getCoffeaFilenames, plotBackgroundEstimate, getHist

SignalToRun = 'ZPrimeDM'
useOldHTcut = False
useBlinding = True

lumi = {
    "2016APV": 19800.,
    "2016": 16120., #35920 - 19800
    "2016all": 35920,
    "2017": 41530./10.,
    "2018": 59800./10., 
    "Full": 35920. + (41530./10.) + (59800./10.) #137190.
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
    oldHTstr = '_oldHTcut'

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
    for ds in [SignalToRun]:#, 'ZPrime1', 'ZPrime10', 'ZPrime30', 'ZPrimeDM']:
        for key, file in coffeaFiles[ds][bkgest_str][year].items():
            LoadedFiles[ds][bkgest_str][year][key] = (util.load(file))
            print(file + ' loaded') # for masspoint ' + key)


# analysis categories #
label_dict = LoadedFiles['QCD']['unweighted']['2016']['3200toInf']['analysisCategories']
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
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut, useBlinding)
    
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
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut, useBlinding)
    
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
    
#     # sum all hists from dataset eras or pt bins
    
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
    coffeaFiles = getCoffeaFilenames(False, useOldHTcut, useBlinding)
    
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
        'plusErrs': np.sqrt(sumsqrUps),
        'minusErrs': np.sqrt(sumsqrDowns)
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
            # sqrOfUps, sqrOfDowns = [], []
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
        'plusErrs': np.sqrt(sumsqrUps),
        'minusErrs': np.sqrt(sumsqrDowns)
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
    hsignal2000 = HistsAPV['hsignal2000'] + HistsnoAPV['hsignal2000'] + Hists17['hsignal2000'] + Hists18['hsignal2000']
    hsignal3000 = HistsAPV['hsignal3000'] + HistsnoAPV['hsignal3000'] + Hists17['hsignal3000'] + Hists18['hsignal3000']
    hsignal4000 = HistsAPV['hsignal4000'] + HistsnoAPV['hsignal4000'] + Hists17['hsignal4000'] + Hists18['hsignal4000']
    
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
                if 'prefiring' not in corr:
                    Up_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                    Down_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=['anacat'], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                    Up = (Up_apv + Up_noapv + Up_17 + Up_18) + -1.*nominalHist
                    Down = (Down_apv + Down_noapv + Down_17 + Down_18) + -1.*nominalHist
                else:
                    Up = (Up_apv + Up_noapv + Up_17) + -1.*nominalHist
                    Down = (Down_apv + Down_noapv + Down_17) + -1.*nominalHist

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
                if 'prefiring' not in corr:
                    Up_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Up'})
                    Down_18 = getHist(Variable, 'TTbar', contam, '2018', sum_axes=[], integrate_axes={'anacat':Cats, 'systematic':f'{corr}Down'})
                    Up = (Up_apv + Up_noapv + Up_17 + Up_18) + -1.*nominalHist
                    Down = (Down_apv + Down_noapv + Down_17 + Down_18) + -1.*nominalHist
                else:
                    Up = (Up_apv + Up_noapv + Up_17) + -1.*nominalHist
                    Down = (Down_apv + Down_noapv + Down_17) + -1.*nominalHist
                    
                sumsqrUps += Up.values()**2
                sumsqrDowns += Down.values()**2

                               
        else:
            print('\n\nWrong Variable.  Will make more later...\n\n')
            sumsqrUps = np.zeros(len(nominalHist.values()))
            sumsqrDowns = np.zeros(len(nominalHist.values()))
            
    return {
        'plusErrs': np.sqrt(sumsqrUps),
        'minusErrs': np.sqrt(sumsqrDowns)
    }


Signals = {
    'ZPrime1' : ['1000', '1200', '1400', '1600', '1800', '2000', '2500', '3000', '3500', '4000', '4500'],
    'ZPrime10': ['1000', '1200', '1400', '1600', '1800', '2000', '2500', '3000', '3500', '4000', '4500', '5000'],
    'ZPrime30': ['1000', '1200', '1400', '1600', '1800', '2000', '2500', '3000', '3500', '4000', '4500', '5000'],
    'ZPrimeDM': ['1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000'],
    'RSGluon':  ['1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']
}

IOVs = ['2016APV', '2016', '2017', '2018']

cats = ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']
cat_labels = ['cen0b', 'fwd0b', 'cen1b', 'fwd1b', 'cen2b', 'fwd2b']


syst_labels = ['nominal']
for s in systematics:
    if not 'nominal' in s:
        syst_labels.append(s+'Down')
        syst_labels.append(s+'Up')
        

# savefileheader = '../outputs/combine2/TTbarAllHad{}_'.format(IOV.replace('20', '').replace('all',''))
savefileheader = '../outputs/combine2/TTbarAllHadFull_'

#### -------- Switch this after the file is created -------- ####
froot = uproot.recreate(savefileheader+'CombineRoot_Cat_'+SignalToRun+'_Blinded.root')
# froot = uproot.reading.open(savefileheader+'CombineRoot_Cat_'+SignalToRun+'.root')

variable = 'ttbarmass'

for IOV in IOVs:
    signal = SignalToRun
    for cat, catname in zip(cats, cat_labels):

        signal_cat = label_to_int_dict['2t'+cat]
        pretag_cat = label_to_int_dict['pret'+cat]

        hsignal = {}
        hsignal['ZPrime1'] = {}
        hsignal['ZPrime10'] = {}
        hsignal['ZPrime30'] = {}
        hsignal['ZPrimeDM'] = {}
        hsignal['RSGluon'] = {}

        if '2016all' in IOV:
            HistDict = Use2016allIOV(variable, signal, cat)
        elif 'Full' in IOV:
            HistDict = UseFullIOV(variable, signal, cat)
        else:
            HistDict = UseIOV(variable, IOV, signal, cat)

        Hbkg = HistDict['ntmj'] + HistDict['ttbar']
        Hbkg_fixed = HistDict['ntmj_fixed'] + HistDict['ttbar']
        Ndenom = HistDict['antitag_data'].values() - HistDict['antitag_ttbar'].values()
        mistag_data = np.where(HistDict['pretag'].values()>0., HistDict['ntmj_fixed'].values() / HistDict['pretag'].values(), 0.)
        term1 = np.where(HistDict['pretag'].values()>0., 1. / HistDict['pretag'].values(), 0.)
        term2 = np.where( ((Ndenom)>0.), mistag_data*(np.ones(len(mistag_data))-mistag_data)/(Ndenom), 0. ) 
        mistagErrProp =  HistDict['ntmj_fixed']*np.sqrt( term1 + term2 )# 1 + 2 = stat. error of NTMJ & mistag error

        mmTerm1 = (HistDict['ntmj_fixed_noMM'].values() - HistDict['ntmj_fixed'].values()) # "source" of Mass Mod error
        mmTerm2 = (HistDict['ntmj_fixed_noMM'].values() + HistDict['ntmj_fixed'].values()) / 2.
        mmPercentErr = np.where( (mmTerm2>0.), mmTerm1/mmTerm2, 0. ) # Calculated percent error of the difference
        mmErrProp = HistDict['ntmj_fixed']*np.abs(mmPercentErr) # Mass Mod error

        qcdTerm1 = (HistDict['QCDbkgest'].values() - HistDict['QCDsignal'].values()) # "source" of QCD closure error
        qcdTerm2 = (HistDict['QCDbkgest'].values() + HistDict['QCDsignal'].values()) / 2.
        qcdPercentErr = np.where((qcdTerm2 > 0), qcdTerm1/qcdTerm2, 0.) # Calculated percent error of the difference
        qcdErrProp = HistDict['ntmj_fixed']*np.abs(qcdPercentErr) #QCD closure error
        
        froot["bkgest_"+catname+IOV+'_mistagUp'] = HistDict['ntmj_fixed'] + mistagErrProp
        froot["bkgest_"+catname+IOV+'_mistagDown'] = HistDict['ntmj_fixed'] + -1.0*mistagErrProp

        froot["bkgest_"+catname+IOV+'_massmodUp'] = HistDict['ntmj_fixed'] + mmErrProp
        froot["bkgest_"+catname+IOV+'_massmodDown'] = HistDict['ntmj_fixed'] + -1.0*mmErrProp

        froot["bkgest_"+catname+IOV+'_qcdclosureUp'] = HistDict['ntmj_fixed'] + qcdErrProp
        froot["bkgest_"+catname+IOV+'_qcdclosureDown'] = HistDict['ntmj_fixed'] + -1.0*qcdErrProp

        for syst in syst_labels:

            if '2016all' in IOV:

                catsystString = catname+'_'+syst

                httbar_apv        = getHist(variable, 'TTbar', False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                hcontam_apv       = getHist(variable, 'TTbar', True, '2016APV', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                httbar_noapv      = getHist(variable, 'TTbar', False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                hcontam_noapv     = getHist(variable, 'TTbar', True, '2016', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})

                print('loading signals...')
                for sig in Signals.keys():
                    if sig == SignalToRun:
                        for mass in Signals[sig]:
                            hsignal_apv   = getHist(variable, sig, False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                            hsignal_noapv = getHist(variable, sig, False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                            hsignal[sig][mass] = hsignal_apv + hsignal_noapv

                httbar  = httbar_apv + httbar_noapv
                hcontam = hcontam_apv + hcontam_noapv

                print('filling root files...')
                if 'nominal' in syst:
                    syst = ''
                    catsystString = catname+syst
                    hntmj_apv   = getHist(variable, 'JetHT', True, '2016APV',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_apv   = getHist(variable, 'JetHT', False, '2016APV',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})
                    hntmj_noapv = getHist(variable, 'JetHT', True, '2016',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_noapv = getHist(variable, 'JetHT', False, '2016',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})

                    hntmj = hntmj_apv + hntmj_noapv
                    hdata = hdata_apv + hdata_noapv
                    hntmj_fixed = hntmj + -1*hcontam
                    print('data_obs_'+catsystString+'\nbkgest_'+catsystString)
                    froot["data_obs_"+catsystString] = hdata
                    froot["bkgest_"+catsystString] = hntmj_fixed

                print('TTbar_'+catsystString)
                froot["TTbar_"+catsystString] = httbar


                for sig in Signals.keys():
                    if sig == SignalToRun:
                        if 'RSGluon' not in sig:
                            if sig != 'ZPrime1': # ZPrime10, 30 and DM
                                [print(sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString] = hsignal[sig][mass]
                            else: # ZPrime1
                                [print(sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString] = hsignal[sig][mass]
                        else: # RSGluon
                            [print(sig+mass+'_'+catsystString) for mass in Signals[sig]]
                            for mass in Signals[sig]:
                                froot[sig+mass+'_'+catsystString] = hsignal[sig][mass]

            elif 'Full' in IOV:

                catsystString = catname+'_'+syst

                if 'prefiring' in syst:
                    httbar_apv        = getHist(variable, 'TTbar', False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_apv       = getHist(variable, 'TTbar', True, '2016APV', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                    httbar_noapv      = getHist(variable, 'TTbar', False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_noapv     = getHist(variable, 'TTbar', True, '2016', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                    httbar_17      = getHist(variable, 'TTbar', False, '2017', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_17     = getHist(variable, 'TTbar', True, '2017', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})

                    httbar  = httbar_apv + httbar_noapv + httbar_17
                    hcontam = hcontam_apv + hcontam_noapv + hcontam_17
                else: #if 'prefiring' not in syst:
                    #print(syst)
                    httbar_apv        = getHist(variable, 'TTbar', False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_apv       = getHist(variable, 'TTbar', True, '2016APV', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                    httbar_noapv      = getHist(variable, 'TTbar', False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_noapv     = getHist(variable, 'TTbar', True, '2016', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                    httbar_17      = getHist(variable, 'TTbar', False, '2017', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_17     = getHist(variable, 'TTbar', True, '2017', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})
                    httbar_18      = getHist(variable, 'TTbar', False, '2018', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam_18     = getHist(variable, 'TTbar', True, '2018', sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})

                    httbar  = httbar_apv + httbar_noapv + httbar_17 + httbar_18
                    hcontam = hcontam_apv + hcontam_noapv + hcontam_17 + hcontam_18

                print('loading signals...')
                for sig in Signals.keys():
                    if sig == SignalToRun:
                        for mass in Signals[sig]:
                            if 'prefiring' in syst:
                                hsignal_apv   = getHist(variable, sig, False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal_noapv = getHist(variable, sig, False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal_17   = getHist(variable, sig, False, '2017', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal[sig][mass] = hsignal_apv + hsignal_noapv + hsignal_17
                            else: #if 'prefiring' not in syst:
                                hsignal_apv   = getHist(variable, sig, False, '2016APV', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal_noapv = getHist(variable, sig, False, '2016', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal_17   = getHist(variable, sig, False, '2017', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal_18 = getHist(variable, sig, False, '2018', sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)
                                hsignal[sig][mass] = hsignal_apv + hsignal_noapv + hsignal_17 + hsignal_18

                # httbar  = httbar_apv + httbar_noapv + httbar_17 + httbar_18
                # hcontam = hcontam_apv + hcontam_noapv + hcontam_17 + hcontam_18

                print('filling root files...')
                if 'nominal' in syst:
                    syst = ''
                    catsystString = catname+syst
                    hntmj_apv   = getHist(variable, 'JetHT', True, '2016APV',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_apv   = getHist(variable, 'JetHT', False, '2016APV',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})
                    hntmj_noapv = getHist(variable, 'JetHT', True, '2016',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_noapv = getHist(variable, 'JetHT', False, '2016',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})
                    hntmj_17 = getHist(variable, 'JetHT', True, '2017',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_17 = getHist(variable, 'JetHT', False, '2017',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})
                    hntmj_18 = getHist(variable, 'JetHT', True, '2018',   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata_18 = getHist(variable, 'JetHT', False, '2018',  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})

                    hntmj = hntmj_apv + hntmj_noapv + hntmj_17 + hntmj_18
                    hdata = hdata_apv + hdata_noapv + hdata_17 + hdata_18
                    hntmj_fixed = hntmj + -1*hcontam
                    print('data_obs_'+catsystString+'\nbkgest_'+catsystString)
                    froot["data_obs_"+catsystString] = hdata
                    froot["bkgest_"+catsystString] = hntmj_fixed

                print('TTbar_'+catsystString)
                froot["TTbar_"+catsystString] = httbar


                for sig in Signals.keys():
                    if sig == SignalToRun:
                        if 'RSGluon' not in sig:
                            if sig != 'ZPrime1': # ZPrime10, 30 and DM
                                [print(sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString] = hsignal[sig][mass]
                            else: # ZPrime1
                                [print(sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString] = hsignal[sig][mass]
                        else: # RSGluon
                            [print(sig+mass+'_'+catsystString) for mass in Signals[sig]]
                            for mass in Signals[sig]:
                                froot[sig+mass+'_'+catsystString] = hsignal[sig][mass]   
            else:
                if ('prefiring' in syst) and (IOV == '2018'):
                    print('\nMaking '+IOV+' in root file; No shape variation for Up/Down prefiring systematic\n')
                    catsystString = catname+IOV+'_'+syst
                    
                    httbar      = getHist(variable, 'TTbar', False, IOV, sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})
                    hcontam     = getHist(variable, 'TTbar', True, IOV, sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})

                    print('loading signals...')
                    for sig in Signals.keys():
                        if sig == SignalToRun:
                            for mass in Signals[sig]:
                                hsignal[sig][mass] = getHist(variable, sig, False, IOV, sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'}, masspoint=mass)
                    # continue
                else:
                    print('\nMaking '+IOV+' in root file\n')
                    catsystString = catname+IOV+'_'+syst
                    
                    httbar      = getHist(variable, 'TTbar', False, IOV, sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst})
                    hcontam     = getHist(variable, 'TTbar', True, IOV, sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':syst})

                    print('loading signals...')
                    for sig in Signals.keys():
                        if sig == SignalToRun:
                            for mass in Signals[sig]:
                                hsignal[sig][mass] = getHist(variable, sig, False, IOV, sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':syst}, masspoint=mass)

                print('filling root files...')
                if 'nominal' in syst:
                    syst = ''
                    catsystString = catname+IOV+syst
                    hntmj = getHist(variable, 'JetHT', True, IOV,   sum_axes=[], integrate_axes={'anacat':pretag_cat, 'systematic':'nominal'})
                    hdata = getHist(variable, 'JetHT', False, IOV,  sum_axes=[], integrate_axes={'anacat':signal_cat, 'systematic':'nominal'})

                    hntmj_fixed = hntmj + -1*hcontam
                    print('data_obs_'+catsystString+'\nbkgest_'+catsystString)
                    froot["data_obs_"+catsystString] = hdata
                    froot["bkgest_"+catsystString] = hntmj_fixed

                print('TTbar_'+catsystString)
                # if 'prefiring' in syst and IOV == '2018':
                #     continue
                # else:
                froot["TTbar_"+catsystString] = httbar


                for sig in Signals.keys():
                    if sig == SignalToRun:
                        if 'RSGluon' not in sig:
                            if sig != 'ZPrime1': # ZPrime10, 30 and DM
                                # if 'prefiring' in syst and IOV == '2018':
                                #     continue
                                # else:
                                [print(sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-2]+mass+'_'+sig[-2:]+'_'+catsystString] = hsignal[sig][mass]
                            else: # ZPrime1
                                # if 'prefiring' in syst and IOV == '2018':
                                #     continue
                                # else:
                                [print(sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString) for mass in Signals[sig]]
                                for mass in Signals[sig]:
                                    froot[sig[:-1]+mass+'_'+sig[-1:]+'_'+catsystString] = hsignal[sig][mass]
                        else: # RSGluon
                            # if 'prefiring' in syst and IOV == '2018':
                            #     continue
                            # else:
                            [print(sig+mass+'_'+catsystString) for mass in Signals[sig]]
                            for mass in Signals[sig]:
                                froot[sig+mass+'_'+catsystString] = hsignal[sig][mass]

froot.close()
