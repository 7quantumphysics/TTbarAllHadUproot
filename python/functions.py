import numpy as np
from coffea import util
import glob
import os
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")


lumi = {
    "2016APV": 19800.,
    "2016": 16120., #35920 - 19800
    "2016all": 35920,
    "2017": 41530.,
    "2018": 59740.
    }

t_BR = 0.6741
ttbar_BR = 0.4544 #PDG 2019
ttbar_xs1 = 831.76 * (0.09210) #pb For ttbar mass from 700 to 1000
ttbar_xs2 = 831.76 * (0.02474) #pb For ttbar mass from 1000 to Inf
toptag_sf = 1.0
toptag_kf = 1.0 #0.7
qcd_xs = 13700000.0 #pb From https://cms-gen-dev.cern.ch/xsdb

ttbar_xs = {'700to1000': 831.76 * (0.09210), #pb For ttbar mass from 700 to 1000
            '1000toInf': 831.76 * (0.02474) #pb For ttbar mass from 1000 to Inf
           }

zprime_xs = {
    '1000': 2.222,
    '1500': 0.387,
    '2000': 0.09428,
    '2500': 0.0279,
    '3000': 0.009327,
    '3500': 0.003507,
    '4000': 0.001484,
    '4500': 0.0007087,
    '5000': 0.0003801,
}

rsgluon_xs = {
    '1000': 21.03,
    '1500': 3.656,
    '2000': 0.9417,
    '2500': 0.3039,
    '3000': 0.1163,
    '3500': 0.05138,
    '4000': 0.02556,
    '4500': 0.01422,
    '5000': 0.008631,
}

xs = {
    'QCD': qcd_xs,
    'TTbar': ttbar_xs,
    'RSGluon': rsgluon_xs,
    'ZPrime': zprime_xs,
}




    

def getLabelMap():
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    coffea_file = max(glob.glob(coffea_dir+'*.coffea'), key=os.path.getctime)
    label_map = util.load(coffea_file)['analysisCategories']
    
    return label_map
    
    

def getRapidity(p4):
    
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))



    
    
def loadCoffeaFile(dataset='QCD', year='2016', tag='', bkgest=False):
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    
    coffeaFiles = getCoffeaFilenames()
        
    if bkgest: 
        bkgest_str = 'weighted'
    else: 
        bkgest_str = 'unweighted'

    if len(tag) > 0:
        file = coffeaFiles[dataset][bkgest_str][year][tag]
    else:
        file = coffeaFiles[dataset][bkgest_str][year]
        
        
    return util.load(file)   
    
    
    
    
def getHist(hname, ds, bkgest, year, sum_axes=['anacat'], integrate_axes={}):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # bkgest = boolean, True if bkg estimate applied                                     #
    # year = '2016APV' or '2016' or '2017' or '2018'                                     #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    ######################################################################################    

    
    # load histograms and get scale factors
    coffeaFiles = getCoffeaFilenames()
    
    cfiles = []
    sf = []
    bkgest_str = 'weighted' if bkgest else 'unweighted'
    
    for key, file in coffeaFiles[ds][bkgest_str][year].items():
        loaded_file = util.load(file)
        cfiles.append(loaded_file)
        
        if 'TTbar' in ds and '700to1000' in key:
            sf.append(lumi[year] * ttbar_xs1 * toptag_sf**2 / loaded_file['cutflow']['sumw'])
        elif 'TTbar' in ds and '1000toInf' in key:
            sf.append(lumi[year] * ttbar_xs2 * toptag_sf**2 / loaded_file['cutflow']['sumw'])     
        elif 'QCD' in ds:
            sf.append(lumi[IOV] * qcd_xs / loaded_file['cutflow']['sumw'])  
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




def getHist2(hname, ds, year, sum_axes=['anacat'], integrate_axes={}, tag=''):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # year = '2016APV', '2016', '2017', '2018', '2016all', 'all'                         #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    # tag = string on end of file (ie _bkgest or _test)                                  #
    ######################################################################################    
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/scale/'
    
    # examples: outputs/scale/TTbar_2016.coffea, outputs/scale/QCD_2016all.coffea
    filename = coffea_dir + ds + '_' + year + tag + '.coffea'
    
    output = util.load(filename)
    
    
    # integrate and sum over axes
    sum_axes_dict = {ax:sum for ax in sum_axes}

    # get histogram
    histo = output[hname][integrate_axes][sum_axes_dict]
            
            
    return histo
    
            
def plotBackgroundEstimate(hdata, hntmj, httbar, year, text=''):
    
    hbkg = hntmj + httbar
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

    
    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(lumi[year]/1000.), year=year, loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hdata,  ax=ax1, histtype='errorbar', color='black', label='Data')
    hep.histplot(hbkg,   ax=ax1, histtype='fill', color='xkcd:pale gold', label='NTMJ')
    hep.histplot(httbar, ax=ax1, histtype='fill', color='xkcd:deep red', label='TTbar')


    ratio_plot =  hdata / hbkg.values()
    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(0,2)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')

    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_ylabel('Events')
    ax1.set_xlabel('')
    ax1.set_ylim(1e-1, 1e7)
    ax1.set_xlim(900, 6000)
    ax2.set_xlim(900, 6000)            
    

    
def makeSaveDirectories():
    
        
    plots_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/plots'
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs'
    
    directories = [
        
        # output coffea and root files
        coffea_dir+'/scale',
        coffea_dir+'/twodalphabet',
       
        # QCD closure test
        plots_dir+'/images/png/closureTestQCD/2016all',
        plots_dir+'/images/png/closureTestQCD/2016APV',
        plots_dir+'/images/png/closureTestQCD/2016',
        plots_dir+'/images/png/closureTestQCD/2017',
        plots_dir+'/images/png/closureTestQCD/2018',
        plots_dir+'/images/pdf/closureTestQCD/2016all',
        plots_dir+'/images/pdf/closureTestQCD/2016APV',
        plots_dir+'/images/pdf/closureTestQCD/2016',
        plots_dir+'/images/pdf/closureTestQCD/2017',
        plots_dir+'/images/pdf/closureTestQCD/2018',

        # Closure test
        plots_dir+'/images/png/closureTest/2016all',
        plots_dir+'/images/png/closureTest/2016APV',
        plots_dir+'/images/png/closureTest/2016',
        plots_dir+'/images/png/closureTest/2017',
        plots_dir+'/images/png/closureTest/2018',
        plots_dir+'/images/pdf/closureTest/2016all',
        plots_dir+'/images/pdf/closureTest/2016APV',
        plots_dir+'/images/pdf/closureTest/2016',
        plots_dir+'/images/pdf/closureTest/2017',
        plots_dir+'/images/pdf/closureTest/2018',

        # systematics
        plots_dir+'/images/png/systematics/2016all',
        plots_dir+'/images/png/systematics/2016APV',
        plots_dir+'/images/png/systematics/2016',
        plots_dir+'/images/png/systematics/2017',
        plots_dir+'/images/png/systematics/2018',
        plots_dir+'/images/pdf/systematics/2016all',
        plots_dir+'/images/pdf/systematics/2016APV',
        plots_dir+'/images/pdf/systematics/2016',
        plots_dir+'/images/pdf/systematics/2017',
        plots_dir+'/images/pdf/systematics/2018',
        
        # ttbarmass
        plots_dir+'/images/png/ttbarmass/2016all',
        plots_dir+'/images/png/ttbarmass/2016APV',
        plots_dir+'/images/png/ttbarmass/2016',
        plots_dir+'/images/png/ttbarmass/2017',
        plots_dir+'/images/png/ttbarmass/2018',
        plots_dir+'/images/pdf/ttbarmass/2016all',
        plots_dir+'/images/pdf/ttbarmass/2016APV',
        plots_dir+'/images/pdf/ttbarmass/2016',
        plots_dir+'/images/pdf/ttbarmass/2017',
        plots_dir+'/images/pdf/ttbarmass/2018',
        
    ]


    for path in directories:
        if not os.path.exists(path):
            os.makedirs(path)



def getCoffeaFilenames(useDeepAK8=False, useOldHTcut=False):
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
#     coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/local/history/transferfxn/coffea/'
    
    deepak8str = ''
    if useDeepAK8:
        deepak8str = '_DeepAK8'
    oldHTstr = ''
    if useOldHTcut:
        oldHTstr = '_oldHTcut_oldBTag'
    
    coffeaFiles = {
        "JetHT":{
            "unweighted": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "C": coffea_dir+'JetHT_2016APVC_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "D": coffea_dir+'JetHT_2016APVD_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "E": coffea_dir+'JetHT_2016APVE_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "F": coffea_dir+'JetHT_2016APVF_noSyst'+deepak8str+oldHTstr+'.coffea',
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "G": coffea_dir+'JetHT_2016G_noSyst'+deepak8str+oldHTstr+'.coffea',
                    "H": coffea_dir+'JetHT_2016H_noSyst'+deepak8str+oldHTstr+'.coffea',

                },
                "2017": {
                    "B": coffea_dir+'JetHT_2017B_blinded'+deepak8str+'.coffea',
                    "C": coffea_dir+'JetHT_2017C_blinded'+deepak8str+'.coffea',
                    "D": coffea_dir+'JetHT_2017D_blinded'+deepak8str+'.coffea',
                    "E": coffea_dir+'JetHT_2017E_blinded'+deepak8str+'.coffea',
                    "F": coffea_dir+'JetHT_2017F_blinded'+deepak8str+'.coffea'
                },
                "2018": {
                    "A": coffea_dir+'JetHT_2018A_blinded'+deepak8str+'.coffea',
                    "B": coffea_dir+'JetHT_2018B_blinded'+deepak8str+'.coffea',
                    "C": coffea_dir+'JetHT_2018C_blinded'+deepak8str+'.coffea',
                    "D": coffea_dir+'JetHT_2018D_blinded'+deepak8str+'.coffea'
                }
            },
            "weighted": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "C": coffea_dir+'JetHT_2016APVC_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "D": coffea_dir+'JetHT_2016APVD_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "E": coffea_dir+'JetHT_2016APVE_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "F": coffea_dir+'JetHT_2016APVF_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "G": coffea_dir+'JetHT_2016G_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "H": coffea_dir+'JetHT_2016H_noSyst_bkgest'+deepak8str+oldHTstr+'.coffea',

                },
                "2017": {
                    "B": coffea_dir+'JetHT_2017B_bkgest_blinded'+deepak8str+'.coffea',
                    "C": coffea_dir+'JetHT_2017C_bkgest_blinded'+deepak8str+'.coffea',
                    "D": coffea_dir+'JetHT_2017D_bkgest_blinded'+deepak8str+'.coffea',
                    "E": coffea_dir+'JetHT_2017E_bkgest_blinded'+deepak8str+'.coffea',
                    "F": coffea_dir+'JetHT_2017F_bkgest_blinded'+deepak8str+'.coffea'
                },
                "2018": {
                    "A": coffea_dir+'JetHT_2018A_bkgest_blinded'+deepak8str+'.coffea',
                    "B": coffea_dir+'JetHT_2018B_bkgest_blinded'+deepak8str+'.coffea',
                    "C": coffea_dir+'JetHT_2018C_bkgest_blinded'+deepak8str+'.coffea',
                    "D": coffea_dir+'JetHT_2018D_bkgest_blinded'+deepak8str+'.coffea'
                }
            },
            "noMassMod": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "C": coffea_dir+'JetHT_2016APVC_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "D": coffea_dir+'JetHT_2016APVD_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "E": coffea_dir+'JetHT_2016APVE_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "F": coffea_dir+'JetHT_2016APVF_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "G": coffea_dir+'JetHT_2016G_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "H": coffea_dir+'JetHT_2016H_noSyst_bkgest_noMassMod'+oldHTstr+'.coffea',

                },
                "2017": {
                    "B": coffea_dir+'JetHT_2017B_bkgest_blinded_noMassMod.coffea',
                    "C": coffea_dir+'JetHT_2017C_bkgest_blinded_noMassMod.coffea',
                    "D": coffea_dir+'JetHT_2017D_bkgest_blinded_noMassMod.coffea',
                    "E": coffea_dir+'JetHT_2017E_bkgest_blinded_noMassMod.coffea',
                    "F": coffea_dir+'JetHT_2017F_bkgest_blinded_noMassMod.coffea'
                },
                "2018": {
                    "A": coffea_dir+'JetHT_2018A_bkgest_blinded_noMassMod.coffea',
                    "B": coffea_dir+'JetHT_2018B_bkgest_blinded_noMassMod.coffea',
                    "C": coffea_dir+'JetHT_2018C_bkgest_blinded_noMassMod.coffea',
                    "D": coffea_dir+'JetHT_2018D_bkgest_blinded_noMassMod.coffea'
                }
            }
        },

        "TTbar": {
            "unweighted": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000'+deepak8str+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf'+deepak8str+oldHTstr+'.coffea',
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000'+deepak8str+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf'+deepak8str+oldHTstr+'.coffea',
                },
                "2017": {
                    "700to1000": coffea_dir+'TTbar_2017_700to1000'+deepak8str+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2017_1000toInf'+deepak8str+'.coffea'
                },
                "2018": {
                    "700to1000": coffea_dir+'TTbar_2018_700to1000'+deepak8str+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2018_1000toInf'+deepak8str+'.coffea'
                }
            },
            "weighted": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf_bkgest'+deepak8str+oldHTstr+'.coffea',
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000_bkgest'+deepak8str+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf_bkgest'+deepak8str+oldHTstr+'.coffea',
                },
                "2017": {
                    "700to1000": coffea_dir+'TTbar_2017_700to1000_bkgest'+deepak8str+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2017_1000toInf_bkgest'+deepak8str+'.coffea'
                },
                "2018": {
                    "700to1000": coffea_dir+'TTbar_2018_700to1000_bkgest'+deepak8str+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2018_1000toInf_bkgest'+deepak8str+'.coffea'
                }
                
            },
            "noMassMod": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf_bkgest_noMassMod'+oldHTstr+'.coffea',
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000_bkgest_noMassMod'+oldHTstr+'.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf_bkgest_noMassMod'+oldHTstr+'.coffea',
                },
                "2017": {
                    "700to1000": coffea_dir+'TTbar_2017_700to1000_bkgest_noMassMod.coffea',
                    "1000toInf": coffea_dir+'TTbar_2017_1000toInf_bkgest_noMassMod.coffea'
                },
                "2018": {
                    "700to1000": coffea_dir+'TTbar_2018_700to1000_bkgest_noMassMod.coffea',
                    "1000toInf": coffea_dir+'TTbar_2018_1000toInf_bkgest_noMassMod.coffea'
                }
                
            },
        },
        
        "QCD": {
            "unweighted": {
                "2016APV": {
                    "300to470": coffea_dir+'QCD_2016APV_300to470_noSyst'+oldHTstr+'.coffea',
                    "470to600": coffea_dir+'QCD_2016APV_470to600_noSyst'+oldHTstr+'.coffea',
                    "600to800": coffea_dir+'QCD_2016APV_600to800_noSyst'+oldHTstr+'.coffea',
                    "800to1000": coffea_dir+'QCD_2016APV_800to1000_noSyst'+oldHTstr+'.coffea',
                    "1000to1400": coffea_dir+'QCD_2016APV_1000to1400_noSyst'+oldHTstr+'.coffea',
                    "1400to1800": coffea_dir+'QCD_2016APV_1400to1800_noSyst'+oldHTstr+'.coffea',
                    "1800to2400": coffea_dir+'QCD_2016APV_1800to2400_noSyst'+oldHTstr+'.coffea',
                    "2400to3200": coffea_dir+'QCD_2016APV_2400to3200_noSyst'+oldHTstr+'.coffea',
                    "3200toInf": coffea_dir+'QCD_2016APV_3200toInf_noSyst'+oldHTstr+'.coffea'
                },
                "2016": {
                    "300to470": coffea_dir+'QCD_2016_300to470_noSyst'+oldHTstr+'.coffea',
                    "470to600": coffea_dir+'QCD_2016_470to600_noSyst'+oldHTstr+'.coffea',
                    "600to800": coffea_dir+'QCD_2016_600to800_noSyst'+oldHTstr+'.coffea',
                    "800to1000": coffea_dir+'QCD_2016_800to1000_noSyst'+oldHTstr+'.coffea',
                    "1000to1400": coffea_dir+'QCD_2016_1000to1400_noSyst'+oldHTstr+'.coffea',
                    "1400to1800": coffea_dir+'QCD_2016_1400to1800_noSyst'+oldHTstr+'.coffea',
                    "1800to2400": coffea_dir+'QCD_2016_1800to2400_noSyst'+oldHTstr+'.coffea',
                    "2400to3200": coffea_dir+'QCD_2016_2400to3200_noSyst'+oldHTstr+'.coffea',
                    "3200toInf": coffea_dir+'QCD_2016_3200toInf_noSyst'+oldHTstr+'.coffea'
                },
                "2017": coffea_dir+'QCD_2017'+deepak8str+'.coffea',
                "2018": coffea_dir+'QCD_2018'+deepak8str+'.coffea'
            },
            "weighted": {
                "2016APV": {
                    "300to470": coffea_dir+'QCD_2016APV_300to470_noSyst_bkgest'+oldHTstr+'.coffea',
                    "470to600": coffea_dir+'QCD_2016APV_470to600_noSyst_bkgest'+oldHTstr+'.coffea',
                    "600to800": coffea_dir+'QCD_2016APV_600to800_noSyst_bkgest'+oldHTstr+'.coffea',
                    "800to1000": coffea_dir+'QCD_2016APV_800to1000_noSyst_bkgest'+oldHTstr+'.coffea',
                    "1000to1400": coffea_dir+'QCD_2016APV_1000to1400_noSyst_bkgest'+oldHTstr+'.coffea',
                    "1400to1800": coffea_dir+'QCD_2016APV_1400to1800_bkgest'+oldHTstr+'.coffea',
                    "1800to2400": coffea_dir+'QCD_2016APV_1800to2400_bkgest'+oldHTstr+'.coffea',
                    "2400to3200": coffea_dir+'QCD_2016APV_2400to3200_noSyst_bkgest'+oldHTstr+'.coffea',
                    "3200toInf": coffea_dir+'QCD_2016APV_3200toInf_noSyst_bkgest'+oldHTstr+'.coffea'
                },
                "2016": {
                    "300to470": coffea_dir+'QCD_2016_300to470_noSyst_bkgest'+oldHTstr+'.coffea',
                    "470to600": coffea_dir+'QCD_2016_470to600_noSyst_bkgest'+oldHTstr+'.coffea',
                    "600to800": coffea_dir+'QCD_2016_600to800_noSyst_bkgest'+oldHTstr+'.coffea',
                    "800to1000": coffea_dir+'QCD_2016_800to1000_noSyst_bkgest'+oldHTstr+'.coffea',
                    "1000to1400": coffea_dir+'QCD_2016_1000to1400_noSyst_bkgest'+oldHTstr+'.coffea',
                    "1400to1800": coffea_dir+'QCD_2016_1400to1800_noSyst_bkgest'+oldHTstr+'.coffea',
                    "1800to2400": coffea_dir+'QCD_2016_1800to2400_noSyst_bkgest'+oldHTstr+'.coffea',
                    "2400to3200": coffea_dir+'QCD_2016_2400to3200_noSyst_bkgest'+oldHTstr+'.coffea',
                    "3200toInf": coffea_dir+'QCD_2016_3200toInf_noSyst_bkgest'+oldHTstr+'.coffea'
                },
                "2017": coffea_dir+'QCD_2017_bkgest.coffea',
                "2018": coffea_dir+'QCD_2018_bkgest.coffea'
            }
        },
        
        "ZPrimeDM": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_DM_2016APV'+oldHTstr+'.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2016APV'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2016APV'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2016APV'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2016APV'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2016APV'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2016APV'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2016APV'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2016APV'+oldHTstr+'.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_DM_2016'+oldHTstr+'.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2016'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2016'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2016'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2016'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2016'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2016'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2016'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2016'+oldHTstr+'.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_DM_2017.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2017.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2017.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2017.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2017.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_DM_2018.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2018.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2018.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2018.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2018.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2018.coffea'
                }
            }
        },

        "ZPrime1": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_1_2016APV'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2016APV'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2016APV'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2016APV'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2016APV'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2016APV'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2016APV'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2016APV'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2016APV'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2016APV'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2016APV'+oldHTstr+'.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_1_2016'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2016'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2016'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2016'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2016'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2016'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2016'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2016'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2016'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2016'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2016'+oldHTstr+'.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_1_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_1_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2018.coffea'
                }
            }
        },
        
        "ZPrime10": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_10_2016APV'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2016APV'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2016APV'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2016APV'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2016APV'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2016APV'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2016APV'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2016APV'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2016APV'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2016APV'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2016APV'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2016APV'+oldHTstr+'.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_10_2016'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2016'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2016'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2016'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2016'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2016'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2016'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2016'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2016'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2016'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2016'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2016'+oldHTstr+'.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_10_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_10_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2018.coffea'
                }
            }
        },
            
        "ZPrime30": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_30_2016APV'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2016APV'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2016APV'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2016APV'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2016APV'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2016APV'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2016APV'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2016APV'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2016APV'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2016APV'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2016APV'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2016APV'+oldHTstr+'.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_30_2016'+oldHTstr+'.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2016'+oldHTstr+'.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2016'+oldHTstr+'.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2016'+oldHTstr+'.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2016'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2016'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2016'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2016'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2016'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2016'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2016'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2016'+oldHTstr+'.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_30_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_30_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2018.coffea'
                }
            }
        },
        
        "RSGluon": {
            "unweighted": {
                "2016": {
                    "1000": coffea_dir+'RSGluon1000_2016'+oldHTstr+'.coffea',
                    "1500": coffea_dir+'RSGluon1500_2016'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'RSGluon2000_2016'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'RSGluon2500_2016'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'RSGluon3000_2016'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'RSGluon3500_2016'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'RSGluon4000_2016'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'RSGluon4500_2016'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'RSGluon5000_2016'+oldHTstr+'.coffea'
                },
                "2016APV": {
                    "1000": coffea_dir+'RSGluon1000_2016APV'+oldHTstr+'.coffea',
                    "1500": coffea_dir+'RSGluon1500_2016APV'+oldHTstr+'.coffea',
                    "2000": coffea_dir+'RSGluon2000_2016APV'+oldHTstr+'.coffea',
                    "2500": coffea_dir+'RSGluon2500_2016APV'+oldHTstr+'.coffea',
                    "3000": coffea_dir+'RSGluon3000_2016APV'+oldHTstr+'.coffea',
                    "3500": coffea_dir+'RSGluon3500_2016APV'+oldHTstr+'.coffea',
                    "4000": coffea_dir+'RSGluon4000_2016APV'+oldHTstr+'.coffea',
                    "4500": coffea_dir+'RSGluon4500_2016APV'+oldHTstr+'.coffea',
                    "5000": coffea_dir+'RSGluon5000_2016APV'+oldHTstr+'.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'RSGluon1000_2017.coffea',
                    "1500": coffea_dir+'RSGluon1500_2017.coffea',
                    "2000": coffea_dir+'RSGluon2000_2017.coffea',
                    "2500": coffea_dir+'RSGluon2500_2017.coffea',
                    "3000": coffea_dir+'RSGluon3000_2017.coffea',
                    "3500": coffea_dir+'RSGluon3500_2017.coffea',
                    "4000": coffea_dir+'RSGluon4000_2017.coffea',
                    "4500": coffea_dir+'RSGluon4500_2017.coffea',
                    "5000": coffea_dir+'RSGluon5000_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'RSGluon1000_2018.coffea',
                    "1500": coffea_dir+'RSGluon1500_2018.coffea',
                    "2000": coffea_dir+'RSGluon2000_2018.coffea',
                    "2500": coffea_dir+'RSGluon2500_2018.coffea',
                    "3000": coffea_dir+'RSGluon3000_2018.coffea',
                    "3500": coffea_dir+'RSGluon3500_2018.coffea',
                    "4000": coffea_dir+'RSGluon4000_2018.coffea',
                    "4500": coffea_dir+'RSGluon4500_2018.coffea',
                    "5000": coffea_dir+'RSGluon5000_2018.coffea'
                }
            }
        }
    }
    
    
        
    return coffeaFiles
    
    

    
