#!/usr/bin/env python 
# coding: utf-8

from coffea import processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
import sys
import os, psutil
import copy
import scipy.stats as ss
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
import random
import correctionlib
import hist
import json

import awkward as ak

# for dask, `from python.corrections import` does not work
sys.path.append(os.getcwd()+'/python/')

from corrections import (
    GetFlavorEfficiency,
    HEMCleaning,
    HEMVeto,
    GetL1PreFiringWeight,
    GetJECUncertainties,
    GetPDFWeights,
    GetPUSF,
    GetQ2weights,
    GetTopTagSFweights,
    GetLumiweights,
    getLumiMaskRun2,
    getMETFilter,
    pTReweighting,
)
from btagCorrections import btagCorrections
from functions import getRapidity



#ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]


def update(events, collections):
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self,
                 htCut=1400.,
                 ak8PtMin=400.,
                 minMSD=105.,
                 maxMSD=210.,
                 tau32Cut=0.65,
                 bdisc=0.5847,
                 deepAK8Cut='tight',
                 useDeepAK8=False,
                 blinding=True,
                 MassMod=True,
                 iov='2016',
                 bkgEst=False,
                 noSyst=False,
                 Sideband=False,
                 systematics = ['nominal', 'pileup'],
                 anacats = ['2t0bcen'],
                 #rpf_params = {'params':[1.0], 'errors':[0.0]},
                ):
                 
        self.iov = iov
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.useDeepAK8 = useDeepAK8
        self.means_stddevs = defaultdict()
        self.bkgEst = bkgEst
        self.noSyst = noSyst
        self.blinding = blinding
        self.MassMod = MassMod
        self.systematics = systematics
        self.Sideband = Sideband
        #self.rpf_params = rpf_params        
        
#         self.transfer_function = np.load('plots/save.npy')

        deepak8cuts = {
            'loose':{ # 1%
                '2016APV': 0.486, 
                '2016': 0.475,
                '2017': 0.487,
                '2018': 0.477,
            },
            'medium':{ # 0.5%
                '2016APV': 0.677, 
                '2016': 0.666,
                '2017': 0.673,
                '2018': 0.669,
            },
            'tight': { # 0.1%
                '2016APV': 0.902, 
                '2016': 0.897,
                '2017': 0.898,
                '2018': 0.900,
            } 
        }
    
        btagcuts = {
            'loose':{
                '2016APV': 0.2027, 
                '2016':    0.1918,
                '2017':    0.1355,
                '2018':    0.1208,
            },
            'medium':{
                '2016APV': 0.6001, 
                '2016':    0.5847,
                '2017':    0.4506,
                '2018':    0.4506,
            } 
        }

        self.weights = {}
    
        
        
        self.deepAK8Cut = deepak8cuts['medium'][self.iov]
        
        
        
        self.bdisc = btagcuts['medium'][self.iov] # old 2016AN --> 0.8484
        
        
        
        
        
        # analysis categories #
        self.anacats = anacats
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}

        
        # systematics
        syst_category_strings = ['nominal']
        if not self.noSyst:
            for s in self.systematics:
                if (s != 'nominal'):
                    
                    if ('hem' in s):
                        syst_category_strings.append(s)
                    else:
                        syst_category_strings.append(s+'Down')
                        syst_category_strings.append(s+'Up')
        
#         syst_category_strings = ['nominal', 'test1', 'test2', 'test3', 'test4']
        manual_jetht_bins = [800, 840, 880, 920, 960, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1600, 1800]
    
        # axes
        dataset_axis     = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")
        syst_axis        = hist.axis.StrCategory(syst_category_strings, name="systematic")
        ttbarmass_axis   = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass_axis     = hist.axis.Regular(50, 0, 500, name="jetmass", label=r"Jet $m$ [GeV]")
        ttbarmass2D_axis = hist.axis.Regular(20, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetHT2D_axis     = hist.axis.Variable(manual_jetht_bins, name = "Jet_HT", label = r'$AK4\ Jet\ HT$')
        jetmass2D_axis   = hist.axis.Regular(20, 0, 500, name="sdjetmass", label=r"Jet $m_{SD}$ [GeV]")
        jetpt_axis       = hist.axis.Regular(50, 400, 2000, name="jetpt", label=r"Jet $p_{T}$ [GeV]")
        jetp_axis        = hist.axis.Regular(100, 400, 3600, name="jetp", label=r"Jet $p$ [GeV]")
        jeteta_axis      = hist.axis.Regular(50, -2.4, 2.4, name="jeteta", label=r"Jet $\eta$")
        jetphi_axis      = hist.axis.Regular(50, -np.pi, np.pi, name="jetphi", label=r"Jet $\phi$")
        cats_axis        = hist.axis.IntCategory(range(len(self.anacats)), name="anacat", label="Analysis Category")
        manual_axis      = hist.axis.Variable(manual_bins, name="jetp", label=r"Jet Momentum [GeV]")
        btag_axis        = hist.axis.Regular(10, 0, 1, name="bdisc", label=r"DeepCSV")
        ttag_axis        = hist.axis.Regular(10, 0, 1, name="tdisc", label=r"DeepAK8")
        nsub_axis        = hist.axis.Regular(10, 0, 1, name="nsub", label=r"$\tau_{3} / \tau_{2}$")

        
        # output
        self.histo_dict = {

            
            # histograms
            'ttbarmass'  : hist.Hist(syst_axis, cats_axis, ttbarmass2D_axis, storage="weight", name="Counts"),
            'ttbarmass_fine'  : hist.Hist(syst_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_bare'  : hist.Hist(syst_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'numerator'  : hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'denominator': hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'jetmass' : hist.Hist(cats_axis, jetmass_axis, storage="weight", name="Counts"),
            'jetpt'  : hist.Hist(cats_axis, jetpt_axis, storage="weight", name="Counts"),
            'jeteta'  : hist.Hist(cats_axis, jeteta_axis, storage="weight", name="Counts"),
            'jetphi'  : hist.Hist(cats_axis, jetphi_axis, storage="weight", name="Counts"),
            'jetp'  : hist.Hist(cats_axis, jetp_axis, storage="weight", name="Counts"),
            'jetp_m' : hist.Hist(cats_axis, manual_axis, storage="weight", name="Counts"),
            'discriminators'  : hist.Hist(cats_axis,
                                          jetp_axis,
                                          btag_axis,
                                          ttag_axis,
                                          nsub_axis,
                                          storage="weight", name="Counts"),
            'deepak8'  : hist.Hist(cats_axis,
                                          jetp_axis,
                                          ttbarmass_axis,
                                          ttag_axis,
                                          storage="weight", name="Counts"),
            
            
            'mtt_vs_HT' : hist.Hist(syst_axis, cats_axis, ttbarmass2D_axis, jetHT2D_axis, storage="weight", name="Counts"),

            
            'deepak8_over_jetp': hist.Hist(cats_axis, ttag_axis, jetp_axis, storage="weight", name="Counts"),
            'tau32_over_jetp': hist.Hist(cats_axis, nsub_axis, jetp_axis, storage="weight", name="Counts"),
            'bdisc_over_jetpt': hist.Hist(cats_axis, btag_axis, jetp_axis, storage="weight", name="Counts"),


                        
            # accumulators
            'cutflow': processor.defaultdict_accumulator(int),
            'weights': processor.defaultdict_accumulator(float),
            
        }
        
        
        
    @property
    def accumulator(self):
        return self._accumulator
    
    
    
    def process(self, events):
        
        # reference for return processor.accumulate
        # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
        

        nEvents = len(events.event)
        
        # print('================================== FIRST LINE ========================================\n')
        # print('# of events before cut:', nEvents)
        # print('All Gen weights: ', ak.sort(events.genWeight, ascending=False))
        # print('Largest Gen weight = ', ak.max(events.genWeight))
        # print('Gen weights to be removed from variation cut:', ak.sort(events.genWeight[events.Generator.binvar > 400], ascending=False))
        # print('Largest Gen weight kept = ', ak.max(events.genWeight[events.Generator.binvar > 400]))
        # print('# of events to be removed from variation cut:', len(events.event)-len(events[ events.Generator.binvar > 400 ].event))
        # Remove events with large weights
        # if "QCD" in events.metadata['dataset']: # and ('2017' not in self.iov): 
        #     events = events[ events.Generator.binvar > 600 ]
        #     # print('# of events after variation of 400 cut:', len(events.event))
        #     if events.metadata['dataset'] not in self.means_stddevs : 
        #         average = np.average( events.genWeight )
        #         stddev = np.std( events.genWeight )
        #         self.means_stddevs[events.metadata['dataset']] = (average, stddev) #Defines structure/order of default dict
        #     average,stddev = self.means_stddevs[events.metadata['dataset']] #Assigns variables
        #     # print('Average Gen weight after variation of 400 cut:', average)
        #     vals = (events.genWeight - average ) / stddev # is variation within some stddev ?
        # #     # print('Gen weights removed; larger than 2 sigma:', ak.sort(events.genWeight[(np.abs(vals) < 2)], ascending=False))
        #     events = events[(vals < 2)] # Only accept gen weights within 2 stddev of average
            # print('Average Gen weight after sigma cut:', np.average( events.genWeight ))
            # print('Gen weights kept:', ak.sort(events.genWeight, ascending=False))
            # print('# of events that go to pre-selection:', len(events.event))
            # print('\n================================== LAST LINE ========================================\n\n')

        
        isData = ('JetHT' in events.metadata['dataset']) or ('SingleMu' in events.metadata['dataset'])
        
        noCorrections = (not 'jes' in self.systematics and not 'jer' in self.systematics)

        # if isData or noCorrections:
        #     return self.process_analysis(events, 'nominal')

        if noCorrections or self.noSyst:
            return self.process_analysis(events, 'nominal', nEvents)
        
        
        if isData:
            
            return processor.accumulate([
                self.process_analysis(events, 'nominal', nEvents),
                self.process_analysis(events, 'hemVeto', nEvents)
            ]) 
        
        
        FatJets = events.FatJet
        GenJets = events.GenJet
        Jets = events.Jet
        
                
        
        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        
        # FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        # GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        # Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        FatJets["matched_gen_0p2"] = FatJets.p4.nearest(GenJets.p4, threshold=0.2)
        FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.matched_gen_0p2.pt, 0), np.float32)

        Jets["matched_gen_0p2"] = Jets.p4.nearest(GenJets.p4, threshold=0.2)
        Jets["pt_gen"] = ak.values_astype(ak.fill_none(Jets.matched_gen_0p2.pt, 0), np.float32)


        corrected_fatjets = GetJECUncertainties(FatJets, events, self.iov, R='AK8', isData=isData)
        corrected_jets = GetJECUncertainties(Jets, events, self.iov, R='AK4', isData=isData)
        
        
        
        if 'jes' in self.systematics:
            corrections = [
                ({"Jet": corrected_jets, "FatJet": corrected_fatjets}, 'nominal'),
                ({"Jet": corrected_jets.JES_jes.up, "FatJet": corrected_fatjets.JES_jes.up}, "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down, "FatJet": corrected_fatjets.JES_jes.down}, "jesDown"),
            ]
        if 'jer' in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JER.up, "FatJet": corrected_fatjets.JER.up}, "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": corrected_fatjets.JER.down}, "jerDown"),
            ])
            
            
                
        # loop through corrections
        outputs = []
        for collections, name in corrections:
            outputs.append(self.process_analysis(update(events, collections), name, nEvents))
            
        output_total = processor.accumulate(outputs)                       

                        
        return output_total
        


    def process_analysis(self, events, correction, nEvents):
        
        
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        
        isNominal = (correction=='nominal')
        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        
        if (self.iov == '2018'):
            
            if isData:
                
                # keep events below 
                    
                    
                events = events[HEMVeto(events.Jet, events.FatJet, events.run)]


            else:
                events = events[HEMVeto(events.Jet, events.FatJet, events.run)]
                
        output = self.histo_dict

        if isNominal:
            output['cutflow']['all events'] += nEvents
            if not isData:
                output['cutflow']['sumw'] += np.sum(events.genWeight)
            
        # lumi mask #
        if (isData):
            
            lumi_mask = np.array(getLumiMaskRun2(self.iov)(events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
            del lumi_mask
            
        # event selection #
        selection = PackedSelection()
        
        
        # Remove events with large weights
        if "15to7000" in events.metadata['dataset']: # and ('2017' not in self.iov): 
            # print('events: ', events)
            events = events[ events.Generator.binvar > 400 ]
            # print('events after weight removal: ', events)
            if events.metadata['dataset'] not in self.means_stddevs : 
                average = np.average( events.genWeight )
                stddev = np.std( events.genWeight )
                self.means_stddevs[events.metadata['dataset']] = (average, stddev)            
            average,stddev = self.means_stddevs[events.metadata['dataset']]
            vals = (events.genWeight - average ) / stddev
            events = events[(vals < 2)]
            # print('events after final weight removal: ', events)
        
        
        # blinding #
        if (isData and self.blinding) and (('2017' in self.iov) or ('2018' in self.iov)): 
            events = events[::10]
            

        # trigger cut #
        if isData:
            
            triggernames = { 
            
            "2016APV": ["PFHT900"],
            "2016" : ["PFHT900"], #["PFHT900", "AK8PFJet360_TrimMass30", "AK8PFJet450"],
            "2017" : ["PFHT1050"],
            "2018" : ["PFHT1050"],
        
            }
                        
            selection.add('trigger', events.HLT[triggernames[self.iov][0]])
            trigPass = events.HLT[triggernames[self.iov][0]]
            events = events[trigPass]
            output['cutflow']['Passed Trigger(s)'] += len(events.FatJet)#ak.sum(trigPass)
            del trigPass
            

        # objects #
        
        FatJets = events.FatJet
        SubJets = events.SubJet
        Jets    = events.Jet

        FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        SubJets["p4"] = ak.with_name(SubJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        Jets["p4"]    = ak.with_name(Jets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")

        if not isData:
            GenJets = events.GenJet
            GenJets["p4"] = ak.with_name(GenJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    
        
        # ---- Get event weights from dataset ---- #

        # if blinding + trigger results in few events
        if (len(events) < 10): return output
        
        if isData:
            evtweights = np.ones(len(events))
        else:
            if "LHEWeight_originalXWGTUP" not in events.fields: 
                evtweights = events.genWeight
            else: 
                evtweights = events.LHEWeight_originalXWGTUP
                print('LHE event weights used: ', evtweights)
                print('Generator events replaced: ', events.genWeight)
                if isNominal:
                    print('SumW after change to LHE: ', np.sum(evtweights))
                    print('-----------------------\n')
                    
        # if correction == 'nominal':
            # output['cutflow']['all events'] += len(FatJets)
            # output['cutflow']['sumw'] += np.sum(evtweights)
            # output['cutflow']['sumw2'] += np.sum(evtweights**2)

        # # Remove events with large weights
        # if "QCD" in events.metadata['dataset']: # and ('2017' not in self.iov): 
        #     events = events[ events.Generator.binvar > 600 ]
            
        #     if events.metadata['dataset'] not in self.means_stddevs : 
        #         average = np.average( events.genWeight )
        #         stddev = np.std( events.genWeight )
        #         self.means_stddevs[events.metadata['dataset']] = (average, stddev)            
        #     average,stddev = self.means_stddevs[events.metadata['dataset']]
        #     vals = (events.genWeight - average ) / stddev
        #     events = events[(vals < 2)]
              
        
        # ---- event selection and object selection ---- #
        
        # met filters #
        if isData:
#             selection.add('metfilter', getMETFilter(self.iov, events))
            filteredEvents = getMETFilter(self.iov, events)#np.array([getattr(events, f'Flag_{getMETFilter(self.iov, events)[i]}') for i in range(len(getMETFilter(self.iov, events)))])
#             filteredEvents = np.logical_or.reduce(filteredEvents, axis=0)
        
            if ak.sum(filteredEvents) < 1 :
                print("\nNo events passed the MET filters.\n", flush=True)
                return output
            else:
                FatJets = FatJets[filteredEvents]
                Jets = Jets[filteredEvents]
                SubJets = SubJets[filteredEvents]
                evtweights = evtweights[filteredEvents]
                events = events[filteredEvents]
            output['cutflow']['Passed MET Filters'] += ak.sum(filteredEvents)
        
        # ht cut #
#         selection.add('htCut',
#             ak.sum(Jets.pt, axis=1) > self.htCut
#         )
        htPass = ak.sum(Jets.pt, axis=1) > self.htCut
        FatJets = FatJets[htPass]
        Jets = Jets[htPass]
        SubJets = SubJets[htPass]
        evtweights = evtweights[htPass]
        events = events[htPass]
        if not isData:
            GenJets = GenJets[htPass]
        output['cutflow']['Passed HT Cut'] += ak.sum(htPass)
        del htPass
                
        # jet id #
#         selection.add('jetid', ak.any((FatJets.jetId > 0), axis=1))
        idPass = FatJets.jetId > 0
        FatJets = FatJets[idPass]
        output['cutflow']['Passed Loose Jet ID'] += len(FatJets)
        del idPass
                
        # jet kinematics # 
        jetkincut = (FatJets.pt > self.ak8PtMin) & (np.abs(getRapidity(FatJets.p4)) < 2.4)
#         selection.add('jetkincut', ak.any(jetkincut, axis=1))
        FatJets = FatJets[jetkincut]
        output['cutflow']['Passed pT,y Cut'] += len(FatJets)
        del jetkincut
        
        
        # at least 2 ak8 jets #
#         selection.add('twoFatJets', (ak.num(FatJets) >= 2))
        twoak8Pass = (ak.num(FatJets) >= 2)
        FatJets = FatJets[twoak8Pass]
        SubJets = SubJets[twoak8Pass]
        Jets = Jets[twoak8Pass]
        events = events[twoak8Pass]
        evtweights = evtweights[twoak8Pass]
        if not isData:
            GenJets = GenJets[twoak8Pass]
        output['cutflow']['>= 2 AK8 Jets'] += len(FatJets)
        del twoak8Pass

        # event cuts #
        
        # save cutflow
#         if isNominal:
#             cuts = []
#             for cut in selection.names:
#                 cuts.append(cut)
#                 output['cutflow'][cut] += len(FatJets[selection.all(*cuts)])
#             del cuts
        
#         eventCut = selection.all(*selection.names)
#         FatJets = FatJets[eventCut]
#         SubJets = SubJets[eventCut]
#         Jets    = Jets[eventCut]
#         evtweights = evtweights[eventCut]
#         events = events[eventCut]

#         if not isData: GenJets = GenJets[eventCut]
            

        # ---- ttbar candidates ---- #
        
        # index = [[0], [1], [0], ... [0], [1], [1]] type='{# events} * var * int64'
        index = ak.unflatten( np.random.RandomState(random.seed()).randint(2, size=len(FatJets)), np.ones(len(FatJets), dtype='i'))
        
        jet0 = FatJets[index]
        jet1 = FatJets[1 - index]        
        ttbarcands = ak.cartesian([jet0, jet1])
        del index
        
        
        
        # ttbar event cuts  #
        
        # at least 1 ttbar candidate #
        oneTTbarPass = (ak.num(ttbarcands) >= 1)
        ttbarcands = ttbarcands[oneTTbarPass]
        FatJets = FatJets[oneTTbarPass]
        Jets = Jets[oneTTbarPass]
        SubJets = SubJets[oneTTbarPass]
        events = events[oneTTbarPass]
        evtweights = evtweights[oneTTbarPass]
        if not isData:
            GenJets = GenJets[oneTTbarPass]
        output['cutflow']['>= one TTbar'] += len(FatJets)
        del oneTTbarPass
        
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        dPhiCutPass = ak.flatten(np.abs(ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4)) > 2.1)  
        ttbarcands = ttbarcands[dPhiCutPass]
        FatJets = FatJets[dPhiCutPass] 
        Jets = Jets[dPhiCutPass]
        SubJets = SubJets[dPhiCutPass] 
        events = events[dPhiCutPass]
        evtweights = evtweights[dPhiCutPass]
        if not isData:
            GenJets = GenJets[dPhiCutPass]
        output['cutflow']['Passed dPhi Cut'] += len(FatJets)
        del dPhiCutPass
        
        # ttbar candidates have 2 subjets #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1))
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1))
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1)))
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        FatJets = FatJets[GoodSubjets]
        SubJets = SubJets[GoodSubjets]
        events = events[GoodSubjets]
        Jets = Jets[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        if not isData:
            GenJets = GenJets[GoodSubjets]
        output['cutflow']['Good Subjets'] += len(FatJets)
        del GoodSubjets, hasSubjets0, hasSubjets1
        
        # apply ttbar event cuts #
#         if correction == 'nominal':
#             output['cutflow']['oneTTbar'] += len(FatJets[oneTTbarPass])
#             output['cutflow']['dPhiCut'] += len(FatJets[(oneTTbarPass & dPhiCutPass)])
#             output['cutflow']['Good Subjets'] += len(FatJets[(oneTTbarPass & dPhiCutPass & GoodSubjets)])

#         ttbarcandCuts = (oneTTbar & dPhiCut & GoodSubjets)
#         ttbarcands = ttbarcands[ttbarcandCuts]
#         FatJets = FatJets[ttbarcandCuts]
#         Jets = Jets[ttbarcandCuts]
#         SubJets = SubJets[ttbarcandCuts]
#         events = events[ttbarcandCuts]
#         evtweights = evtweights[ttbarcandCuts]
        
#         if not isData: GenJets = GenJets[ttbarcandCuts]
#         del oneTTbar, dPhiCut, ttbarcandCuts, hasSubjets0, hasSubjets1, GoodSubjets
        

        # ttbarmass
        ttbarmass = (ttbarcands.slot0.p4 + ttbarcands.slot1.p4).mass
        
        # subjets
        SubJet00 = SubJets[ttbarcands.slot0.subJetIdx1]
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx2]
        SubJet10 = SubJets[ttbarcands.slot1.subJetIdx1]
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx2]
        
        
        
        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        if self.useDeepAK8:
            ttag_s0_disc = ttbarcands.slot0.deepTagMD_TvsQCD > self.deepAK8Cut
            ttag_s1_disc = ttbarcands.slot1.deepTagMD_TvsQCD > self.deepAK8Cut
            antitag_disc = (ttbarcands.slot0.deepTagMD_TvsQCD < self.deepAK8Cut) & (ttbarcands.slot0.deepTagMD_TvsQCD > 0.2)
            
            mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
            mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

            ttag_s0 = (ttag_s0_disc) #& (mcut_s0)
            ttag_s1 = (ttag_s1_disc) #& (mcut_s1)
            antitag = (antitag_disc) #& (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

            
        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        else:
            tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
            tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

            taucut_s0 = tau32_s0 < self.tau32Cut
            taucut_s1 = tau32_s1 < self.tau32Cut

            mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
            mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

            ttag_s0 = (taucut_s0) & (mcut_s0)
            ttag_s1 = (taucut_s1) & (mcut_s1)
            antitag = (~taucut_s0) & (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

            if self.Sideband:
                mcut_s0 = (40. < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < 105.) # Side Band Window for Closure Test
                mcut_s1 = (40. < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < 105.) 
                
                ttag_s0 = (taucut_s0) & (mcut_s0) # Side Band Window for Closure Test
                ttag_s1 = (taucut_s1) & (mcut_s1)

                antitag = (~taucut_s0) & (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)
        
        
        # tau32 cuts for plotting
        tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
        tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )

        taucut_s0 = tau32_s0 < self.tau32Cut
        taucut_s1 = tau32_s1 < self.tau32Cut
        
        
        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (AT&Pt)
        pretag =  ttag_s0 # Only jet0 (pret)
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttag1 =   ttag_s0 ^ ttag_s1 # Exclusively one top tagged (1t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_s0 & ttag_s1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
                         
        
        
        # b tagger #
        
        bdisc_s0 = np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB)
        bdisc_s1 = np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB)
        # bdisc_s0 = np.maximum(SubJet00.btagCSVV2 , SubJet01.btagCSVV2)
        # bdisc_s1 = np.maximum(SubJet10.btagCSVV2 , SubJet11.btagCSVV2)
        tdisc_s0 = ttbarcands.slot0.deepTagMD_TvsQCD
        tdisc_s1 = ttbarcands.slot1.deepTagMD_TvsQCD

        
        btag_s0 = ( np.maximum(SubJet00.btagDeepB , SubJet01.btagDeepB) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet10.btagDeepB , SubJet11.btagDeepB) > self.bdisc )
        # btag_s0 = ( np.maximum(SubJet00.btagCSVV2 , SubJet01.btagCSVV2) > self.bdisc )
        # btag_s1 = ( np.maximum(SubJet10.btagCSVV2 , SubJet11.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
        # rapidity #
        cen = np.abs(getRapidity(ttbarcands.slot0.p4) - getRapidity(ttbarcands.slot1.p4)) < 1.0
        fwd = (~cen)
    
    
        # rapidity, btag and top tag categories
        regs = {'cen': cen, 'fwd': fwd}
        btags = {'0b': btag0, '1b':btag1, '2b':btag2}
        ttags = {"AT&Pt": antitag_probe, 
                 "at":antitag, 
                 "pret":pretag, 
                 "0t":ttag0, 
                 "1t":ttag1, 
                 ">=1t":ttagI, 
                 "2t":ttag2,
                 ">=0t":Alltags
                }
        
        
        # get all analysis category masks
        categories = { t[0]+b[0]+y[0] : (t[1]&b[1]&y[1])  for t,b,y in itertools.product( ttags.items(), 
                                                                        btags.items(), 
                                                                        regs.items())
            }
        
        # use subset of analysis category masks from ttbaranalysis.py
        labels_and_categories = {label:categories[label] for label in self.anacats}
    
        
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
        jetmsd = ttbarcands.slot0.msoftdrop

        
        
        # event weights #
        
        # if few events
        if (len(evtweights) < 10): return output
        
        weights = Weights(len(evtweights))
        weights.add('genWeight', evtweights)
                        
        # if running background estimation
        if (self.bkgEst):

            # for mistag rate weights
            mistag_rate_df = pd.read_csv('data/corrections/backgroundEstimate/mistag_rate_'+self.iov+'_inc.csv')
            pbins = mistag_rate_df['jetp bins'].values
            mistag_weights = np.ones(len(FatJets), dtype=float) # initialization

            if 'QCD' in dataset:
                mistag_rate_df = pd.read_csv('data/corrections/backgroundEstimate/QCD_mistag_rate_'+self.iov+'_inc.csv')
            
            
            # for mass modification
            if self.MassMod:
                
            	qcd_jetmass_dict = json.load(open('data/corrections/backgroundEstimate/QCD_jetmass_'+self.iov+'.json'))
            	qcd_jetmass_bins = qcd_jetmass_dict['bins']
    
            for ilabel,icat in labels_and_categories.items():
            
                icat = ak.flatten(icat)

                # get antitag region and signal region labels
                # ilabel[-5:] = bcat + ycat (0bcen for example)
                label_at = 'at'+ilabel[-5:]
                label_2t = '2t'+ilabel[-5:]
                label_inc = ilabel[-5:-3]
#                 print(label_inc)
                
                
                # get mistag rate for antitag region
                #print(mistag_rate_df[label_at])
                mistag_rate = mistag_rate_df[label_inc].values
                #print(mistag_rate)
                
                # get p bin for probe jet p
                mistag_pbin = np.digitize(ak.flatten(jetp[icat]), pbins) - 1

                # store mistag weights for events in this category
                mistag_weights[icat] = mistag_rate[mistag_pbin]



                # qcd mass modification #
                if self.MassMod:
                    
                    # get distribution of jet mass in QCD signal ('2t') region
                    qcd_jetmass_counts = qcd_jetmass_dict[label_2t]
                    
                    # randomly select jet mass from distribution
                    ModMass_hist_dist = ss.rv_histogram([qcd_jetmass_counts[:-1], qcd_jetmass_bins])
                    ttbarcands.slot1.p4[icat]["fMass"] = ModMass_hist_dist.rvs(size=len(ttbarcands.slot1.p4[icat]))
                
                
            weights.add('mistag', mistag_weights)
    
        del jetmass, jetp, jetmsd
        
        jetpt = ttbarcands.slot1.p4.pt
        jeteta = ttbarcands.slot1.p4.eta
        jetphi = ttbarcands.slot1.p4.phi
        jetmass = ttbarcands.slot1.p4.mass
        jetp = ttbarcands.slot1.p4.p
        
        # plot same jetmass as pre-tagged, anti-tagged jet
        jetmsd = ttbarcands.slot0.msoftdrop
           
        
        # values for mistag rate calculation #
        numerator = np.where(antitag_probe, ttbarcands.slot1.p4.p, -1)
        denominator = np.where(antitag, ttbarcands.slot1.p4.p, -1)
        
        # pt reweighting #
#         if ('TTbar' in dataset):
#             ttbar_wgt = pTReweighting(ttbarcands.slot0.pt, ttbarcands.slot1.pt)
#             weights.add('ptReweighting', ak.flatten(ttbar_wgt))
                 
        if not self.noSyst and not isData:
                    
            if 'pileup' in self.systematics:
                
                puNom, puUp, puDown = GetPUSF(events, self.iov)
                weights.add("pileup", 
                    weight=puNom, 
                    weightUp=puUp, 
                    weightDown=puDown,
                           )

            if ('prefiring' in self.systematics) and ("L1PreFiringWeight" in events.fields):
                if ('2016' in self.iov) or ('2017' in self.iov):
                
                    prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events)
                    weights.add("prefiring", 
                        weight=prefiringNom, 
                        weightUp=prefiringUp, 
                        weightDown=prefiringDown,
                               )
                    
            if 'pdf' in self.systematics:
                
                pdfUp, pdfDown, pdfNom = GetPDFWeights(events)
                weights.add("pdf", 
                    weight=pdfNom, 
                    weightUp=pdfUp, 
                    weightDown=pdfDown,
                           )    
            
            if 'q2' in self.systematics:
                
                q2Nom, q2Up, q2Down = GetQ2weights(events)
                
                weights.add("q2", 
                    weight=q2Nom, 
                    weightUp=q2Up, 
                    weightDown=q2Down,
                           )   
                
            if 'btag' in self.systematics:
                
                btag_wgts_nom = np.ones(len(events))
                btag_wgts_up  = np.ones(len(events))
                btag_wgts_down = np.ones(len(events))
                
                btag_wgts_nom_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='central')
                
                btag_wgts_up_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='up')
                
                btag_wgts_down_bcats = btagCorrections([btag0, btag1, btag2], 
                                                      [SubJet00, SubJet01, SubJet10, SubJet11], 
                                                      isData, 
                                                      self.bdisc,
                                                      sysType='down')
                
                
                btag_wgts_nom[ak.flatten(btag0)]  = btag_wgts_nom_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_up[ak.flatten(btag0)]   = btag_wgts_up_bcats['0b'][ak.flatten(btag0)]
                btag_wgts_down[ak.flatten(btag0)] = btag_wgts_down_bcats['0b'][ak.flatten(btag0)]
                # print('\t\t ---- AFTER btag0 CUT ----')
                # print('up: ', btag_wgts_up)
                # print('nom: ', btag_wgts_nom)
                # print('down: ', btag_wgts_down)
                # print('\t\t ---- BEFORE btag1 CUT ----')
                btag_wgts_nom[ak.flatten(btag1)]  = btag_wgts_nom_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_up[ak.flatten(btag1)]   = btag_wgts_up_bcats['1b'][ak.flatten(btag1)]
                btag_wgts_down[ak.flatten(btag1)] = btag_wgts_down_bcats['1b'][ak.flatten(btag1)]
                # print('\t\t ---- AFTER btag1 CUT ----')
                # print('up: ', btag_wgts_up)
                # print('nom: ', btag_wgts_nom)
                # print('down: ', btag_wgts_down)
                # print('\t\t ---- BEFORE btag2 CUT ----')
                btag_wgts_nom[ak.flatten(btag2)]  = btag_wgts_nom_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_up[ak.flatten(btag2)]   = btag_wgts_up_bcats['2b'][ak.flatten(btag2)]
                btag_wgts_down[ak.flatten(btag2)] = btag_wgts_down_bcats['2b'][ak.flatten(btag2)]
                # print('\t\t ---- AFTER btag2 CUT ----')
                # print('up: ', btag_wgts_up)
                # print('nom: ', btag_wgts_nom)
                # print('down: ', btag_wgts_down)
                # print('\t\t ---- BEFORE INCLUDING WEIGHTS ----')
                
                # print('\t\t ---- FIRST LINE ----')
                # print('up: ', btag_wgts_up_bcats['0b'][ak.flatten(btag0)])
                # print('nom: ', btag_wgts_nom_bcats['0b'][ak.flatten(btag0)])
                # print('down: ', btag_wgts_down_bcats['0b'][ak.flatten(btag0)])
                # print('\t\t ---- LAST LINE ----')
                
                weights.add("btag", 
                    weight=btag_wgts_nom, 
                    weightUp=btag_wgts_up, 
                    weightDown=btag_wgts_down,
                           )
                
                del btag_wgts_nom, btag_wgts_up, btag_wgts_down
                del btag_wgts_nom_bcats, btag_wgts_up_bcats, btag_wgts_down_bcats
                
                
            if 'toptagsf' in self.systematics and 'TTbar' in dataset:
                
                toptagNom  = np.ones(len(events))
                toptagUp   = np.ones(len(events))
                toptagDown = np.ones(len(events))
                
                toptagNom, toptagUp, toptagDown = GetTopTagSFweights([ttag1, ttag2])
                
                # print('\t\t ---- FIRST LINE ----')
                # print('up: ', ak.flatten(toptagUp))
                # print('nom: ', ak.flatten(toptagNom))
                # print('down: ', ak.flatten(toptagDown))
                
                
                weights.add("toptagsf",
                            weight=ak.flatten(toptagNom),
                            weightUp=ak.flatten(toptagUp),
                            weightDown=ak.flatten(toptagDown),
                    )
                # print('\t\t ---- LAST LINE ----')
            
            if 'toptagxs' in self.systematics and 'TTbar' in dataset:
                
                weights.add("toptagxs",
                            weight=ak.Array(np.ones(len(events))), # XS already applied to plots after processing
                            weightUp=ak.Array(np.full(len(events), 1.08)),
                            weightDown=ak.Array(np.full(len(events), 0.92))
                    )
            
            if 'lumi' in self.systematics:
                
                lumiUp, lumiDown = GetLumiweights(self.iov)

                # print('\t\t ---- FIRST LINE ----')
                # print('up: ', np.full(len(events), lumiUp))
                # print('nom: ', np.ones(len(events)))
                # print('down: ', np.full(len(events), lumiDown))
                weights.add("lumi",
                            weight=ak.Array(np.ones(len(events))), # Luminosity already applied to plots after processing
                            weightUp=ak.Array(np.full(len(events), lumiUp)),
                            weightDown=ak.Array(np.full(len(events), lumiDown)),
                    )
                # print('\t\t ---- LAST LINE ----')     
                



        for i, [ilabel,icat] in enumerate(labels_and_categories.items()):
        
            icat = ak.flatten(icat)
            
            output['cutflow'][ilabel] += np.sum(icat)
                
            if correction == 'nominal':                    
                output['numerator'].fill(anacat = i,
                                         jetp = ak.flatten(numerator[icat]),
                                         weight = weights.weight()[icat],
                                        )

                output['denominator'].fill(anacat = i,
                                           jetp = ak.flatten(denominator[icat]),
                                           weight = weights.weight()[icat],
                                        )
                output['jetp'].fill(anacat = i,
                                   jetp = ak.flatten(jetp[icat]),
                                   weight = weights.weight()[icat],
                                        )
                output['jetmass'].fill(anacat = i,
                                   jetmass = ak.flatten(jetmass[icat]),
                                   weight = weights.weight()[icat],
                                  )
#                 output['sdjetmass'].fill(anacat = i,
#                                    sdjetmass = ak.flatten(jetmsd[icat]),
#                                    weight = weights.weight()[icat],
#                                   )
                output['jetpt'].fill(anacat = i,
                                     jetpt = ak.flatten(jetpt[icat]),
                                     weight = weights.weight()[icat],
                                      )

                output['jeteta'].fill(anacat = i,
                                      jeteta = ak.flatten(jeteta[icat]),
                                      weight = weights.weight()[icat],
                                      )
                output['jetphi'].fill(anacat = i,
                                      jetphi = ak.flatten(jetphi[icat]),
                                      weight = weights.weight()[icat],
                                      )
                output['discriminators'].fill(#systematic=correction,
                                          anacat = i,
                                          jetp = ak.flatten(jetp[icat]),
                                          bdisc = ak.flatten(bdisc_s1[icat]),
                                          tdisc = ak.flatten(tdisc_s1[icat]),
                                          nsub = ak.flatten(tau32_s1)[icat],
                                          weight = weights.weight()[icat],
                                         )
            
            output['ttbarmass'].fill(systematic=correction,
                                     anacat = i,
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = weights.weight()[icat],
                                    )
            output['ttbarmass_fine'].fill(systematic=correction,
                                     anacat = i,
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                     weight = weights.weight()[icat],
                                    )
            output['ttbarmass_bare'].fill(systematic=correction,
                                     anacat = i,
                                     ttbarmass = ak.flatten(ttbarmass[icat]),
                                    )
            
            
            
            # output['mtt_vs_mt'].fill(systematic=correction,
            #                          anacat = i,
            #                          jetmass = ak.flatten(jetmsd[icat]),
            #                          ttbarmass = ak.flatten(ttbarmass[icat]),
            #                          weight = weights.weight()[icat],
            #                         )
            
            

#             output['deepak8'].fill(anacat = i,
#                                    jetp = ak.flatten(jetp[icat]),
#                                    ttbarmass = ak.flatten(ttbarmass[icat]),
#                                    tdisc = ak.flatten(tdisc_s1[icat]),
#                                    weight = weights.weight()[icat],
#                                   )
            
            
            
            
            # save weights
            
            output['weights'][correction] += np.sum(weights.weight())


                
            if not 'jes' in correction and not 'jer' in correction:    
                

                for syst in weights.variations:
                    
                    
                    output['weights'][syst] += np.sum(weights.weight(syst))

                    output['ttbarmass'].fill(systematic=syst,
                                         anacat = i,
                                         ttbarmass = ak.flatten(ttbarmass[icat]),
                                         weight = weights.weight(syst)[icat],
                                        )

                    # output['mtt_vs_mt'].fill(systematic=syst,
                    #                      anacat = i,
                    #                      ttbarmass = ak.flatten(ttbarmass[icat]),
                    #                      jetmass = ak.flatten(jetmsd[icat]),
                    #                      weight = weights.weight(syst)[icat],
                    #                     )

        
        


        return output

    def postprocess(self, accumulator):
        return accumulator
        
        
        
