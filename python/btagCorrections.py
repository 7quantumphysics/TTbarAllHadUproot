import numpy as np
import awkward as ak
import correctionlib

def plotratio2d(numerator, denominator, ax=None, cmap='Blues', cbar=True):
    NumeratorAxes = numerator.axes
    DenominatorAxes = denominator.axes
    
    # integer number of bins in this axis #
    NumeratorAxis1_BinNumber = NumeratorAxes[0].size - 3 # Subtract 3 to remove overflow
    NumeratorAxis2_BinNumber = NumeratorAxes[1].size - 3
    
    DenominatorAxis1_BinNumber = DenominatorAxes[0].size - 3 
    DenominatorAxis2_BinNumber = DenominatorAxes[1].size - 3 
    
    if(NumeratorAxis1_BinNumber != DenominatorAxis1_BinNumber 
       or NumeratorAxis2_BinNumber != DenominatorAxis2_BinNumber):
        raise Exception('Numerator and Denominator axes are different sizes; Cannot perform division.')
    # else:
    #     Numerator = numerator.to_hist()
    #     Denominator = denominator.to_hist()
        
    ratio = numerator / denominator.values()

    return hep.hist2dplot(ratio, ax=ax, cmap=cmap, norm=colors.Normalize(0.,1.), cbar=cbar)

def BtagUpdater(subjet, Eff_filename_list, ScaleFactorFilename, FittingPoint, OperatingPoint, bdisc):  
    """
    subjet (Flattened Awkward Array)       ---> One of the Four preselected subjet awkward arrays (e.g. SubJet01)
    Eff_filename_list (Array of strings)   ---> List of imported b-tagging efficiency files of the selected subjet (corresponding to the hadron flavour of the subjet)
    ScaleFactorFilename (string)           ---> CSV file containing info to evaluate scale factors with
    FittingPoint (string)                  ---> "loose"  , "medium", "tight"
    OperatingPoint (string)                ---> "central", "up"    , "down"
    """
    # ---- Declare flattened pT and Eta variables ---- #
    pT = np.asarray(ak.flatten(subjet.p4.pt))
    Eta = np.asarray(ak.flatten(subjet.p4.eta))

    # ---- Import Flavor Efficiency Tables as Dataframes ---- #
    subjet_flav_index = np.arange(ak.to_numpy(subjet.hadronFlavour).size)
    df_list = [ pd.read_csv(Eff_filename_list[i]) for i in subjet_flav_index ] # List of efficiency dataframes; imported to extract list of eff_vals
    # print(df_list[0])
    eff_vals_list = [ df_list[i]['efficiency'].values for i in subjet_flav_index ] # efficiency values for each file read in; one file per element of subjet array
    # print(eff_vals_list[0])

    # ---- Match subjet pt and eta to appropriate bins ---- #
    pt_BinKeys = np.arange(np.array(manual_subjetpt_bins).size - 1) # the -1 ensures proper size for bin labeling
    eta_BinKeys = np.arange(np.array(manual_subjeteta_bins).size - 1) # the -1 ensures proper size for bin labeling
    pt_Bins = np.array(manual_subjetpt_bins)
    eta_Bins = np.array(manual_subjeteta_bins)

    # ---- Usable pt and eta bin indices ---- #
    pt_indices = np.digitize(pT, pt_Bins, right=True) - 1 # minus one because digitize labels first element as 1 instead of 0
    eta_indices = np.digitize(Eta, eta_Bins, right=True) - 1

    pt_indices = np.where(pt_indices == pt_BinKeys.size, pt_indices-1, pt_indices) # if value is larger than largest bin, bin number will be defaulted to largest bin
    eta_indices = np.where(eta_indices == eta_BinKeys.size, eta_indices-1, eta_indices)

    pt_indices = np.where(pt_indices < 0, 0, pt_indices) # if value is less than smallest bin, bin number will be defaulted to smallest bin (zeroth)
    eta_indices = np.where(eta_indices < 0, 0, eta_indices)

    # ---- Pair the indices together ---- #
    index_pairs = np.vstack((pt_indices, eta_indices)).T  # Pairs of pt and eta bin indices to be mapped to corresponding efficiency bin number
    index_pairs_tuples = [tuple(e) for e in index_pairs] # This can be indexed easily for reading from dictionary

    # ---- Get Efficiencies from  ---- #
    eff_BinKeys_comb = CartesianProduct(pt_BinKeys, eta_BinKeys) #List of Combined pt and eta keys (should be 40 of them)
    effBinKeys = np.arange( len(eff_BinKeys_comb) )
    EffKeys_Dict = dict(zip([tuple(eff_BinKeys_comb[i]) for i in effBinKeys], effBinKeys)) # Mapping combined pt and eta keys to a single integer (for boradcasting)
    Eff_indices = [EffKeys_Dict[index_pairs_tuples[i]] for i in range(pt_indices.size)] # Indices for selecting efficiency values from the lists for each subjet index

    eff_val = np.asarray([ eff_vals_list[i][Eff_indices[i]] for i in subjet_flav_index ])
    # print(eff_val)

    """
                                !! NOTE !!
            Some efficiency values (eff_val array elements) may be zero
            and must be taken into account when dividing by the efficiency
    """

    ###############  Btag Update Method ##################
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
    #https://github.com/rappoccio/usercode/blob/Dev_53x/EDSHyFT/plugins/BTagSFUtil_tprime.h

    coin = np.random.uniform(0,1,len(subjet)) # used for randomly deciding which jets' btag status to update or not
    subjet_btag_status = np.asarray((subjet.btagCSVV2 > bdisc)) # do subjets pass the btagger requirement

    '''
*******************************************************************************************************************
                Correction Library Logic for Applying Subjet Scale Factors
                -----------------------------------------------------------
        1.) Declare CorrectionSet object by importing the desired JSON file
                CorrectionlibObject = correctionlib.CorrectionSet.from_file(<name and/or path of file (.json.gz)>)
        2.) Flatten the subjet's flavor, pt and eta arrays
        3.) Convert these arrays to Numpy arrays (as Correctionlib hates Awkward Arrays)
        4.) Split flavor array into two arrays of the same length:
            i.) Pretend the entire array were only heavy quarks (b and c) by replacing light quraks with c quarks
                [4 4 0 5 4 0 0 5 5] ---> [4 4 4 5 4 4 4 5 5]
            ii.) Pretend the entire array were only light quarks by replacing b and c with light quarks
                [4 4 0 5 4 0 0 5 5] ---> [0 0 0 0 0 0 0 0 0]
        5.) Check the "name" of the tagging corrections at the beginning of the JSON file
                {
                  "schema_version": 2,
                  "description": "This json file contains the corrections for deepCSV subjet tagging. ",
                  "corrections": [
                    {
                      "name": "deepCSV_subjet",
        6.) Fill in the required "inputs" in the order shown in the JSON file
                "inputs": [
                    {
                      "name": "systematic",
                      "type": "string"
                    },
                    {
                      "name": "method",
                      "type": "string",
                      "description": "incl for light jets, lt for b/c jets"
                    },
                    {
                      "name": "working_point",
                      "type": "string",
                      "description": "L/M"
                    },
                    {
                      "name": "flavor",
                      "type": "int",
                      "description": "hadron flavor definition: 5=b, 4=c, 0=udsg"
                    },
                    {
                      "name": "abseta",
                      "type": "real"
                    },
                    {
                      "name": "pt",
                      "type": "real"
                    }
                  ],
        7.) Create one Scale Factor array by comparing both 'pretend' arrays with the original flavor array
            if flavor array is 0, use scale factors evaluated with 'pretend' light quark array
            otherwise, use scale factors evaluated with 'pretend' heavy quark array
                For Original Flavor Array [4 4 0 5 4 0 0 5 5]:
                0 ---> use scale factor element made from [0 0 0 0 0 0 0 0 0]
                4 ---> use scale factor element made from [4 4 4 5 4 4 4 5 5]
                5 ---> use scale factor element made from [4 4 4 5 4 4 4 5 5]

*******************************************************************************************************************
    ''' 
    # Step 1.)
    btag_sf = correctionlib.CorrectionSet.from_file(ScaleFactorFilename)
    # Step 2.) and 3.)
    hadronFlavour = ak.to_numpy(ak.flatten(subjet.hadronFlavour))
    eta = ak.to_numpy(ak.flatten(subjet.eta))
    pt = ak.to_numpy(ak.flatten(subjet.pt))
    # ---- Ensure eta and pt fall within the allowed binning for corrections ---- #
    Min_etaval = 0.
    Max_etaval = 2.5
    Min_ptval = 30.
    Max_ptval = 450.

    eta = np.where(abs(eta)>=Max_etaval, Max_etaval-0.500, eta)
    pt = np.where(pt<=Min_ptval, Min_ptval+1.00, pt)
    pt = np.where(pt>=Max_ptval, Max_ptval-1.00, pt)

    # Step 4.)
    allHeavy = np.where(hadronFlavour == 0, 4, hadronFlavour)
    allLight = np.zeros_like(allHeavy) 
    # Step 5.) and 6.)
    BSF_allHeavy = btag_sf['deepCSV_subjet'].evaluate(OperatingPoint, 'lt', FittingPoint, allHeavy, abs(eta), pt)
    BSF_allLight = btag_sf['deepCSV_subjet'].evaluate(OperatingPoint, 'incl', FittingPoint, allLight, abs(eta), pt)
    # Step 7.)
    BSF = np.where(hadronFlavour == 0, BSF_allLight, BSF_allHeavy) # btag scale factors
    # print(BSF)

    """
*******************************************************************************************************************        
                    Does the Subjet Pass the Discriminator Cut?
                   ---------------------------------------------
                  True                                      False
                  ----                                      -----
    | SF = 1  |  SF < 1  |  SF > 1  |         |  SF = 1  |  SF < 1  |  SF > 1  |

    |    O    |Downgrade?|    O     |         |     X    |    X     | Upgrade? |
               ----------                                             --------
    |         |True|False|          |         |          |          |True|False|
                ---  ---                                              ---  ---
    |         |  X |  O  |          |         |          |          |  O |  X  |

    --------------------------------------------------------------------------------

    KEY:
         O ---> btagged subjet     (boolean 'value' = True)
         X ---> non btagged subjet (boolean 'value' = False)

    Track all conditions where elements of 'btag_update' will be true (4 conditions marked with 'O')
*******************************************************************************************************************        
    """ 

    f_less = abs(1. - BSF) # fraction of subjets to be downgraded
    f_greater = np.where(eff_val > 0., abs(f_less/(1. - 1./eff_val)), 0.) # fraction of subjets to be upgraded  

    condition1 = (ak.flatten(subjet_btag_status) == True) & (BSF == 1.)
    condition2 = (ak.flatten(subjet_btag_status) == True) & ((BSF < 1.0) & (coin < BSF)) 
    condition3 = (ak.flatten(subjet_btag_status) == True) & (BSF > 1.)
    condition4 = (ak.flatten(subjet_btag_status) == False) & ((BSF > 1.) & (coin < f_greater))

    subjet_new_btag_status = np.where((condition1 ^ condition2) ^ (condition3 ^ condition4), True, False)   

    return subjet_new_btag_status

def GetFlavorEfficiency(Subjet, Flavor): # Return "Flavor" efficiency numerator and denominator
    '''
    Subjet --> awkward array object after preselection i.e. SubJetXY
    Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
    '''
    # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
    pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
    eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
    flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark

    subjet_btagged = (Subjet.btagCSVV2 > bdisc)

    Eff_Num_pT = np.where(subjet_btagged & (flav == Flavor), pT, -1) # if not collecting pT of subjet, then put non exisitent bin, i.e. -1
    Eff_Num_eta = np.where(subjet_btagged & (flav == Flavor), eta, -1) # if not collecting eta of subjet, then put non exisitent bin, i.e. 5

    Eff_Num_pT = ak.flatten(Eff_Num_pT) # extra step needed for numerator to gaurantee proper shape for filling hists
    Eff_Num_eta = ak.flatten(Eff_Num_eta)

    Eff_Denom_pT = np.where(flav == Flavor, pT, -1)
    Eff_Denom_eta = np.where(flav == Flavor, eta, -1)

    EffStuff = {
        'Num_pT' : Eff_Num_pT,
        'Num_eta' : Eff_Num_eta,
        'Denom_pT' : Eff_Denom_pT,
        'Denom_eta' : Eff_Denom_eta,
    }

    return EffStuff

def FlavEffList(Flavor, Output, Dataset, bdiscDirectory, Save):
    """
    Flavor          ---> string: either 'b', 'c', or 'udsg'
    Output          ---> Coffea Object: Output that is returned from running processor
    Dataset         ---> string: the dataset string (ex QCD, RSGluon1000, etc...) corresponding to Output
    bdiscDirectory  ---> string; Directory path for chosen b discriminator
    Save            ---> bool; Save mistag rates or not
    """
    SaveDirectory = maindirectory + '/FlavorTagEfficiencies/' + bdiscDirectory + Flavor + 'tagEfficiencyTables/'
    mkdir_p(SaveDirectory)
    for subjet in ['s01', 's02', 's11', 's12']:

        eff_numerator = Output[Flavor + '_eff_numerator_' + subjet + '_manualbins'][{'dataset': Dataset}]
        eff_denominator = Output[Flavor + '_eff_denominator_' + subjet + '_manualbins'][{'dataset': Dataset}]

        eff = plotratio2d(eff_numerator, eff_denominator) #ColormeshArtists object
        
        eff_data = eff[0].get_array().data # This is what goes into pandas dataframe
        eff_data = np.nan_to_num(eff_data, nan=0.0) # If eff bin is empty, call it zero

        # ---- Define pt and eta bins from the numerator or denominator hist objects ---- #
        pt_bins = []
        eta_bins = []

        for iden in eff_numerator.axes['subjetpt']:
            pt_bins.append(iden)
        for iden in eff_numerator.axes['subjeteta']:
            eta_bins.append(iden)

        # ---- Define the Efficiency List as a Pandas Dataframe ---- #
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        EfficiencyList = pd.DataFrame(
                            eff_data,
                            pd.MultiIndex.from_product( [pt_bins, eta_bins], names=['pt', 'eta'] ),
                            ['efficiency']
                        )

        print('\n\t--------------------- Subjet ' + subjet + ' ' + Flavor + ' Efficiency ---------------------\n', flush=True)
        print('====================================================================\n', flush=True)
        print(EfficiencyList, flush=True)
        
        # ---- Save the Efficiency List as .csv ---- #
        if Save:
            filename = dataset + '_' + subjet + '_' + Flavor + 'tageff.csv'
            EfficiencyList.to_csv(SaveDirectory+filename)
            print('\nSaved ' + filename, flush=True)


def btagCorrections(btags, subjets, isData, bdisc, sysType='central'):
    
    
    btag0, btag1, btag2 = btags
    SubJet01, SubJet02, SubJet11, SubJet12 = subjets
    
    
    btag_s0 = ( np.maximum(SubJet01.btagDeepB , SubJet02.btagDeepB) > bdisc )
    btag_s1 = ( np.maximum(SubJet11.btagDeepB , SubJet12.btagDeepB) > bdisc )
    
    Btag_wgts = {} # To be filled with "btag_wgts" corrections below (Needs to be defined for higher scope)
    
    if not isData:


        # **************************************************************************************** #
        # --------------------------- Method 1c) Apply Event Weights ----------------------------- #
        # -------------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods -------------- #
        # **************************************************************************************** #

        # ---- Temporarily define the 'outline' of the collection of weights to calculate ---- # 
        btag_wgts = {'0b':np.array([None]),
                     '1b':np.array([None, None]),
                     '2b':np.array([None, None, None])}

        """
        ******************************************************************************************************
        btag_wgts['mb'][n] --> w(n|m) --> "Probability" of n number of b-tags given m number of "true" b jets
        ------------------------------------------------------------------------------------------------------
        w(0|0) = 1

        w(0|1), w(1|1) = 1 - BSF, 
                       = BSF

        w(0|2), w(1|2), w(2|2) = (1 - BSF_s0)(1 - BSF_s1), 
                               = (1 - BSF_s0)BSF_s1 + BSF_s0(1 - BSF_s1),
                               = (BSF_s0)(BSF_s1)

        w(1|0), w(2|0), w(2|1) = Undef.
        ******************************************************************************************************
        """

        # ---- Use the leading subjet again to get the scale factors ---- #
        
        LeadingSubjet_s0 = np.where(SubJet01.btagDeepB>SubJet02.btagDeepB, SubJet01, SubJet02)
        LeadingSubjet_s1 = np.where(SubJet11.btagDeepB>SubJet12.btagDeepB, SubJet11, SubJet12)
     
        # ---- Define the BSF for each of the two fatjets ---- #
        SF_filename = 'data/corrections/subjet_btagging.json.gz'
        Fitting = "M"
        if bdisc < 0.5:
            Fitting = "L"

        btag_sf = correctionlib.CorrectionSet.from_file(SF_filename)

        s0_hadronFlavour = ak.to_numpy(ak.flatten(LeadingSubjet_s0.hadronFlavour))
        s1_hadronFlavour = ak.to_numpy(ak.flatten(LeadingSubjet_s1.hadronFlavour))

        s0_eta = ak.to_numpy(ak.flatten(LeadingSubjet_s0.eta))
        s1_eta = ak.to_numpy(ak.flatten(LeadingSubjet_s1.eta))

        s0_pt = ak.to_numpy(ak.flatten(LeadingSubjet_s0.pt))
        s1_pt = ak.to_numpy(ak.flatten(LeadingSubjet_s1.pt))

        s0_allHeavy = np.where(s0_hadronFlavour == 0, 4, s0_hadronFlavour)
        s0_allLight = np.zeros_like(s0_allHeavy) 

        s1_allHeavy = np.where(s1_hadronFlavour == 0, 4, s1_hadronFlavour)
        s1_allLight = np.zeros_like(s1_allHeavy) 

        # ---- Ensure eta and pt fall within the allowed binning for corrections ---- #
        Min_etaval = 0.
        Max_etaval = 2.5
        Min_ptval = 30.
        Max_ptval = 450.

        s0_eta = np.where(abs(s0_eta)>=Max_etaval, Max_etaval-0.01, s0_eta)
        s1_eta = np.where(abs(s1_eta)>=Max_etaval, Max_etaval-0.01, s1_eta)

        s0_pt = np.where(abs(s0_pt)<=Min_ptval, Min_ptval+1.00, s0_pt)
        s1_pt = np.where(abs(s1_pt)<=Min_ptval, Min_ptval+1.00, s1_pt)
        s0_pt = np.where(abs(s0_pt)>=Max_ptval, Max_ptval-1.00, s0_pt)
        s1_pt = np.where(abs(s1_pt)>=Max_ptval, Max_ptval-1.00, s1_pt)

        try:
            BSF_s0_allHeavy = btag_sf['deepCSV_subjet'].evaluate(sysType, 'lt', Fitting, s0_allHeavy, abs(s0_eta), s0_pt)
        except RuntimeError as re:
            print('flavor (with light mask): \n', s0_allHeavy, flush=True)
            print('eta: \n', s0_eta, flush=True)
            print('pt: \n', s0_pt, flush=True)
            print('These subjets\' all heavy SFs evaluation failed', flush=True)
            print(re, flush=True)
        try:
            BSF_s1_allHeavy = btag_sf['deepCSV_subjet'].evaluate(sysType, 'lt', Fitting, s1_allHeavy, abs(s1_eta), s1_pt)
        except RuntimeError as RE:
            print('flavor (with light mask): \n', s1_allHeavy, flush=True)
            print('eta: \n', s1_eta, flush=True)
            print('pt: \n', s1_pt, flush=True)
            print('These subjets\' all heavy SFs evaluation failed', flush=True)
            print(RE, flush=True)

        BSF_s0_allLight = btag_sf['deepCSV_subjet'].evaluate(sysType, 'incl', Fitting, s0_allLight, abs(s0_eta), s0_pt)
        BSF_s1_allLight = btag_sf['deepCSV_subjet'].evaluate(sysType, 'incl', Fitting, s1_allLight, abs(s1_eta), s1_pt)

        BSF_s0 = np.where(s0_hadronFlavour == 0, BSF_s0_allLight, BSF_s0_allHeavy)
        BSF_s1 = np.where(s1_hadronFlavour == 0, BSF_s1_allLight, BSF_s1_allHeavy)

        # ---- w(0|0) ---- #
        btag_wgts['0b'][0] = np.where(btag0, np.ones_like(BSF_s0), 0.)

        # ---- w(0|1) and w(1|1) ---- # 
        btag_wgts['1b'][0] = np.where(btag0, np.where(btag_s0, 1.-BSF_s0, 1.-BSF_s1), 0.)
        btag_wgts['1b'][1] = np.where(btag1, np.where(btag_s0, BSF_s0, BSF_s1), 0.)

        # ---- w(0|2), w(1|2), w(2|2) ---- # 
        btag_wgts['2b'][0] = np.where(btag0, (1 - BSF_s0)*(1 - BSF_s1), 0.) 
        btag_wgts['2b'][1] = np.where(btag1, (1 - BSF_s0)*BSF_s1 + BSF_s0*(1 - BSF_s1), 0.) 
        btag_wgts['2b'][2] = np.where(btag2, BSF_s0*BSF_s1, 0.) 

        # ---- 'Matrix Multiplied' weights to apply to each b-tag region ---- #
        Wgts_to_0btag_region = ak.flatten(btag_wgts['0b'][0] + btag_wgts['1b'][0] + btag_wgts['2b'][0])
        Wgts_to_1btag_region = ak.flatten(btag_wgts['1b'][1] + btag_wgts['2b'][1])
        Wgts_to_2btag_region = ak.flatten(btag_wgts['2b'][2])

        # ---- 'Matrix Multiplied' non-zero weights to apply to each b-tag region ---- #
        Wgts_to_0btag_region_nonzero = np.where(Wgts_to_0btag_region==0., 1., Wgts_to_0btag_region)
        Wgts_to_1btag_region_nonzero = np.where(Wgts_to_1btag_region==0., 1., Wgts_to_1btag_region)
        Wgts_to_2btag_region_nonzero = np.where(Wgts_to_2btag_region==0., 1., Wgts_to_2btag_region)

        # ---- Avoid Potential Scope Issues (Unpredictable when/if scope error occurs, so best to avoid it entirely) ---- #
        Btag_wgts['0b'] = Wgts_to_0btag_region_nonzero
        Btag_wgts['1b'] = Wgts_to_1btag_region_nonzero
        Btag_wgts['2b'] = Wgts_to_2btag_region_nonzero
        
        
        # print("Btag_wgts['0b']", Btag_wgts['0b'])
        # print("len Btag_wgts['0b']", len(Btag_wgts['0b']))
        # print("count Btag_wgts['0b']", ak.count(Btag_wgts['0b']))
        # print("len events", len(btag0))
        
        
        # print("Btag_wgts['1b']", Btag_wgts['1b'])
        # print("len Btag_wgts['1b']", len(Btag_wgts['1b']))
        # print("count Btag_wgts['1b']", ak.count(Btag_wgts['1b']))
        # print("len events", len(btag0))
        
        # print("Btag_wgts['2b']", Btag_wgts['2b'])
        # print("len Btag_wgts['2b']", len(Btag_wgts['2b']))
        # print("count Btag_wgts['2b']", ak.count(Btag_wgts['2b']))
        # print("len events", len(btag0))
        


        # Upgrade or Downgrade btag status based on btag efficiency of all four subjets

        # **************************************************************************************** #
        # --------------------------- Method 2a) Update B-tag Status ----------------------------- #
        # -------------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods -------------- #
        # **************************************************************************************** #

                    # ---- Import MC 'flavor' efficiencies ---- #

#         # -- Scale Factor File -- #
#         SF_filename = 'data/corrections/subjet_btagging.json.gz'
#         Fitting = "M"
#         if bdisc < 0.5:
#             Fitting = "L"

#         # -- Get Efficiency .csv Files -- #
#         FlavorTagsDict = {
#             5 : 'btag',
#             4 : 'ctag',
#             0 : 'udsgtag'
#         }

#         SubjetNumDict = {
#             'SubJet01' : [SubJet01, 's01'],
#             'SubJet02' : [SubJet02, 's02'],
#             'SubJet11' : [SubJet11, 's11'],
#             'SubJet12' : [SubJet12, 's12']
#         }

#         EffFileDict = {
#             'Eff_File_s01' : [], # List of eff files corresponding to 1st subjet's flavours
#             'Eff_File_s02' : [], # List of eff files corresponding to 2nd subjet's flavours
#             'Eff_File_s11' : [], # List of eff files corresponding to 3rd subjet's flavours
#             'Eff_File_s12' : []  # List of eff files corresponding to 4th subjet's flavours
#         }

#         for subjet,subjet_info in SubjetNumDict.items():
#             flav_tag_list = [FlavorTagsDict[num] for num in np.abs(ak.flatten(subjet_info[0].hadronFlavour))] # List of tags i.e.) ['btag', 'udsgtag', 'ctag',...]
#             for flav_tag in flav_tag_list:
#                 EffFileDict['Eff_File_'+subjet_info[1]].append('data/FlavorTagEfficiencies/' 
#                                                                + self.BDirect + flav_tag 
#                                                                + 'EfficiencyTables/' + dataset + '_' + subjet_info[1] 
#                                                                + '_' + flav_tag + 'eff.csv')

#         # -- Does Subjet pass the discriminator cut and is it updated -- #
#         SubJet01_isBtagged = BtagUpdater(SubJet01, EffFileDict['Eff_File_s01'], SF_filename, Fitting, sysType, bdisc)
#         SubJet02_isBtagged = BtagUpdater(SubJet02, EffFileDict['Eff_File_s02'], SF_filename, Fitting, sysType, bdisc)
#         SubJet11_isBtagged = BtagUpdater(SubJet11, EffFileDict['Eff_File_s11'], SF_filename, Fitting, sysType, bdisc)
#         SubJet12_isBtagged = BtagUpdater(SubJet12, EffFileDict['Eff_File_s12'], SF_filename, Fitting, sysType, bdisc)

#         # If either subjet 1 or 2 in FatJet 0 and 1 is btagged after update, then that FatJet is considered btagged #
#         btag_s0 = (SubJet01_isBtagged) | (SubJet02_isBtagged)  
#         btag_s1 = (SubJet11_isBtagged) | (SubJet12_isBtagged)

#         # --- Re-Define b-Tag Regions with "Updated" Tags ---- #
#         btag0 = (~btag_s0) & (~btag_s1) #(0b)
#         btag1 = btag_s0 ^ btag_s1 #(1b)
#         btag2 = btag_s0 & btag_s1 #(2b)
        
        
    return Btag_wgts