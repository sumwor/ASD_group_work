import os
import pandas as pd
import glob
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

class ASD_behavior:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        animal_info_path = os.path.join(self.root_dir, 'Data','AnimalList.csv')
        animal_info = pd.read_csv(animal_info_path)
        
        self.animalID = animal_info['AnimalID']
        self.Genotype = animal_info['Genotype']

        self.data_index = pd.DataFrame()

        # 3 sessions per animal
        idx = 0

        for aidx, aa in enumerate(self.animalID):
            datafolder = os.path.join(self.root_dir, 'Data', str(aa))
            behavioralFile = glob.glob(os.path.join(datafolder, '*.csv'))
            nFiles = len(behavioralFile)
            # sort behavioralFile in order if it's not

            for ff in range(nFiles):
                self.data_index.loc[idx, 'Animal'] = str(aa)
                self.data_index.loc[idx, 'Genotype'] = self.Genotype[aidx]
                self.data_index.loc[idx, 'BehaviorFile'] = behavioralFile[ff]
                self.data_index.loc[idx, 'Session'] = ff+1
                idx += 1
    


    def average_performance(self):
        # calculate average performance for each session, and plot the learning curve
        nFiles = self.data_index.shape[0]
        ave_performace = np.full((nFiles, 20), np.nan)
        for ii in range(nFiles):
            behDF = pd.read_csv(self.data_index['BehaviorFile'][ii])
            nTrials = behDF.shape[0]
            blockLength = 100
            nBlock = np.ceil(nTrials/blockLength)
            for bb in range(int(nBlock)):
                startTrial = bb*blockLength
                endTrial = (bb+1)*blockLength-1
                if endTrial > nTrials:
                    endTrial = nTrials-1

                avePerf = np.sum(~np.isnan(behDF['reward'][startTrial:endTrial]))/blockLength
                ave_performace[ii, bb] = avePerf

        # plot them in 1-3 AB sessions, in WT and Mut
        WT = 'WT'
        Mut = 'HET'

        WTMask = self.data_index['Genotype'] == WT
        MutMask = self.data_index['Genotype'] == Mut

        Ses1Mask = self.data_index['Session'] == 1
        Ses2Mask = self.data_index['Session'] == 2
        Ses3Mask = self.data_index['Session'] == 3

        # make subplots

        
        plt.plot(np.mean(ave_performace[WTMask & Ses1Mask, :], axis=0))
        plt.plot(np.mean(ave_performace[MutMask & Ses1Mask, :], axis=0))
        
                     
        

        x=1



if __name__ == "__main__":
    plt.ion()
    root_dir = r'C:\Users\Linda\Documents\GitHub\ASD_group_work'
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()