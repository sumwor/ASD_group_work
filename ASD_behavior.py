class ASD_behavior:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        animal.info = os.oath.join(self.root_dir, 'AnimalList.csv')
        animal_info= pd.read_csv(animal_info_path)

self.data_index = pd.DataFrame()
    
    self.animalID = animal_info['AnimalID']
    self.Genotype = animal_info['Genotype']
   self.data_index = pd.DataFrame ()

#3 sessions per animal
idx = 3

for aa in self.animalID:
    datafolder = os.path.join(self.root_dir, 'Data', str(aa))
    behavioralFile = glob-glob(datafolder, '*.csv')
    nFiles = len(behavioralFile)
    for ff in range(nFiles):
        self.data_index.loc[idx, 'Animal'] = aa
        self.data_index.loc[idx, 'Genotype'] = self.Genotype[aidx]
        self.data_index.loc[idx, 'BehaviorFile'] = behavioralFile[ff]
        
        idx += 1
        
#nFiles should be 3, bc 3 trials per animal
    #there are two "for" loops because data structured in different layers (by animal,
    #which has 3 csv files per)
    #HAVE TO IMPORT GLOB PACKAGE TO BRING IN ASD DATA
    
#* means anything can precede the csv file in the file name
#*10* means anything before 10 is irrelevant, anything after 10 is, too
    #this prevents inconsistencies in file names from making the file not get pulled from folder
    
    def average_performance(self):
        # calculate average performance for each session, and plot the learning curve
        pass

def average_perf(root_directory):
   sum(reward) / len(100)
    

if __name__ == "__main__":
    root_dir = r'C:\\Users\\Linda\\Documents\\GitHub\\ASD_group_work\\Data'
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()

#reroute the data directory to "ASD_group_work" in my personal downloads? (Line beginning with "root_dir")
#SUM/LEN the correct method?


