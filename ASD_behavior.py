import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class ASD_behavior:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.animals = pd.read_csv(self.root_dir / "AnimalList.csv")

    def average_performance(self):

        #this creates a dictionary to hold the performance data for each session
        session_data = {1: [], 2: [], 3: []}

        # this iterates through each row in the animal list and retrieves animal ID/gentype
        for animal_row in self.animals.itertuples():
            animal_id = animal_row.AnimalID
            genotype = animal_row.Genotype
            animal_folder = self.root_dir / str(animal_id)

            # this gets all the csv files for the current animal and sorts them
            csv_files = sorted(animal_folder.glob("*.csv"))

            # this loops through our spreadsheets and also counts which spreadsheet we're on
            for session_num, csv_file in enumerate(csv_files, start=1):
                df = pd.read_csv(csv_file)

                # split into 100-trial blocks and calculate performance for each block
                num_blocks = len(df) // 100
                for block in range(num_blocks):
                    block_df = df.iloc[block*100 : (block+1)*100]
                    rewarded = block_df["reward"].notna().sum()
                    performance = rewarded / 100

                    # append this block's results as a dictionary entry into the proper session
                    session_data[session_num].append({
                        "animal_id": animal_id,
                        "genotype": genotype,
                        "block": block + 1,
                        "performance": performance
                    })

        # plots one graph per session
        session_names = {1: "AB1", 2: "AB2", 3: "AB3"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        for session_num, ax in enumerate(axes, start=1):
            df_session = pd.DataFrame(session_data[session_num])
            avg = df_session.groupby(["block", "genotype"])["performance"].mean().reset_index()
            sem = df_session.groupby(["block", "genotype"])["performance"].sem().reset_index()

            for genotype, group in avg.groupby("genotype"):
                err = sem[sem["genotype"] == genotype]["performance"].values
                ax.plot(group["block"], group["performance"], marker="o", label=genotype)
                ax.errorbar(group["block"], group["performance"], yerr=err, fmt="none", capsize=5)

            ax.set_title(session_names[session_num])
            ax.set_xlabel("100-Trial Block")
            ax.set_ylim(0, 1)
            #gives us percentages on the y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.legend()

        axes[0].set_ylabel("Average Performance (% rewarded trials)")
        fig.suptitle("WT vs HET Performance by Session", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.root_dir.parent / "performance_by_session.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    root_dir = '/Users/neilyraymond/Code/ASD_group_work/Data'
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()