
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ASD_behavior:
    def __init__(self, root_dir, output_dir=None):


        self.root_dir = root_dir
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(self.root_dir), "plots")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_animal_list(self):
        animal_list_path = os.path.join(self.root_dir, "AnimalList.csv")
        animal_info = pd.read_csv(animal_list_path, encoding="utf-8-sig")
        animal_info.columns = [col.strip() for col in animal_info.columns]
        animal_info["AnimalID"] = animal_info["AnimalID"].astype(str).str.strip()
        animal_info["Genotype"] = animal_info["Genotype"].astype(str).str.strip().str.upper()
        return animal_info

    def _calculate_single_file_performance(self, file_path):
        data = pd.read_csv(file_path)

        if "reward" not in data.columns or "trial" not in data.columns:
            raise ValueError(
                f"{file_path} must contain both 'reward' and 'trial' columns."
            )

        reward_col = data["reward"]

        # transfer blank cell into 0
        reward_filled = reward_col.fillna(0)

        # output True for cells' number > 0, False for cells' number = 0
        is_correct = reward_filled > 0

        # transfer True/False into 1/0
        correct_col = is_correct.astype(int)

        data["correct"] = correct_col

        # every 100 trials as a group
        trial_col = data["trial"]

        # make trial 1-100 as group 1
        trial_minus_one = trial_col - 1

        block_index = trial_minus_one // 100

        # label group number, begining from 1
        block_number = block_index + 1

        data["group_number"] = block_number.astype(int)

        # calculate the average value of each group
        grouped_data = data.groupby("group_number")

        average_correct = grouped_data["correct"].mean()

        # transfer the data into excel
        result = average_correct.reset_index()

        # transfer the result into percentage form
        result["accuracy_percent"] = result["correct"] * 100

        result = result.rename(
            columns={
                "group_number": "block",
                "correct": "performance"
            }
        )

        return result

    def collect_all_performance(self):
        # read animal list
        animal_info = self._load_animal_list()
        all_results = []

        # loop through each animal and their sessions to get info
        for _, row in animal_info.iterrows():
            animal_id = row["AnimalID"]
            genotype = row["Genotype"]

            # find the path and all csv files for this animal
            animal_dir = os.path.join(self.root_dir, animal_id)
            pattern = os.path.join(animal_dir, "*_behaviorDF.csv")
            file_list = sorted(glob.glob(pattern))

            # if no file found, print warning and skip this animal
            if len(file_list) == 0:
                print(f"Warning: no behaviorDF csv found for animal {animal_id}")
                continue
            # loop through each file, and enumerate session index starting from 1
            for session_idx, file_path in enumerate(file_list, start=1):
                
                # add animal/session info to the result
                result = self._calculate_single_file_performance(file_path)
                result["AnimalID"] = animal_id
                result["Genotype"] = genotype
                result["Session"] = session_idx
                result["FileName"] = os.path.basename(file_path)
                # add all results/info into a list
                all_results.append(result)

        if len(all_results) == 0:
            raise ValueError("No valid behavior files were found under root_dir.")

        # concatenate all lists into a single dataframe
        all_df = pd.concat(all_results, ignore_index=True)

        # reorder columns for better readability
        all_df = all_df[
            [
                "AnimalID",
                "Genotype",
                "Session",
                "FileName",
                "block",
                "performance",
                "accuracy_percent",
            ]
        ]

        return all_df

    def plot_session_learning_curves(self, all_df):
        colors = {
            "WT": "tab:blue",
            "HET": "tab:orange",
        }

        summary_rows = []
        sessions = sorted(all_df["Session"].unique())

        for session in sessions:
            session_df = all_df[all_df["Session"] == session].copy()

            # find the smallest available block number across mice in this session
            mouse_block_counts = (
                session_df.groupby("AnimalID")["block"]
                .max()
                .reset_index(name="max_block")
            )

            common_max_block = int(mouse_block_counts["max_block"].min())

            # keep only blocks that every mouse has
            session_df = session_df[session_df["block"] <= common_max_block].copy()

            fig, ax = plt.subplots(figsize=(8, 5))

            for genotype in ["WT", "HET"]:
                geno_df = session_df[session_df["Genotype"] == genotype].copy()

                if geno_df.empty:
                    continue

                # dashed lines for each mouse
                for animal_id, mouse_df in geno_df.groupby("AnimalID"):
                    mouse_df = mouse_df.sort_values("block")
                    ax.plot(
                        mouse_df["block"],
                        mouse_df["performance"],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.45,
                        color=colors[genotype],
                    )

                # solid mean line + SEM
                summary = (
                    geno_df.groupby("block")["performance"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                summary["sem"] = summary["std"].fillna(0) / np.sqrt(summary["count"])
                summary["Session"] = session
                summary["Genotype"] = genotype
                summary_rows.append(summary.copy())

                ax.errorbar(
                    summary["block"],
                    summary["mean"],
                    yerr=summary["sem"],
                    fmt="o-",
                    linewidth=2.5,
                    markersize=5,
                    capsize=3,
                    color=colors[genotype],
                )

            ax.set_title(f"Session {session}", fontsize=14)
            ax.set_xlabel("Trial block (100 trials/block)", fontsize=12)
            ax.set_ylabel("Performance", fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticks(sorted(session_df["block"].unique()))
            ax.grid(alpha=0.3)

            legend_handles = [
                Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.5, label="WT individual"),
                Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2.5, label="WT mean ± SEM"),
                Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.5, label="HET individual"),
                Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2.5, label="HET mean ± SEM"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower right",     
                fontsize=9,          
                frameon=False
            )

            plt.tight_layout()

            save_path = os.path.join(
                self.output_dir,
                f"Session_{session}_learning_curve.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df = summary_df.rename(columns={"mean": "mean_performance", "std": "sd", "count": "n_mice"})
        return summary_df

    def average_performance(self):
        all_df = self.collect_all_performance()
        summary_df = self.plot_session_learning_curves(all_df)

        all_df.to_csv(
            os.path.join(self.output_dir, "all_mouse_block_performance.csv"),
            index=False
        )
        summary_df.to_csv(
            os.path.join(self.output_dir, "genotype_session_summary.csv"),
            index=False
        )

        print("Analysis finished.")
        print(f"Plots saved to: {self.output_dir}")
        print("Per-mouse block data saved as: all_mouse_block_performance.csv")
        print("Genotype summary saved as: genotype_session_summary.csv")

        return all_df, summary_df


if __name__ == "__main__":

    root_dir = r"C:\Users\Ding\Desktop\Linda lab\python analysis\ASD_group_work\Data"
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()
