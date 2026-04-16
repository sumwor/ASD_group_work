class ASD_behavior:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        pass

    def average_performance(self):
        # calculate average performance for each session, and plot the learning curve
        pass



if __name__ == "__main__":
    root_dir = r'Y:\HongliWang\Miniscope\ASD'
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()

#Practice Comment 
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ASD_behavior:
    def __init__(self, root_dir, block_size=20):
        """
        root_dir  : top-level folder containing one subfolder per animal
        block_size: number of trials per block for within-session averaging
        """
        self.root_dir = root_dir
        self.block_size = block_size
        self.metadata = self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(self):
        """Load AnimalID/Genotype table from root_dir if present."""
        for name in ("metadata.csv", "animals.csv", "genotype.csv"):
            path = os.path.join(self.root_dir, name)
            if os.path.isfile(path):
                df = pd.read_csv(path)
                print(f"[Metadata] Loaded {name}")
                return df
        print("[Metadata] No metadata file found — genotype will be 'Unknown'")
        return pd.DataFrame(columns=["AnimalID", "Genotype"])

    def _get_genotype(self, animal_id):
        row = self.metadata[self.metadata["AnimalID"] == animal_id]
        return row["Genotype"].values[0] if len(row) else "Unknown"

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def _parse_date(self, filename):
        """
        Extract YYYYMMDD from filenames like: 365_20231227_behaviorDF.csv
        Returns a string 'YYYYMMDD', or None if not found.
        """
        match = re.search(r"(\d{8})", filename)
        return match.group(1) if match else None

    def _load_all_data(self):
        """
        Walk root_dir → animal subfolders → CSV files.
        Returns a long-form DataFrame:
            AnimalID | Genotype | Date | reward
        """
        records = []
        root = Path(self.root_dir)

        for animal_path in sorted(root.iterdir()):
            if not animal_path.is_dir():
                continue
            animal_id = animal_path.name
            genotype = self._get_genotype(animal_id)

            csv_files = [f for f in os.listdir(animal_path)
                         if f.lower().endswith(".csv")]

            for fname in csv_files:
                date = self._parse_date(fname)
                if date is None:
                    print(f"  [Skip] Can't parse date from: {fname}")
                    continue

                try:
                    df = pd.read_csv(os.path.join(animal_path, fname))
                except Exception as e:
                    print(f"  [Error] {fname}: {e}")
                    continue

                if "reward" not in df.columns:
                    print(f"  [Skip] No 'reward' column in: {fname}")
                    continue

                # Keep only the reward column, tag with animal/date info
                tmp = df[["reward"]].copy()
                tmp["AnimalID"] = animal_id
                tmp["Genotype"] = genotype
                tmp["Date"] = date
                records.append(tmp)

        if not records:
            return pd.DataFrame()

        return pd.concat(records, ignore_index=True)

    # ------------------------------------------------------------------
    # Performance calculation
    # ------------------------------------------------------------------

    def _block_reward_rate(self, reward_series):
        """
        Split one session's reward column into blocks of self.block_size.
        Rewarded trial  = any non-NaN integer value.
        Unrewarded trial = NaN.
        Returns list of (block_number, reward_rate) tuples.
        """
        rewards = reward_series.reset_index(drop=True)
        n_blocks = max(1, len(rewards) // self.block_size)
        results = []
        for b in range(n_blocks):
            block = rewards.iloc[b * self.block_size : (b + 1) * self.block_size]
            rate = block.notna().mean()   # non-NaN = rewarded
            results.append((b + 1, rate))
        return results

    def compute_performance(self):
        """
        Returns a DataFrame averaged across animals within each genotype:
            Genotype | Session | Date | Block | mean | sem
        """
        raw = self._load_all_data()
        if raw.empty:
            return pd.DataFrame()

        # Assign session numbers by sorted date (earliest date = session 1)
        dates_sorted = sorted(raw["Date"].unique())
        date_to_session = {d: i + 1 for i, d in enumerate(dates_sorted)}
        raw["Session"] = raw["Date"].map(date_to_session)

        # Compute per-animal, per-session block reward rates
        per_animal = []
        for (animal, date, geno), group in raw.groupby(["AnimalID", "Date", "Genotype"]):
            session = date_to_session[date]
            for block, rate in self._block_reward_rate(group["reward"]):
                per_animal.append({
                    "AnimalID": animal,
                    "Genotype": geno,
                    "Date": date,
                    "Session": session,
                    "Block": block,
                    "RewardRate": rate,
                })

        df = pd.DataFrame(per_animal)

        # Average across animals: one mean/sem per genotype × session × block
        agg = (
            df.groupby(["Genotype", "Session", "Date", "Block"])["RewardRate"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )
        return agg

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def average_performance(self, save_path=None, show=True):
        """
        One subplot per session (AB sess 1, 2, 3 ...).
        Each subplot: WT (orange) and HET (blue) reward rate across trial blocks.
        """
        agg = self.compute_performance()
        if agg.empty:
            print("[Error] No data to plot.")
            return

        sessions = sorted(agg["Session"].unique())
        n_sess = len(sessions)

        # Colors matching the sketch
        geno_style = {
            "WT":      {"color": "#E87722"},
            "HET":     {"color": "#3BAFD9"},
            "Unknown": {"color": "#888888"},
        }

        fig, axes = plt.subplots(1, n_sess, figsize=(4 * n_sess, 4), sharey=True)
        if n_sess == 1:
            axes = [axes]   # always iterable

        for ax, sess in zip(axes, sessions):
            sess_data = agg[agg["Session"] == sess]
            date_str = sess_data["Date"].iloc[0]   # 'YYYYMMDD'
            date_label = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"

            for geno, style in geno_style.items():
                sub = sess_data[sess_data["Genotype"] == geno].sort_values("Block")
                if sub.empty:
                    continue

                ax.plot(
                    sub["Block"], sub["mean"],
                    color=style["color"],
                    linewidth=2,
                    marker="o", markersize=4,
                    label=geno,
                )
                # Shaded SEM band
                ax.fill_between(
                    sub["Block"],
                    sub["mean"] - sub["sem"].fillna(0),
                    sub["mean"] + sub["sem"].fillna(0),
                    color=style["color"],
                    alpha=0.18,
                )

            ax.set_title(f"AB sess {sess}\n{date_label}", fontsize=10)
            ax.set_xlabel("Trial block", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(left=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

        axes[0].set_ylabel("Reward rate", fontsize=10)

        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, frameon=False, fontsize=9)

        plt.suptitle("Average performance per session", fontsize=12, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Saved] {save_path}")

        if show:
            plt.show()

        return agg


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root_dir = r'Y:\HongliWang\Miniscope\ASD'
    ASD_beh = ASD_behavior(root_dir, block_size=20)
    ASD_beh.average_performance(save_path="learning_curve.png")
