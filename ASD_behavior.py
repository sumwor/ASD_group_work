import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class ASD_behavior:
    def __init__(self, root_dir, block_size=20):
        self.root_dir = root_dir
        self.block_size = block_size
        self.metadata = self._load_metadata()
        self.genotype_map = self._build_genotype_map()

    # ----------------------------
    # Metadata (AnimalList)
    # ----------------------------
    def _load_metadata(self):
        for name in ("AnimalList.xlsx", "AnimalList.csv", "metadata.csv"):
            path = os.path.join(self.root_dir, name)
            if os.path.isfile(path):
                print(f"[Metadata] Loaded {name}")

                if name.endswith(".xlsx"):
                    return pd.read_excel(path)
                else:
                    return pd.read_csv(path)

        print("[ERROR] AnimalList not found.")
        return pd.DataFrame(columns=["AnimalID", "Genotype"])

    def _build_genotype_map(self):
        df = self.metadata.copy()

        if "AnimalID" not in df.columns or "Genotype" not in df.columns:
            print("[ERROR] Metadata missing required columns.")
            return {}

        # 🔧 CLEANING STEP (CRITICAL)
        df["AnimalID"] = df["AnimalID"].astype(str).str.strip()
        df["Genotype"] = df["Genotype"].astype(str).str.strip().str.upper()

        print("[DEBUG] Metadata preview:")
        print(df.head())

        return dict(zip(df["AnimalID"], df["Genotype"]))

    def _get_genotype(self, animal_id):
        animal_id = str(animal_id).strip()
        return self.genotype_map.get(animal_id, "Unknown")

    # ----------------------------
    # File parsing
    # ----------------------------
    def _parse_date(self, filename):
        match = re.search(r"(\d{8})", filename)
        return match.group(1) if match else None

    # ----------------------------
    # Load all data
    # ----------------------------
    def _load_all_data(self):
        records = []
        root = Path(self.root_dir)

        if not root.exists():
            print(f"[ERROR] Root directory not found: {self.root_dir}")
            return pd.DataFrame()

        for animal_path in root.iterdir():
            if not animal_path.is_dir():
                continue

            animal_id = animal_path.name.strip()
            genotype = self._get_genotype(animal_id)

            for fname in os.listdir(animal_path):
                if not fname.endswith(".csv"):
                    continue

                date = self._parse_date(fname)
                if date is None:
                    continue

                file_path = os.path.join(animal_path, fname)

                try:
                    df = pd.read_csv(file_path)
                except:
                    continue

                if "reward" not in df.columns:
                    continue

                tmp = df[["reward"]].copy()
                tmp["AnimalID"] = animal_id
                tmp["Genotype"] = genotype
                tmp["Date"] = date

                records.append(tmp)

        if not records:
            print("[ERROR] No valid data found.")
            return pd.DataFrame()

        return pd.concat(records, ignore_index=True)

    # ----------------------------
    # Block processing
    # ----------------------------
    def _block_reward_rate(self, reward_series):
        rewards = reward_series.reset_index(drop=True)
        n_blocks = max(1, len(rewards) // self.block_size)

        results = []
        for b in range(n_blocks):
            block = rewards.iloc[b*self.block_size:(b+1)*self.block_size]
            rate = block.notna().mean()
            results.append((b + 1, rate))

        return results

    # ----------------------------
    # Compute performance
    # ----------------------------
    def compute_performance(self):
        raw = self._load_all_data()
        if raw.empty:
            return pd.DataFrame()

        # ✅ assign sessions per animal
        raw["Session"] = None

        for animal, sub in raw.groupby("AnimalID"):
            unique_dates = sorted(sub["Date"].unique())
            date_to_session = {d: i + 1 for i, d in enumerate(unique_dates)}
            raw.loc[sub.index, "Session"] = sub["Date"].map(date_to_session)

        raw = raw[raw["Session"].isin([1, 2, 3])]

        rows = []

        for (animal, date, geno), group in raw.groupby(
            ["AnimalID", "Date", "Genotype"]
        ):
            session = group["Session"].iloc[0]

            for block, rate in self._block_reward_rate(group["reward"]):
                rows.append({
                    "AnimalID": animal,
                    "Genotype": geno,
                    "Session": session,
                    "Block": block,
                    "RewardRate": rate
                })

        df = pd.DataFrame(rows)

        # 🔥 KEEP ONLY WT + HET (no Unknown mixing)
        df = df[df["Genotype"].isin(["WT", "HET"])]

        agg = (
            df.groupby(["Genotype", "Session", "Block"])["RewardRate"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )

        return agg

    # ----------------------------
    # Plotting
    # ----------------------------
    def average_performance(self, save_path=None, show=True):
        agg = self.compute_performance()

        if agg.empty:
            print("[ERROR] No data to plot.")
            return

        colors = {
            "WT": "black",
            "HET": "red"
        }

        sessions = sorted(agg["Session"].unique())
        fig, axes = plt.subplots(1, len(sessions), figsize=(5*len(sessions), 4), sharey=True)

        if len(sessions) == 1:
            axes = [axes]

        for ax, sess in zip(axes, sessions):
            sess_data = agg[agg["Session"] == sess]

            for geno in ["WT", "HET"]:
                sub = sess_data[sess_data["Genotype"] == geno]

                if sub.empty:
                    continue

                ax.plot(
                    sub["Block"],
                    sub["mean"],
                    color=colors[geno],
                    linewidth=2.5,
                    label=geno
                )

                ax.fill_between(
                    sub["Block"],
                    sub["mean"] - sub["sem"].fillna(0),
                    sub["mean"] + sub["sem"].fillna(0),
                    color=colors[geno],
                    alpha=0.15
                )

            ax.set_title(f"Session {sess}")
            ax.set_xlabel("Block")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, linestyle="--", color="gray")

        axes[0].set_ylabel("Reward rate")
        axes[0].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[Saved] {save_path}")

        if show:
            plt.show()

        return agg


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    root_dir = r"C:\Users\imtre\Downloads\ASD_group_work-main\ASD_group_work-main\Data"

    ASD = ASD_behavior(root_dir, block_size=20)
    ASD.average_performance(save_path="learning_curve.png")
