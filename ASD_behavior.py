from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ASD_behavior:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

    @staticmethod
    def _session_performance(session_file):
        """Return percentage correct for one session CSV."""
        session_df = pd.read_csv(session_file)
        reward_numeric = pd.to_numeric(session_df["reward"], errors="coerce")
        is_correct = reward_numeric.notna() & (reward_numeric != 0)
        return is_correct.mean() * 100

    def average_performance(self):
        # Calculate average performance for each session, then plot learning curves.
        animal_list_file = self.root_dir / "AnimalList.csv"
        animal_map = pd.read_csv(animal_list_file)
        animal_to_genotype = {
            str(row["AnimalID"]): str(row["Genotype"]).strip().upper()
            for _, row in animal_map.iterrows()
        }

        records = []
        for animal_dir in sorted(self.root_dir.iterdir()):
            if not animal_dir.is_dir():
                continue
            animal_id = animal_dir.name
            if animal_id not in animal_to_genotype:
                continue

            session_files = sorted(animal_dir.glob(f"{animal_id}_*_behaviorDF.csv"))
            for day_index, session_file in enumerate(session_files, start=1):
                performance = self._session_performance(session_file)
                genotype = animal_to_genotype[animal_id]
                group = "Wild Type" if genotype == "WT" else "ASD"
                records.append(
                    {
                        "animal_id": animal_id,
                        "group": group,
                        "day": day_index,
                        "session_file": session_file.name,
                        "performance": performance,
                    }
                )

        if not records:
            raise ValueError("No session files were found under the provided root_dir.")

        performance_df = pd.DataFrame(records)
        summary = (
            performance_df.groupby(["group", "day"], as_index=False)["performance"]
            .mean()
            .sort_values(["group", "day"])
        )

        days = [1, 2, 3]
        wild_type = (
            summary[summary["group"] == "Wild Type"]
            .set_index("day")
            .reindex(days)["performance"]
        )
        asd = summary[summary["group"] == "ASD"].set_index("day").reindex(days)["performance"]

        plt.figure(figsize=(8, 5))
        plt.plot(days, wild_type.values, marker="o", linewidth=2, label="Wild Type")
        plt.plot(days, asd.values, marker="o", linewidth=2, label="ASD")
        plt.xticks(days, ["Day 1", "Day 2", "Day 3"])
        plt.ylabel("Correctness (%)")
        plt.xlabel("Test Day")
        plt.ylim(0, 100)
        plt.title("Average Performance: Wild Type vs ASD")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_file = self.root_dir / "WT_vs_ASD_average_performance.png"
        plt.savefig(output_file, dpi=300)
        plt.close()

        print("Saved plot:", output_file)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    root_dir = project_dir / "Data"
    ASD_beh = ASD_behavior(root_dir)
    ASD_beh.average_performance()