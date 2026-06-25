from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/gpfs/work/aac/yifansun2302/cineCMR_SAM")

PATIENT_LIST = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t20.xlsx"
QA_FILE = PROJECT_ROOT / "outputs/mnm2_lax_t20_preprocess_check.xlsx"

OUT_TRAIN = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t20_train300.xlsx"
OUT_VAL = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t20_val60.xlsx"
OUT_SUMMARY = PROJECT_ROOT / "outputs/mnm2_lax_t20_split_summary.xlsx"

SEED = 2026
N_VAL = 60


def normalize_patient_id(x):
    return str(x).replace(".0", "").zfill(3)


def main():
    patient_df = pd.read_excel(PATIENT_LIST)
    qa_df = pd.read_excel(QA_FILE)

    patient_df["patient_id"] = patient_df["patient_id"].apply(normalize_patient_id)
    qa_df["patient_id"] = qa_df["patient_id"].apply(normalize_patient_id)

    if len(patient_df) != 360:
        raise RuntimeError(f"Expected 360 cases, got {len(patient_df)}")

    rng = np.random.default_rng(SEED)
    patient_ids = np.array(sorted(patient_df["patient_id"].unique()))
    rng.shuffle(patient_ids)

    val_ids = set(patient_ids[:N_VAL])
    train_ids = set(patient_ids[N_VAL:])

    train_df = patient_df[patient_df["patient_id"].isin(train_ids)].copy()
    val_df = patient_df[patient_df["patient_id"].isin(val_ids)].copy()

    train_df = train_df.sort_values("patient_id").reset_index(drop=True)
    val_df = val_df.sort_values("patient_id").reset_index(drop=True)

    if len(train_df) != 300:
        raise RuntimeError(f"Expected 300 train cases, got {len(train_df)}")
    if len(val_df) != 60:
        raise RuntimeError(f"Expected 60 val cases, got {len(val_df)}")
    if set(train_df["patient_id"]) & set(val_df["patient_id"]):
        raise RuntimeError("Train/val leakage detected")

    split_df = pd.concat(
        [
            train_df[["patient_id"]].assign(split="train"),
            val_df[["patient_id"]].assign(split="val"),
        ],
        ignore_index=True,
    )

    summary_df = qa_df.merge(split_df, on="patient_id", how="left")
    if summary_df["split"].isna().any():
        missing = summary_df.loc[summary_df["split"].isna(), "patient_id"].tolist()
        raise RuntimeError(f"Missing split assignment: {missing[:10]}")

    train_df.to_excel(OUT_TRAIN, index=False)
    val_df.to_excel(OUT_VAL, index=False)
    summary_df.to_excel(OUT_SUMMARY, index=False)

    print(f"Saved train list: {OUT_TRAIN}")
    print(f"Saved val list: {OUT_VAL}")
    print(f"Saved split summary: {OUT_SUMMARY}")
    print(f"Train cases: {len(train_df)}")
    print(f"Val cases: {len(val_df)}")
    print("Available summary columns:")
    print(list(summary_df.columns))

    time_col = None
    for candidate in ["time_frames", "original_time_frames", "original_t", "time_frame", "T"]:
        if candidate in summary_df.columns:
            time_col = candidate
            break

    if time_col is not None:
        print(f"Train {time_col} summary:")
        print(summary_df.loc[summary_df["split"] == "train", time_col].describe())
        print(f"Val {time_col} summary:")
        print(summary_df.loc[summary_df["split"] == "val", time_col].describe())
    else:
        print("No time-frame column found in split summary; split files were still saved successfully.")


if __name__ == "__main__":
    main()