from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/gpfs/work/aac/yifansun2302/cineCMR_SAM")

PATIENT_LIST = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t15.xlsx"
QA_FILE = PROJECT_ROOT / "outputs/mnm2_lax_t15_preprocess_check.xlsx"

OUT_TRAIN = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t15_debug_train5.xlsx"
OUT_VAL = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t15_debug_val5.xlsx"
OUT_SUMMARY = PROJECT_ROOT / "outputs/mnm2_lax_t15_debug_split_summary.xlsx"

SEED = 2026
N_TRAIN = 5
N_VAL = 5


def normalize_patient_id(x):
    return str(x).replace(".0", "").zfill(3)


def main():
    patient_df = pd.read_excel(PATIENT_LIST)
    qa_df = pd.read_excel(QA_FILE)

    patient_df["patient_id"] = patient_df["patient_id"].apply(normalize_patient_id)
    qa_df["patient_id"] = qa_df["patient_id"].apply(normalize_patient_id)

    if len(patient_df) < N_TRAIN + N_VAL:
        raise RuntimeError(
            f"Need at least {N_TRAIN + N_VAL} cases, got {len(patient_df)}"
        )

    rng = np.random.default_rng(SEED)
    patient_ids = np.array(sorted(patient_df["patient_id"].unique()))
    rng.shuffle(patient_ids)

    train_ids = set(patient_ids[:N_TRAIN])
    val_ids = set(patient_ids[N_TRAIN:N_TRAIN + N_VAL])

    train_df = patient_df[patient_df["patient_id"].isin(train_ids)].copy()
    val_df = patient_df[patient_df["patient_id"].isin(val_ids)].copy()

    train_df = train_df.sort_values("patient_id").reset_index(drop=True)
    val_df = val_df.sort_values("patient_id").reset_index(drop=True)

    split_df = pd.concat(
        [
            train_df[["patient_id"]].assign(split="train"),
            val_df[["patient_id"]].assign(split="val"),
        ],
        ignore_index=True,
    )

    summary_df = qa_df.merge(split_df, on="patient_id", how="inner")

    train_df.to_excel(OUT_TRAIN, index=False)
    val_df.to_excel(OUT_VAL, index=False)
    summary_df.to_excel(OUT_SUMMARY, index=False)

    print(f"Saved train list: {OUT_TRAIN}")
    print(f"Saved val list: {OUT_VAL}")
    print(f"Saved split summary: {OUT_SUMMARY}")
    print(f"Train cases: {len(train_df)}")
    print(f"Val cases: {len(val_df)}")
    print("Train IDs:", ",".join(train_df["patient_id"].astype(str)))
    print("Val IDs:", ",".join(val_df["patient_id"].astype(str)))
    print("Debug split frame summary:")
    print(summary_df[["patient_id", "split", "original_time_frames", "selected_frames", "nonzero_gt_frames"]])


if __name__ == "__main__":
    main()
