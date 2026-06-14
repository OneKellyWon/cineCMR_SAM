from pathlib import Path
import os
import numpy as np
import pandas as pd
import nibabel as nib


MNM2_ROOT = Path("/gpfs/work/aac/yifansun2302/data/MnM2")
DATASET_DIR = MNM2_ROOT / "dataset"

PROJECT_ROOT = Path("/gpfs/work/aac/yifansun2302/cineCMR_SAM")
SUMMARY_FILE = PROJECT_ROOT / "outputs/mnm2_lax_summary.xlsx"

OUT_DATA_DIR = PROJECT_ROOT / "example_data/data/MnM2_LAX"
OUT_PATIENT_LIST = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax.xlsx"

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATIENT_LIST.parent.mkdir(parents=True, exist_ok=True)


def load_2d_gt(path):
    arr = nib.load(str(path)).get_fdata()
    arr = np.squeeze(arr)
    arr = np.round(arr).astype(np.int16)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D GT, got {arr.shape}: {path}")
    return arr


def make_symlink_or_replace(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main():
    summary = pd.read_excel(SUMMARY_FILE)

    patient_rows = []
    qa_rows = []

    for _, row in summary.iterrows():
        if row["status"] != "ok":
            raise RuntimeError(f"Bad case in summary: {row}")

        cine_file = Path(row["cine_file"])
        ed_gt_file = Path(row["ed_gt_file"])
        es_gt_file = Path(row["es_gt_file"])

        # Use the real folder name, not Excel's numeric patient_id, to preserve leading zeros.
        patient_id = cine_file.parent.name

        ed_frame = int(row["ed_frame"])
        es_frame = int(row["es_frame"])

        cine_img = nib.load(str(cine_file))
        cine = np.squeeze(np.asanyarray(cine_img.dataobj))

        if cine.ndim != 3:
            raise ValueError(f"Expected LA_CINE as [H,W,T], got {cine.shape}: {cine_file}")

        h, w, t = cine.shape

        if not (0 <= ed_frame < t):
            raise ValueError(f"Invalid ED frame {ed_frame} for {patient_id}, T={t}")
        if not (0 <= es_frame < t):
            raise ValueError(f"Invalid ES frame {es_frame} for {patient_id}, T={t}")
        if ed_frame == es_frame:
            raise ValueError(f"ED and ES frame are identical for {patient_id}: {ed_frame}")

        ed_gt = load_2d_gt(ed_gt_file)
        es_gt = load_2d_gt(es_gt_file)

        if ed_gt.shape != (h, w):
            raise ValueError(f"ED GT shape mismatch for {patient_id}: {ed_gt.shape} vs {(h, w)}")
        if es_gt.shape != (h, w):
            raise ValueError(f"ES GT shape mismatch for {patient_id}: {es_gt.shape} vs {(h, w)}")

        seg = np.zeros((h, w, t), dtype=np.int16)
        seg[:, :, ed_frame] = ed_gt
        seg[:, :, es_frame] = es_gt

        case_out_dir = OUT_DATA_DIR / patient_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        out_img = case_out_dir / "img_LAX.nii.gz"
        out_seg = case_out_dir / "seg_LAX.nii.gz"

        # Save LAX cine as 3D [H, W, T], because the dataloader expects LAX before adding slice axis.
        cine_to_save = np.asarray(cine)
        cine_img_out = nib.Nifti1Image(cine_to_save, affine=cine_img.affine, header=cine_img.header)
        cine_img_out.set_data_dtype(cine_to_save.dtype)
        nib.save(cine_img_out, str(out_img))

        # Save segmentation with the cine affine/header, matching [H,W,T] geometry.
        seg_img = nib.Nifti1Image(seg, affine=cine_img.affine, header=cine_img.header)
        seg_img.set_data_dtype(np.int16)
        nib.save(seg_img, str(out_seg))

        labels = ",".join(str(x) for x in np.unique(seg))
        nonzero_frames = [i for i in range(t) if np.sum(seg[:, :, i] > 0) > 0]

        patient_rows.append({
            "patient_id": patient_id,
            "img_file": str(out_img),
            "seg_file": str(out_seg),
            "total_slice_num": 1,
            "lax_type": "LAX",
        })

        qa_rows.append({
            "patient_id": patient_id,
            "time_frames": t,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "nonzero_gt_frames": ",".join(str(x) for x in nonzero_frames),
            "seg_labels": labels,
            "img_file": str(out_img),
            "seg_file": str(out_seg),
        })

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_excel(OUT_PATIENT_LIST, index=False)

    qa_df = pd.DataFrame(qa_rows)
    qa_file = PROJECT_ROOT / "outputs/mnm2_lax_preprocess_check.xlsx"
    qa_df.to_excel(qa_file, index=False)

    print(f"Saved patient list: {OUT_PATIENT_LIST}")
    print(f"Saved QA file: {qa_file}")
    print(f"Cases: {len(patient_df)}")
    print("Time frame summary:")
    print(qa_df["time_frames"].describe())
    print("Seg label sets:")
    print(qa_df["seg_labels"].value_counts())
    print("Example rows:")
    print(qa_df.head())


if __name__ == "__main__":
    main()