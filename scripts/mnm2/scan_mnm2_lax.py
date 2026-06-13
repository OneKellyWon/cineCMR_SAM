from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


MNM2_ROOT = Path("/gpfs/work/aac/yifansun2302/data/MnM2")
DATASET_DIR = MNM2_ROOT / "dataset"
OUT_DIR = Path("/gpfs/work/aac/yifansun2302/cineCMR_SAM/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_array(path):
    arr = nib.load(str(path)).get_fdata()
    arr = np.squeeze(arr)
    return arr


def label_values(path):
    arr = np.round(load_array(path)).astype(int)
    return ",".join(str(x) for x in np.unique(arr))


def match_frame(cine, frame_img):
    cine = np.squeeze(cine)
    frame_img = np.squeeze(frame_img)

    if cine.ndim != 3:
        raise ValueError(f"Expected cine to be 3D [H,W,T], got shape {cine.shape}")

    if frame_img.ndim != 2:
        raise ValueError(f"Expected ED/ES image to be 2D [H,W], got shape {frame_img.shape}")

    if cine.shape[:2] != frame_img.shape:
        raise ValueError(f"Shape mismatch: cine {cine.shape}, frame {frame_img.shape}")

    errors = []
    for t in range(cine.shape[-1]):
        diff = cine[..., t].astype(np.float32) - frame_img.astype(np.float32)
        errors.append(float(np.mean(diff * diff)))

    return int(np.argmin(errors)), float(np.min(errors))


def main():
    rows = []

    case_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        patient_id = case_dir.name

        cine_file = case_dir / f"{patient_id}_LA_CINE.nii.gz"
        ed_file = case_dir / f"{patient_id}_LA_ED.nii.gz"
        es_file = case_dir / f"{patient_id}_LA_ES.nii.gz"
        ed_gt_file = case_dir / f"{patient_id}_LA_ED_gt.nii.gz"
        es_gt_file = case_dir / f"{patient_id}_LA_ES_gt.nii.gz"

        row = {
            "patient_id": patient_id,
            "cine_file": str(cine_file),
            "ed_file": str(ed_file),
            "es_file": str(es_file),
            "ed_gt_file": str(ed_gt_file),
            "es_gt_file": str(es_gt_file),
            "status": "ok",
        }

        required = [cine_file, ed_file, es_file, ed_gt_file, es_gt_file]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            row["status"] = "missing_files"
            row["missing_files"] = ";".join(missing)
            rows.append(row)
            continue

        try:
            cine_img = nib.load(str(cine_file))
            cine_shape = cine_img.shape
            cine = np.squeeze(cine_img.get_fdata())

            ed = load_array(ed_file)
            es = load_array(es_file)

            ed_frame, ed_match_mse = match_frame(cine, ed)
            es_frame, es_match_mse = match_frame(cine, es)

            ed_gt = np.round(load_array(ed_gt_file)).astype(int)
            es_gt = np.round(load_array(es_gt_file)).astype(int)

            row.update({
                "cine_shape": str(cine_shape),
                "squeezed_cine_shape": str(cine.shape),
                "time_frames": int(cine.shape[-1]) if cine.ndim == 3 else None,
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "ed_match_mse": ed_match_mse,
                "es_match_mse": es_match_mse,
                "ed_gt_shape": str(ed_gt.shape),
                "es_gt_shape": str(es_gt.shape),
                "ed_gt_labels": ",".join(str(x) for x in np.unique(ed_gt)),
                "es_gt_labels": ",".join(str(x) for x in np.unique(es_gt)),
                "ed_gt_foreground_pixels": int(np.sum(ed_gt > 0)),
                "es_gt_foreground_pixels": int(np.sum(es_gt > 0)),
            })

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)

        rows.append(row)

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / "mnm2_lax_summary.xlsx"
    df.to_excel(out_file, index=False)

    print(f"Saved: {out_file}")
    print(f"Total cases: {len(df)}")
    print(df["status"].value_counts(dropna=False))

    if "time_frames" in df.columns:
        print("time frame summary:")
        print(df["time_frames"].describe())

    bad = df[df["status"] != "ok"]
    if len(bad) > 0:
        print("Bad cases:")
        print(bad[["patient_id", "status", "missing_files", "error"]])


if __name__ == "__main__":
    main()