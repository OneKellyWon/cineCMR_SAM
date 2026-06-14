from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


PROJECT_ROOT = Path("/gpfs/work/aac/yifansun2302/cineCMR_SAM")
SUMMARY_FILE = PROJECT_ROOT / "outputs/mnm2_lax_summary.xlsx"

OUT_DATA_DIR = PROJECT_ROOT / "example_data/data/MnM2_LAX_T20"
OUT_PATIENT_LIST = PROJECT_ROOT / "example_data/data/Patient_list/patient_list_mnm2_lax_t20.xlsx"
OUT_QA_FILE = PROJECT_ROOT / "outputs/mnm2_lax_t20_preprocess_check.xlsx"

TARGET_T = 20

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATIENT_LIST.parent.mkdir(parents=True, exist_ok=True)
OUT_QA_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_array(path):
    arr = nib.load(str(path)).get_fdata()
    return np.squeeze(arr)


def load_2d_gt(path):
    arr = load_array(path)
    arr = np.round(arr).astype(np.int16)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D GT, got {arr.shape}: {path}")
    return arr


def select_frames(t, ed_frame, es_frame, target_t):
    if t <= target_t:
        return list(range(t))

    base = np.linspace(0, t - 1, target_t)
    selected = set(int(round(x)) for x in base)

    selected.add(int(ed_frame))
    selected.add(int(es_frame))

    selected = sorted(selected)

    protected = {int(ed_frame), int(es_frame)}

    while len(selected) > target_t:
        removable = [f for f in selected if f not in protected]
        if not removable:
            raise RuntimeError("No removable frame available while reducing selected frames.")

        # Remove frames that are closest to another selected frame, while never removing ED/ES.
        best_frame = None
        best_score = None

        for f in removable:
            others = [x for x in selected if x != f]
            nearest_gap = min(abs(f - x) for x in others)
            distance_to_gt = min(abs(f - ed_frame), abs(f - es_frame))

            # Lower score means more redundant and less important.
            score = (nearest_gap, distance_to_gt)

            if best_score is None or score < best_score:
                best_score = score
                best_frame = f

        selected.remove(best_frame)

    selected = sorted(selected)

    if ed_frame not in selected:
        raise RuntimeError(f"ED frame {ed_frame} lost from selected frames: {selected}")
    if es_frame not in selected:
        raise RuntimeError(f"ES frame {es_frame} lost from selected frames: {selected}")

    return selected


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

        patient_id = cine_file.parent.name

        ed_frame_original = int(row["ed_frame"])
        es_frame_original = int(row["es_frame"])

        cine_img = nib.load(str(cine_file))
        cine = np.squeeze(np.asanyarray(cine_img.dataobj))

        if cine.ndim != 3:
            raise ValueError(f"Expected LA_CINE as [H,W,T], got {cine.shape}: {cine_file}")

        h, w, t = cine.shape

        if not (0 <= ed_frame_original < t):
            raise ValueError(f"Invalid ED frame {ed_frame_original} for {patient_id}, T={t}")
        if not (0 <= es_frame_original < t):
            raise ValueError(f"Invalid ES frame {es_frame_original} for {patient_id}, T={t}")
        if ed_frame_original == es_frame_original:
            raise ValueError(f"ED and ES frame are identical for {patient_id}: {ed_frame_original}")

        selected = select_frames(
            t=t,
            ed_frame=ed_frame_original,
            es_frame=es_frame_original,
            target_t=TARGET_T,
        )

        selected_cine = cine[:, :, selected]

        num_selected = len(selected)
        padding_frames = TARGET_T - num_selected

        out_cine = np.zeros((h, w, TARGET_T), dtype=selected_cine.dtype)
        out_cine[:, :, :num_selected] = selected_cine

        ed_gt = load_2d_gt(ed_gt_file)
        es_gt = load_2d_gt(es_gt_file)

        if ed_gt.shape != (h, w):
            raise ValueError(f"ED GT shape mismatch for {patient_id}: {ed_gt.shape} vs {(h, w)}")
        if es_gt.shape != (h, w):
            raise ValueError(f"ES GT shape mismatch for {patient_id}: {es_gt.shape} vs {(h, w)}")

        ed_frame_new = selected.index(ed_frame_original)
        es_frame_new = selected.index(es_frame_original)

        out_seg = np.zeros((h, w, TARGET_T), dtype=np.int16)
        out_seg[:, :, ed_frame_new] = ed_gt
        out_seg[:, :, es_frame_new] = es_gt

        case_out_dir = OUT_DATA_DIR / patient_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        out_img_file = case_out_dir / "img_LAX.nii.gz"
        out_seg_file = case_out_dir / "seg_LAX.nii.gz"

        out_img = nib.Nifti1Image(out_cine, affine=cine_img.affine, header=cine_img.header)
        out_img.set_data_dtype(out_cine.dtype)
        nib.save(out_img, str(out_img_file))

        seg_img = nib.Nifti1Image(out_seg, affine=cine_img.affine, header=cine_img.header)
        seg_img.set_data_dtype(np.int16)
        nib.save(seg_img, str(out_seg_file))

        labels = ",".join(str(x) for x in np.unique(out_seg))
        nonzero_frames = [i for i in range(TARGET_T) if np.sum(out_seg[:, :, i] > 0) > 0]

        patient_rows.append({
            "patient_id": patient_id,
            "img_file": str(out_img_file),
            "seg_file": str(out_seg_file),
            "total_slice_num": 1,
            "lax_type": "LAX",
        })

        qa_rows.append({
            "patient_id": patient_id,
            "original_time_frames": t,
            "target_time_frames": TARGET_T,
            "selected_frames": ",".join(str(x) for x in selected),
            "num_selected_before_padding": num_selected,
            "padding_frames": padding_frames,
            "ed_frame_original": ed_frame_original,
            "es_frame_original": es_frame_original,
            "ed_frame_new": ed_frame_new,
            "es_frame_new": es_frame_new,
            "nonzero_gt_frames": ",".join(str(x) for x in nonzero_frames),
            "seg_labels": labels,
            "img_shape": str(out_cine.shape),
            "seg_shape": str(out_seg.shape),
            "img_file": str(out_img_file),
            "seg_file": str(out_seg_file),
        })

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_excel(OUT_PATIENT_LIST, index=False)

    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_excel(OUT_QA_FILE, index=False)

    print(f"Saved patient list: {OUT_PATIENT_LIST}")
    print(f"Saved QA file: {OUT_QA_FILE}")
    print(f"Cases: {len(patient_df)}")
    print("Original time frame summary:")
    print(qa_df["original_time_frames"].describe())
    print("Target time frame summary:")
    print(qa_df["target_time_frames"].describe())
    print("Padding frame summary:")
    print(qa_df["padding_frames"].describe())
    print("Cases with padding > 0:", int((qa_df["padding_frames"] > 0).sum()))
    print("Seg label sets:")
    print(qa_df["seg_labels"].value_counts())
    print("Example rows:")
    print(qa_df.head())


if __name__ == "__main__":
    main()