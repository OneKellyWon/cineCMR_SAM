import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

sys.path.append("/gpfs/work/aac/yifansun2302")

from cineCMR_SAM.segment_anything.model import build_model
from cineCMR_SAM.utils.config_util import Config

import cineCMR_SAM.dataset.build_CMR_datasets as build_CMR_datasets
import cineCMR_SAM.functions_collection as ff
import cineCMR_SAM.get_args_parser as get_args_parser


main_path = "/gpfs/work/aac/yifansun2302/cineCMR_SAM"

trial_name = "mnm2_lax_3class_t15_debug_train5_e2"
checkpoint_name = "model-2.pth"

text_prompt = False
box_prompt = False

pretrained_model = os.path.join(
    main_path,
    "example_data/models",
    trial_name,
    "models",
    checkpoint_name,
)

original_sam = os.path.join(
    main_path,
    "example_data/pretrained_sam/sam_vit_h_4b8939.pth",
)

args = get_args_parser.get_args_parser(
    text_prompt=text_prompt,
    box_prompt=box_prompt,
    pretrained_model=pretrained_model,
    original_sam=original_sam,
    vit_type="vit_h",
)
args = args.parse_args([])

args.num_classes = 4
args.max_timeframe = 15
args.turn_zero_seg_slice_into = 10

cfg = Config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

lax_text_prompt_feature = np.load(
    os.path.join(main_path, "example_data/data/text_prompt_clip/lax.npy")
)

patient_list_file_lax = os.path.join(
    main_path,
    "example_data/data/Patient_list/patient_list_mnm2_lax_t15_debug_val5.xlsx",
)
patient_index_list = np.arange(0, 5, 1)

dataset_pred = build_CMR_datasets.build_dataset(
    args,
    view_type="lax",
    patient_list_file=patient_list_file_lax,
    index_list=patient_index_list,
    text_prompt_feature=lax_text_prompt_feature,
    only_myo=False,
    shuffle=False,
    augment=False,
)

data_loader_pred = torch.utils.data.DataLoader(
    dataset_pred,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)

save_root = os.path.join(
    main_path,
    "example_data/models",
    trial_name,
    "predicts_mnm2_lax_t15_debug_val5",
)
ff.make_folder([save_root])


def save_lax_prediction(batch, output, args, save_folder_patient):
    pred_seg_crop = np.rollaxis(
        output["masks"].argmax(1).detach().cpu().numpy(),
        0,
        3,
    )

    original_shape = np.array([x.item() for x in batch["original_shape"]])
    centroid = batch["centroid"].numpy().flatten()

    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(int(centroid[dim]) - size // 2, 0)
        end = start + size

        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)

        crop_start_end_list.append([start, end])

    final_pred_seg = np.zeros(original_shape, dtype=np.int16)
    final_pred_seg[
        crop_start_end_list[0][0]:crop_start_end_list[0][1],
        crop_start_end_list[1][0]:crop_start_end_list[1][1],
        :
    ] = pred_seg_crop.astype(np.int16)

    original_image_file = batch["img_file"][0]
    original_seg_file = batch["seg_file"][0]

    original_img_nii = nib.load(original_image_file)
    affine = original_img_nii.affine

    nib.save(
        nib.Nifti1Image(final_pred_seg, affine),
        os.path.join(save_folder_patient, "seg_pred_LAX.nii.gz"),
    )

    shutil.copy2(original_image_file, os.path.join(save_folder_patient, "img_LAX.nii.gz"))
    shutil.copy2(original_seg_file, os.path.join(save_folder_patient, "seg_gt_LAX.nii.gz"))


with torch.no_grad():
    with torch.cuda.amp.autocast():
        model = build_model(args, device)

        print("loading pretrained model:", args.pretrained_model)
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        for _, batch in tqdm(enumerate(data_loader_pred), total=len(data_loader_pred)):
            patient_id = os.path.basename(os.path.dirname(batch["img_file"][0]))
            print("predict patient:", patient_id)

            save_folder_patient = os.path.join(save_root, patient_id)
            ff.make_folder([save_folder_patient])

            batch["image"] = batch["image"].to(torch.float16).cuda()
            batch["text_prompt_feature"] = batch["text_prompt_feature"].to(torch.float32)

            output = model(batch, args.img_size)
            torch.cuda.synchronize()

            save_lax_prediction(batch, output, args, save_folder_patient)

print("Prediction saved to:", save_root)
