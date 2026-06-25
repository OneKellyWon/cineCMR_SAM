#!/usr/bin/env python

import os
import sys

sys.path.append("/gpfs/work/aac/yifansun2302")

import numpy as np
import pandas as pd
from madgrad import MADGRAD

import torch
import torch.backends.cudnn as cudnn

from cineCMR_SAM.segment_anything.model import build_model
from cineCMR_SAM.train_engine import train_loop
from cineCMR_SAM.utils.config_util import Config
from cineCMR_SAM.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from cineCMR_SAM.utils.model_util import load_from

import cineCMR_SAM.dataset.build_CMR_datasets as build_CMR_datasets
import cineCMR_SAM.functions_collection as ff
import cineCMR_SAM.get_args_parser as get_args_parser


main_path = "/gpfs/work/aac/yifansun2302/cineCMR_SAM"

trial_name = "mnm2_lax_3class_t15_debug_train5_e2"
output_dir = os.path.join(main_path, "example_data/models", trial_name)
ff.make_folder([os.path.join(main_path, "example_data/models"), output_dir])

text_prompt = False
box_prompt = False
pretrained_model = None
start_epoch = 1
total_training_epochs = 2

lax_text_prompt_feature = np.load(
    os.path.join(main_path, "example_data/data/text_prompt_clip/lax.npy")
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
    start_epoch=start_epoch,
    total_training_epochs=total_training_epochs,
    vit_type="vit_h",
)
args = args.parse_args([])
args.num_classes = 4
args.max_timeframe = 15
args.turn_zero_seg_slice_into = 10
args.accum_iter = 2
args.print_freq = 1

cfg = Config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

patient_list_file_lax = os.path.join(
    main_path,
    "example_data/data/Patient_list/patient_list_mnm2_lax_t15_debug_train5.xlsx",
)
patient_index_list = np.arange(0, 5, 1)

dataset_train_lax = build_CMR_datasets.build_dataset(
    args,
    view_type="lax",
    patient_list_file=patient_list_file_lax,
    index_list=patient_index_list,
    text_prompt_feature=lax_text_prompt_feature,
    only_myo=False,
    shuffle=True,
    augment=True,
)

dataset_train = [dataset_train_lax]
data_loader_train = [
    torch.utils.data.DataLoader(
        dataset_train_lax,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
]

model = build_model(args, device)

if args.model_type.startswith("sam"):
    if args.resume.endswith(".pth"):
        with open(args.resume, "rb") as f:
            state_dict = torch.load(f)
        try:
            model.load_state_dict(state_dict)
        except Exception:
            new_state_dict = load_from(model, state_dict, args.img_size, 16, [7, 15, 23, 31])
            model.load_state_dict(new_state_dict)

        freeze_list = ["norm1", "attn", "mlp", "norm2"]
        for name, value in model.named_parameters():
            value.requires_grad = not any(substring in name for substring in freeze_list)

optimizer = MADGRAD(model.parameters(), lr=args.lr)

print("new debug training")
print("trial:", trial_name)
print("patient list:", patient_list_file_lax)
print("epochs:", args.total_training_epochs)
print("accum_iter:", args.accum_iter)
print("max_timeframe:", args.max_timeframe)
print("num_classes:", args.num_classes)

training_log = []
model_save_folder = os.path.join(output_dir, "models")
log_save_folder = os.path.join(output_dir, "logs")
ff.make_folder([model_save_folder, log_save_folder])

for epoch in range(args.start_epoch, args.start_epoch + args.total_training_epochs):
    print("training epoch:", epoch)
    print("learning rate now:", optimizer.param_groups[0]["lr"])

    loss_scaler = NativeScaler()

    train_results = train_loop(
        model=model,
        data_loader_train=data_loader_train,
        optimizer=optimizer,
        epoch=epoch,
        loss_scaler=loss_scaler,
        args=args,
        inputtype=cfg.data.input_type,
    )

    loss, lossCE, lossDICE, sax_loss, sax_lossCE, sax_lossDICE, lax_loss, lax_lossCE, lax_lossDICE = train_results

    print(
        "in epoch:",
        epoch,
        "training average_loss:",
        loss,
        "average_lossCE:",
        lossCE,
        "average_lossDICE:",
        lossDICE,
        "lax_loss:",
        lax_loss,
        "lax_lossCE:",
        lax_lossCE,
        "lax_lossDICE:",
        lax_lossDICE,
    )

    for dataset in dataset_train:
        dataset.on_epoch_end()

    checkpoint_path = os.path.join(model_save_folder, f"model-{epoch}.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": loss_scaler.state_dict(),
            "args": args,
        },
        checkpoint_path,
    )

    training_log.append([
        epoch,
        optimizer.param_groups[0]["lr"],
        train_results[0],
        train_results[1],
        train_results[2],
        train_results[3],
        train_results[4],
        train_results[5],
        train_results[6],
        train_results[7],
        train_results[8],
    ])
    df = pd.DataFrame(
        training_log,
        columns=[
            "epoch",
            "lr",
            "average_loss",
            "average_lossCE",
            "average_lossDICE",
            "sax_loss",
            "sax_lossCE",
            "sax_lossDICE",
            "lax_loss",
            "lax_lossCE",
            "lax_lossDICE",
        ],
    )
    df.to_excel(os.path.join(log_save_folder, "training_log.xlsx"), index=False)

print("debug training finished")
print("output_dir:", output_dir)
