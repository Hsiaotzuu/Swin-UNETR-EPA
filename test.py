#Copyright [2023] [Alkhaleefah]
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import os
import json

import nibabel as nib
import numpy as np
import torch
from networks.Swinunetr import SwinUNETR_EPA
from utils.data_utils import get_loader
from utils.utils import resample_3d, dice, HD
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--pretrained_dir", default="./runs/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--pretrained_model_name",default="model.pt",type=str,help="pretrained model name",)
parser.add_argument("--data_dir", default=" ", type=str, help="dataset directory") 
parser.add_argument("--json_list", default="dataset_1.json", type=str, help="dataset json file")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels") #
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")

parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction") #1.5
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction") #1.5
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction") #2.0

parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction") #(1)96 (2)64
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    args.test_mode = True

    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    print("load path : ", pretrained_pth)
    
    model = SwinUNETR_EPA(
        img_size=96,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    
    model_dict = torch.load(pretrained_pth)
    
    print("keys = " , model_dict.keys())
    model.load_state_dict(model_dict["state_dict"])
    
    model.eval()
    model.to(device)

    pathh = os.path.join(args.data_dir , args.json_list)
    with open(pathh) as f:
        p = json.load(f)
        
        label_key = p["labels"]

    with torch.no_grad():
        dice_list_case = []
        dice_list_file = []
        
        HD_list_case = []
        HD_list_file = []
        
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("\\")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            
            print("target_shape", target_shape)
            print("val_outputs.shape", val_outputs.shape)
            
            val_outputs = resample_3d(val_outputs, target_shape)
            
            save_path = os.path.join("./outputs/", img_name)
            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), save_path)
            print("save output: {}".format(save_path))
            
            
            #===DICE===================================================
            dice_list_sub = []
            for i in range(1, args.out_channels):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
            
            for idx, per_dice_value in enumerate(dice_list_sub):
                print(f"({label_key[str(idx+1)]}) = {per_dice_value:2.5f}")
 
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            print("-----------------------------------------")
            dice_list_file.append(dice_list_sub)
            dice_list_case.append(mean_dice)
            
            #===HD===================================================
            HD_list_sub = []
            for i in range(1, args.out_channels):
                organ_HD = HD(val_outputs == i, val_labels == i)
                HD_list_sub.append(organ_HD)
            
            for idx_2 , per_HD_value in enumerate(HD_list_sub):
                print(f"({label_key[str(idx_2+1)]}) = {per_HD_value:2.5f}")  
                
            mean_HD = np.mean(HD_list_sub)
            print("Mean Organ HD: {}".format(mean_HD))
            print("-----------------------------------------")
            HD_list_file.append(HD_list_sub)
            HD_list_case.append(mean_HD)
             
        print("-----------------------------------------")
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean HD: {}".format(np.mean(HD_list_case)))

        print("Avg Dice value for per organ :")
        buf = np.mean(dice_list_file , axis =0)
        for idx , avg_dice_value in enumerate(buf):
            print(f"{avg_dice_value:2.4f} ({label_key[str(idx+1)]})")
            
        print("Avg HD value for per organ :")
        buf_2 = np.mean(HD_list_file , axis =0)
        for idx , avg_HD_value in enumerate(buf_2):
            print(f"{avg_HD_value:2.4f} ({label_key[str(idx+1)]})")
        print("-------------------END------------------") 



if __name__ == "__main__":
    main()


