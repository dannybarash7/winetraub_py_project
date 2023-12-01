# repo to debug the colab code using regular python. Weights from sam should be downloaded separately.

mkdir weights  
cd weights  
[ ! -f "./weights/sam_vit_h_4b8939.pth" ] && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
cd ..  
conda deactivate  
conda env create -n vhist_segment --file=requirements.yml 
conda activate vhist_segment  
pip3 install -r requirements.txt  
raw_oct_dataset_dir = "GoogleDrive/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"  

update "raw_oct_dataset_dir", in test_iou_of_zero_shot_with_prompts.py, to point to the local oct scans dir.  
cd winetraub_py_project-main  
from root dir, run test_iou_of_zero_shot_with_prompts.py  
checkpoints/oct2hist/latest_net_G.pth

from https://drive.google.com/drive/folders/1cYoZcqYwPxGQ3pQBHTK5JYIUMsgrYMDb?usp=drive_link
download latest_net_G.pth  
into 
winetraub_py_project-main/OCT2Hist_UseModel/pytorch_CycleGAN_and_pix2pix/checkpoints/oct2hist/latest_net_G.pth

