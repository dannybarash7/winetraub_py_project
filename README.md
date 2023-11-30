# repo to debug the colab code using regular python. Weights from sam should be downloaded separately.

mkdir weights  
cd weights  
[ ! -f "./weights/sam_vit_h_4b8939.pth" ] && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
cd ..  
conda deactivate  
conda env create -n vhist_segment --file=requirements.yml    
pip install -r requirements.txt
raw_oct_dataset_dir = "GoogleDrive/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"
conda activate vhist_segment  

update "raw_oct_dataset_dir", in test_iou_of_zero_shot_with_prompts.py, to point to the local oct scans dir.

