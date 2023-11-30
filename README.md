# repo to debug the colab code using regular python. Weights from sam should be downloaded separately.

cd weights
[ ! -f "./weights/sam_vit_h_4b8939.pth" ] && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
!pip install git+https://github.com/facebookresearch/segment-anything.git git+https://github.com/fsemerar/segment-anything-gui.git -q

conda env create -n vhist_segment --file=file.yml

raw_oct_dataset_dir = "GoogleDrive/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"

