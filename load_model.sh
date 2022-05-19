# download parameter files from https://github.com/microsoft/Swin-Transformer
# map22kto1k.txt is in ./data/
# swin_large_patch4_window7_224_22k.yaml is in ./configs/
# swin_large_patch4_window7_224_22k.pth is in https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
mkdir swin_para
# move three files into swin_para

# download parameter files for ImageNet from website https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md
# download ymal config files from https://github.com/facebookresearch/SlowFast/blob/main/configs/ImageNet/MVIT_B_16_CONV.yaml
mkdir mvit_para
# move two files into mvit_para