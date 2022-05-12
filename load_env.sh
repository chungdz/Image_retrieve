# 加载环境
cd /mnt
cp myconda.tar.gz ~/
cd ~
mkdir -p my_env
tar -xzf myconda.tar.gz -C my_env
source ~/my_env/bin/activate
# 加载代码和数据
cd /mnt
cp -r Image_retrieve/ ~/
#加载 github钥匙
cd /mnt/ssh_files/
cp * ~/.ssh/
ssh -T git@github.com
# 加载ResNet的pretrain的参数
cd /mnt
cp resnext101_32x8d-8ba56ff5.pth /root/.cache/torch/hub/checkpoints/


# 保存
cp -r Image_retrieve/ /mnt/Image_retrieve/
# check https://conda.github.io/conda-pack/ to pack conda