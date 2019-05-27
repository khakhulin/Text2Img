#! /bin/bash
curl http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz --output CUB_200_2011.tgz
#wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz CUB_200_2011.tgz CUB_200_2011/images
tar -xvf CUB_200_2011.tgz CUB_200_2011/images.txt
curl -L "https://drive.google.com/uc?export=download&id=1HXnzREyrcYkJiF00rIsyjgA6X3ljPK40" > labels.tar.gz
tar -xvf labels.tar.gz
mkdir -p datasets/CUB_200_2011/
mv CUB_200_2011/images datasets/CUB_200_2011/images
mv CUB_200_2011/images.txt datasets/CUB_200_2011/
mv text_c10 datasets/CUB_200_2011/text_c10
rm -rf CUB_200_2011
python3 scripts/remove_1_channel_img.py --data_path datasets/CUB_200_2011/
sync
