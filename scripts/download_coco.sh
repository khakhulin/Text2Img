#!/usr/bin/env bash
mkdir coco
cd coco
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip


unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip

cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
unzip image_info_test2017.zip

rm annotations_trainval2017.zip
rm image_info_test2017.zip#!/usr/bin/env bash