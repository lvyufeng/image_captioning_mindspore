CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/data

if [ ! -d $DATA_PATH ]
then
    mkdir ${CUR_DIR}/data
fi

cd ${CUR_DIR}/data

# karpathy splits
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -n -d splits caption_datasets.zip
# train dataset
wget http://images.cocodataset.org/zips/train2014.zip
unzip -n -d coco train2014.zip
# val dataset
wget http://images.cocodataset.org/zips/val2014.zip
unzip -n -d coco val2014.zip