if ! [ -d "../data/imgs" ]
then
    mkdir "../data"
    mkdir "../data/tmp" 

    # download
    wget -P "../data/tmp" "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip"

    # unzip
    unzip "../data/tmp/images.zip" -d "../data/imgs"

    # cleanup

    rm -rf "../data/tmp"
fi

if ! [ -d "../data/ann" ]
then
    mkdir "../data/ann" 

    # download
    wget -P "../data/ann" "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
    wget -P "../data/ann" "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json"
    wget -P "../data/ann" "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
fi