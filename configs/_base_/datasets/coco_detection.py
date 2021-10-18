_base_ = './pipelines/_coco.py'

DATASET_TYPE = 'CocoDataset'
DATA_ROOT = '/data/coco/'
data = dict(
    train=dict(
        type=DATASET_TYPE,
        ann_file=DATA_ROOT + 'annotations/instances_train2017.json',
        img_prefix=DATA_ROOT + 'images/train2017/'),
    val=dict(
        type=DATASET_TYPE,
        ann_file=DATA_ROOT + 'annotations/instances_val2017.json',
        img_prefix=DATA_ROOT + 'images/val2017/'),
    test=dict(
        type=DATASET_TYPE,
        # minival
        ann_file=DATA_ROOT + 'annotations/instances_val2017.json',
        img_prefix=DATA_ROOT + 'images/val2017/',
        # testdev
        # ann_file=DATA_ROOT + 'annotations/image_info_test-dev2017.json',
        # img_prefix=DATA_ROOT + 'images/test2017/',
        test_mode=True))
