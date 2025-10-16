_base_ = [
    # './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py',
    # './yolox_tta.py'
]
load_from = r'E:/mmdetection320/work_dir_VIDSmoke/yolox_cspDarknet-s-TSM-PTA68_21-12_cocopretrainedMosaic_FIgLib/epoch_10.pth'
resume = False
work_dir = "E:/mmdetection320/work_dir_VIDSmoke/yolox_cspDarknet-s-TSM-PTA68_21-12_cocopretrainedMosaic_FIgLib"
img_scale = (1024, 1024)  # width, height
related_ids=[-2,-1,1,2]
T = len(related_ids)+1
# model settings
model = dict(
    type='YOLOXTSM_PTA',
    data_preprocessor=dict(
        type='DetDataPreprocessorMultiImages',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResizeMultiImages',
                random_size_range=(800, 1280),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='CSPDarknetTSM',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        num_segments=T,
        shift_div=8
    ),
    neck=dict(
        type='YOLOXPAFPN_PTA',
        ptas_cfg=dict(
            in_channels=[128, 256, 512],
            out_channels=[128, 256, 512],
            T =T,
            partial=8,
            left_partial=6,
            group=2
        ),
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'MultiFrameDatasetFIgLib'
backend_args = None

train_pipeline = [
    dict(type='MosaicMultiImages', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffineMultiImages',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)##只有mosaic dataset会设置非0值
    ),
    dict(
        type='MixUpMultiImages',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAugMultiImages'),
    dict(type='RandomFlipMultiImages', prob=0.5),
    dict(type='ResizeMultiImages', scale=img_scale, keep_ratio=True),
    dict(type='PadMultiImages', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='PackDetInputsMultiImages')
]


train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        related_ids=related_ids,
        annotation_files = [
            r"F:\FIgLib\HPWREN-FIgLib-Data\VID_train.json",
        ],
        pipeline =[]),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='ResizeMultiImages', scale=img_scale, keep_ratio=True),
    dict(
        type='PadMultiImages',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='PackDetInputsMultiImages',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))]

train_dataloader = dict(
    batch_size=3,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=4,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        related_ids=related_ids,
        annotation_files=[  r"F:\FIgLib\HPWREN-FIgLib-Data\VID_train.json",],
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ##None的时候会自动生成coco的临时标注文件，但是代码不通##建议线下生成吧
    ann_file= r"F:\ShengpingtaiCW\shengpingtai_trainY23M05_correct\shengpingtai_trainY23M05_correct_coco.json",
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# training settings
max_epochs = 10
num_last_epochs = 4
interval = 1

train_cfg = dict(type='EpochBasedTrainLoop',max_epochs=max_epochs, val_interval=10000000000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
base_lr =0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict( type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=30  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        skip_type_keys= ('MosaicMultiImages', 'RandomAffineMultiImages', 'MixUpMultiImages'),
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
