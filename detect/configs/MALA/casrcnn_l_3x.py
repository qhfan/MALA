_base_ = [
    '../_base_/models/MALA_cas_rcnn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices = (0, 1, 2, 3),
        embed_dims=[96, 192, 448, 640],
        depths=[4, 7, 19, 8],
        num_heads=[1, 2, 7, 10],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.55,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6]
    ),
    neck=dict(in_channels=[96, 192, 448, 640])
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    train=dict(pipeline=train_pipeline), 
    samples_per_gpu=1,
    workers_per_gpu=1,
)

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(norm_decay_mult=0.,
                     custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 
                                  'relative_position_bias_table': dict(decay_mult=0.),
                                  'rpe_table': dict(decay_mult=0.),
                                  'norm': dict(decay_mult=0.)
                                 }
                                   )
                )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])

fp16 = dict()

resume_from=None

checkpoint_config = dict(max_keep_ckpts=1)
runner = dict(
    work_dir='./',
    max_epochs=36
)