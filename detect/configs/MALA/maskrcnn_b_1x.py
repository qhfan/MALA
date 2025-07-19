_base_ = [
    '../_base_/models/MALA_rcnn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        out_indices = (0, 1, 2, 3),
        embed_dims=[96, 192, 384, 512],
        depths=[4, 6, 12, 6],
        num_heads=[1, 2, 6, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.4,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1e-6, 1e-6]
    ),
    neck = dict(in_channels=[96, 192, 384, 512])
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])

# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# # do not use mmdet version fp16 -> WHY?
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

fp16 = dict()
###########################################################################################################

# place holder for new verison mmdet compatiability
resume_from=None

# custom
checkpoint_config = dict(max_keep_ckpts=1)
runner = dict(work_dir='./')


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)