_base_ = [
    '../_base_/models/MALA_retinanet.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
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
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 448, 640],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.0)
find_unused_parameters=True
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