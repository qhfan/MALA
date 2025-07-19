_base_ = [
    '../_base_/models/MALA_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
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
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))



optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)

runner = dict(max_iters=80000, work_dir='./')

checkpoint_config = dict(interval=4000)

evaluation = dict(interval=4000)