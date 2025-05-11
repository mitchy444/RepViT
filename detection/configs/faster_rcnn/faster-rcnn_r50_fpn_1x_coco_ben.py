_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn_ben.py",
    "../_base_/datasets/coco_detection_ben.py",
    "../_base_/schedules/schedule_1x_ben.py",
    "../_base_/default_runtime_ben.py",
]
model = dict(
    backbone=dict(
        type="repvit_m1_1",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="pretrain/repvit_m1_1_distill_300e.pth",
        ),
        out_indices=[2, 6, 20, 24],
    ),
    neck=dict(
        type="FPN", in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5
    ),
)
