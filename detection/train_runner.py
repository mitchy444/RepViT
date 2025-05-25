import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

import repvit

print(__file__)

config = osp.join(
            osp.dirname(osp.abspath(__file__)),
            "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco_person_ben.py",
        )
cfg = Config.fromfile(config)
cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config))[0])

runner = Runner.from_cfg(cfg)
runner.build_dataloader(runner._train_dataloader)