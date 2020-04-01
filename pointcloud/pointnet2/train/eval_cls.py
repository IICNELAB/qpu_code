from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import argparse

# from pointnet2.models import Pointnet2ClsMSG as Pointnet
# from pointnet2.models import Pointnet2ClsMSGFC as Pointnet
from pointnet2.models import Pointnet2ClsSSGQPU as Pointnet
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ModelNet40Cls
import pointnet2.data.data_utils as d_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "--rotate", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.rotate)
    trans = [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale()
        ]
    if args.rotate:
        trans.append(d_utils.PointcloudArbRotate())
    trans += [
            # d_utils.PointcloudRotate(),
            # d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout()
        ]
    transforms = transforms.Compose(trans)

    test_set = ModelNet40Cls(args.num_points, transforms=transforms, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Pointnet(input_channels=0, num_classes=40, use_xyz=False)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters()
    )

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer
    )

    loss, stat = trainer.eval_epoch(test_loader)
    acc_per_cls = stat['acc']
    loss_per_cls = stat['loss']
    print('loss:', loss)
    print('Average acc:', torch.mean(torch.Tensor(acc_per_cls)).item())
