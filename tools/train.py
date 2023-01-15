# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()

    # 配置文件（一般只用这一个参数就够了）
    parser.add_argument("config", default="../config/nanodet-plus-m_320.yml", help="train config file path")

    # 分布训练相关
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )

    # 随机种子
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    # 将训练参数解析出来
    args = parser.parse_args()
    return args


def main(args):
    # 加载配置文件（yml）， 存储在cfg， 相当于用一个嵌套的字典存储这yml文件中的相关配置， 通过各种build方法， 构建网络训练测试等
    load_config(cfg, args.config)

    # 检查配置文件中的head部分num_classes和类别列表的长度是否一致
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
            "but got {} and {}".format(
                cfg.model.arch.head.num_classes, len(cfg.class_names)
            )
        )

    # 分布式训练相关的参数，一般不需要管（自己学习中没那么大的需求）
    local_rank = int(args.local_rank)

    # 优化运行的效率（不是很清楚），具体可以再详细查阅资料
    torch.backends.cudnn.enabled = True

    # 加速网络训练相关（不是很清楚），具体可以再详细查阅资料
    torch.backends.cudnn.benchmark = True

    # 根据cfg.save_dir生成一个文件夹（存在就不生成）
    mkdir(local_rank, cfg.save_dir)

    # 在cfg.save_dir中 保存训练日志（会看就可以）
    logger = NanoDetLightningLogger(cfg.save_dir)
    # 保存训练的超参数等
    logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    # 通过build_dataset构造训练集
    train_dataset = build_dataset(cfg.data.train, "train")
    # 通过build_dataset构造验证集
    val_dataset = build_dataset(cfg.data.val, "test")

    # 通过build_evaluator评估
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    # 使用torch标准的数据集加载DataLoader加载数据，这里是构造一个训练数据加载的对象
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,                              # 之前定义的数据集的对象（本身并不存储图片，只是存放了各种配置）
        batch_size=cfg.device.batchsize_per_gpu,    # 每个batch包含的样本数
        shuffle=True,                               # 每个epoch，对数据进行重新排序
        num_workers=cfg.device.workers_per_gpu,     # 有几个进程来处理data loading，一个进程处理一个batch
        pin_memory=True,
        collate_fn=naive_collate,                   # 将一个list的sample组成一个样本
        drop_last=True,                             # 将最后不能凑够一个batch的数据扔掉
    )

    # 使用torch标准的数据集加载DataLoader加载数据，这里是构造一个验证数据加载的对象
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,                                # 之前定义的数据集的对象（本身并不存储图片，只是存放了各种配置）
        batch_size=cfg.device.batchsize_per_gpu,    # 每个batch包含的样本数
        shuffle=False,                              # 每个epoch，对数据进行重新排序
        num_workers=cfg.device.workers_per_gpu,     # 有几个进程来处理data loading，一个进程处理一个batch
        pin_memory=True,
        collate_fn=naive_collate,                   # 将一个list的sample组成一个样本
        drop_last=False,                            # 将最后不能凑够一个batch的数据扔掉
    )

    logger.info("Creating model...")
    # 在task中创建了model
    task = TrainingTask(cfg, evaluator)

    # 如果使用了load_model，那么就使用torch.load加载模型
    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)  # 加载模型权重，后面再详细介绍
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    # 模型中断恢复相关 要保存的path
    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )

    # 确定使用的设备id， -1表示使用cpu
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy = "cpu", None, None
    else:
        accelerator, devices, strategy = "gpu", cfg.device.gpu_ids, None

    # 设备大于1，学习时只使用1个gpu，不需要进行分布训练
    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    # 训练
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,              # 模型保存和日志记录默认根路径
        max_epochs=cfg.schedule.total_epochs,       # 最多训练轮数
        check_val_every_n_epoch=cfg.schedule.val_intervals,  # 每几个epoch进行一次验证
        accelerator=accelerator,
        devices=devices,                            # 设备
        log_every_n_steps=cfg.log.interval,         # 更新n次网络权重后记录一次日志
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
    )

    # 开始整个训练
    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    main(args)
