# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
import atexit
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder, seed_all, seed_worker
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch


def configure_torch_threads():
    torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "0"))
    torch_interop_threads = int(os.environ.get("TORCH_INTEROP_THREADS", "0"))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    if torch_interop_threads > 0:
        torch.set_num_interop_threads(torch_interop_threads)
    return torch_threads, torch_interop_threads


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='aitod_v2') 
    parser.add_argument('--coco_path', type=str, default='/path/to/dataset')
    parser.add_argument('--coco_panoptic_path', type=str) 
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_bbox_json', action='store_true',
                        help='save COCO-style bbox detections to JSON during evaluation')
    parser.add_argument('--bbox_json_name', default='bbox_results.json', type=str,
                        help='filename used when --save_bbox_json is enabled')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        help='local rank for DistributedDataParallel',
    )
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    torch_threads, torch_interop_threads = configure_torch_threads()
    if torch_threads > 0 or torch_interop_threads > 0:
        print(
            f"CPU thread limits: torch={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}"
        )
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    #print("Loading config file from {}".format(args.config_file))
    
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        print(k,v)
        if k not in args_vars:  
            setattr(args, k, v)
        else:
            print("ex:",k,v)
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    # Optional Weights & Biases logging (rank0 only).
    # Safe-by-default: if wandb is missing or any init/log step fails, training continues without wandb.
    wandb = None
    wandb_run = None
    wandb_active = bool(getattr(args, 'wandb_enable', False)) and utils.is_main_process()

    def _wandb_disable(msg: str) -> None:
        nonlocal wandb_active
        wandb_active = False
        try:
            logger.warning(msg)
        except Exception:
            print(msg)

    def _wandb_scalar_dict(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else str(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                out[key] = float(v.item())
        return out

    def safe_wandb_log(d: dict, step: int = None, prefix: str = "") -> None:
        nonlocal wandb_active
        if not wandb_active or wandb is None:
            return
        payload = _wandb_scalar_dict(d, prefix=prefix)
        if not payload:
            return
        try:
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=int(step))
        except Exception as e:
            _wandb_disable(f"[wandb] log failed ({e}); disabling wandb.")

    def safe_wandb_finish() -> None:
        nonlocal wandb_active
        if not wandb_active or wandb is None:
            return
        try:
            wandb.finish()
        except Exception:
            pass
        wandb_active = False

    _COCO_BBOX_STAT_NAMES = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    )

    def safe_wandb_log_coco_bbox(stats: dict, step: int = None, prefix: str = "test/") -> None:
        if not isinstance(stats, dict):
            return
        coco = stats.get("coco_eval_bbox", None)
        if not isinstance(coco, (list, tuple)):
            return
        for name, val in zip(_COCO_BBOX_STAT_NAMES, coco):
            safe_wandb_log({f"coco_bbox/{name}": val}, step=step, prefix=prefix)

    if wandb_active:
        try:
            import wandb as _wandb  # optional dependency

            wandb = _wandb
            wandb_dir = str(getattr(args, "wandb_dir", args.output_dir))
            wandb_project = str(getattr(args, "wandb_project", "DQ-DETR"))
            wandb_entity = getattr(args, "wandb_entity", None)
            wandb_name = getattr(args, "wandb_name", None)
            wandb_group = getattr(args, "wandb_group", None)
            wandb_tags = getattr(args, "wandb_tags", None)
            wandb_mode = str(getattr(args, "wandb_mode", "online"))
            if not wandb_name:
                wandb_name = os.path.basename(os.path.abspath(args.output_dir))

            init_kwargs = {
                "project": wandb_project,
                "dir": wandb_dir,
                "name": str(wandb_name),
                "config": vars(args),
                "mode": wandb_mode,
                "settings": wandb.Settings(start_method="thread"),
            }
            if wandb_entity:
                init_kwargs["entity"] = str(wandb_entity)
            if wandb_group:
                init_kwargs["group"] = str(wandb_group)
            tags_list = []
            if isinstance(wandb_tags, (list, tuple)):
                tags_list = [str(t) for t in wandb_tags if str(t)]
            elif isinstance(wandb_tags, str) and wandb_tags:
                tags_list = [wandb_tags]
            if tags_list:
                init_kwargs["tags"] = tags_list

            wandb_run = wandb.init(**init_kwargs)
            atexit.register(safe_wandb_finish)
            logger.info(
                f"[wandb] enabled: project={wandb_project}, name={wandb_name}, mode={wandb_mode}"
            )
        except Exception as e:
            _wandb_disable(f"[wandb] init failed ({e}); continue without wandb.")
            wandb = None
            wandb_run = None


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    #print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    seed_all(seed, deterministic=True)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    dataset_train = build_dataset(image_set='trainval', args=args)
    dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    _g = torch.Generator()
    _g.manual_seed(seed)
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=_g,
    )
    data_loader_val = DataLoader(
        dataset_val,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=_g,
    )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict

        # NOTE: torch.nn.Module.load_state_dict(strict=False) still errors on shape mismatch.
        # For common finetune setups (e.g., different num_classes), we drop keys whose tensor
        # shapes do not match the current model.
        ignore_keywords = args.finetune_ignore if args.finetune_ignore else []
        ignored_by_keyword = []
        dropped_not_in_model = []
        dropped_shape_mismatch = []

        model_state = model_without_ddp.state_dict()
        cleaned_ckpt = utils.clean_state_dict(checkpoint)
        filtered_ckpt = OrderedDict()
        for k, v in cleaned_ckpt.items():
            if any(kw in k for kw in ignore_keywords):
                ignored_by_keyword.append(k)
                continue
            if k not in model_state:
                dropped_not_in_model.append(k)
                continue
            if hasattr(v, 'shape') and v.shape != model_state[k].shape:
                dropped_shape_mismatch.append({
                    'key': k,
                    'ckpt_shape': list(v.shape),
                    'model_shape': list(model_state[k].shape),
                })
                continue
            filtered_ckpt[k] = v

        if ignored_by_keyword:
            logger.info("Ignore keys by keyword: {}".format(json.dumps(ignored_by_keyword, indent=2)))
        if dropped_shape_mismatch:
            logger.warning(
                "Drop keys due to shape mismatch (count={}):\n{}".format(
                    len(dropped_shape_mismatch),
                    json.dumps(dropped_shape_mismatch[:50], indent=2),
                )
            )
        if dropped_not_in_model:
            logger.info("Drop keys not in model (count={}): {}".format(len(dropped_not_in_model), dropped_not_in_model[:50]))

        _load_output = model_without_ddp.load_state_dict(filtered_ckpt, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        safe_wandb_log(train_stats, step=epoch, prefix="train/")
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        if epoch in args.val_epoch:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            safe_wandb_log(test_stats, step=epoch, prefix="test/")
            safe_wandb_log_coco_bbox(test_stats, step=epoch, prefix="test/")
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }

            # eval ema
            if args.use_ema:
                ema_test_stats, ema_coco_evaluator = evaluate(
                    ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                )
                safe_wandb_log(ema_test_stats, step=epoch, prefix="ema_test/")
                safe_wandb_log_coco_bbox(ema_test_stats, step=epoch, prefix="ema_test/")
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                map_ema = ema_test_stats['coco_eval_bbox'][0]
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)

            log_stats.update(best_map_holder.summary())
            safe_wandb_log(best_map_holder.summary(), step=epoch, prefix="best/")
        
            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
        
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
    
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
