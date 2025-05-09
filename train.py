# # 기존 코드 제거
# # from mmcls.core.hook import HOOKS
# # if 'ClassNumCheckHook' in HOOKS.module_dict:
# #     del HOOKS.module_dict['ClassNumCheckHook']

# # 새로운 코드 추가
# from mmcv.runner import HOOKS
# # HOOKS가 존재하면 ClassNumCheckHook 제거 시도
# try:
#     if 'ClassNumCheckHook' in HOOKS.module_dict:
#         del HOOKS.module_dict['ClassNumCheckHook']
# except:
#     pass  # 오류 발생시 무시하고 계속 진행

# # DistributedSampler 등록
# from torch.utils.data.distributed import DistributedSampler
# from mmcls.datasets.builder import SAMPLERS

# if 'DistributedSampler' not in SAMPLERS.module_dict:
#     SAMPLERS.register_module(module=DistributedSampler, name='DistributedSampler')


import argparse
import copy
import os
import os.path as osp
import time
import warnings
warnings.filterwarnings('ignore')
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from model import __version__
from model.apis import init_random_seed, set_random_seed, train_model
from model.datasets import build_dataset
from model.models import build_classifier
from model.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='train config file path', default="configs/NIH_ChestX-ray14_CTransCNN.py")
    parser.add_argument('--work-dir', help='the dir to save logs and models', default="save")
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training',
        default=False)
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--device', help='device used for training. (Deprecated)', default='cuda')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)',
        default=3)
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,

        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '        
             
        '(only applicable to non-distributed training)',
        default=0)
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--ipu-replicas',
        type=int,
        default=None,
        help='num of ipu replicas to use')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks', default=False)
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.', default=False)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
    # 만약 gpu_ids가 int라면 리스트로 만들어줍니다.
        if isinstance(args.gpu_ids, int):
            cfg.gpu_ids = [args.gpu_ids]
        else:
            cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.ipu_replicas is not None:
        cfg.ipu_replicas = args.ipu_replicas
        args.device = 'ipu'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = args.device or auto_select_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()
    print(cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # save mmcls version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES))

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)


if __name__ == '__main__':
    main()
