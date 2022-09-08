# mmlab 代码萤火适配指引

下面以 BEVFormer 为例，介绍 mmlab 代码适配到萤火的关键步骤。

## 适配 1: 启动方式

萤火集群要求启动分布式时 `bind_numa`，因此推荐使用 `torch.multiprocessing.spawn` 启动，不推荐使用 `torch.distributed.launch`。

改动前

```
def main():
    args = parse_args()
    ...

if __name__ == '__main__':
    main()
```

改动后

```
import hfai

def main(local_rank, args):
    ...

if __name__ == '__main__':
    args = parse_args()
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(args,), nprocs=ngpus, bind_numa=True)
```

## 适配 2: 初始化分布式参数
启动萤火任务时，我们通过 --num-nodes 指定节点的数量，每个节点有 8 台 GPU。

萤火集群中的环境变量含义如下：
- `world_size` 代表节点的数量
- `rank` 代表当前节点的 id

因此需要对初始化分布式部分进行改动。

改动前

```
def main(local_rank, args):
    ...
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params) 
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
```

改动后

```
def main(local_rank, args):
    ...
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        # init distributed env first, since logger depends on the dist info.
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "2223")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        torch.cuda.set_device(local_rank)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
```

## 适配 3: 打断继续训练

萤火集群中的任务都会参与分时调度，因此任务需要支持端点续跑。`mmcv.runner` 会自动在每个 epoch 结束保存 checkpoint，因此需要完成下面两步，保证打断后可以继续训练。

1. 训练时指定保存目录 --work-dir <local_relative_path>
2. 启动时增加 --auto-resume 参数


## 适配 4: 保存 config

为了保证集群多个进程的写文件操作只执行一次，需要在写文件前检查 `local_rank` 变量和 `rank` 变量。

改动前

```
def main(local_rank, args):
    ...
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
```

改动后

```
def main(local_rank, args):
    ...
    # dump config
    if local_rank == 0 and rank == 0:
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
```

## 适配 5：转换数据为 FFRecord

转换步骤请参考：[ffrecord_converter](https://github.com/HFAiLab/ffrecord_converters)

## 适配 6：使用 hfai 算子

幻方 AI 对一些常用的 AI 算子进行了重新研发，提升了模型整体训练效率，通过增加如下代码，可以自动替换相应算子。

```
from hfai.nn import to_hfai
def main(local_rank, args):
    ...
    model = to_hfai(model, contiguous_param=False, verbose=True, inplace=True)    
```
注意：在 `batch_size` 较大时，提速明显。



## 常见问题整理
下面整理一些常见的问题，供用户参考。

### 问题 1：cannot pickle 'dict_values' object.

该问题是因为 nuscenes-devkit 中使用了 `dict_values` 的数据类型，导致 `dataloader` 设置 `num_workers` 大于 `0` 时，多进程无法 `pickle dump` 数据集。

修改步骤如下：

1 - 找到 nuscenes-devkit 安装目录 `$nuscenes_devkit_path`：
```
python -c "import nuscenes; print(nuscenes.__file__)"
```

2 - 将 `$nuscenes_devkit_path/eval/detection/data_classes.py` 路径下的 `dict_values` 数据类型进行修改。

修改前
```
self.class_names = self.class_range.keys()
```
修改后
```
self.class_names = list(self.class_range.keys())
```

