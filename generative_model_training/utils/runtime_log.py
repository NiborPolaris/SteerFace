# runtime_log.py
import os, socket, datetime as dt, subprocess, platform
from pathlib import Path
from omegaconf import OmegaConf, open_dict
import numpy as np

# 优先用 importlib.metadata 拿到 **字符串** 版本
try:
    import importlib.metadata as ilmeta    # Py3.8+
except Exception:
    import importlib_metadata as ilmeta    # backport

def _to_primitives(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.bool_,)):      return bool(x)
    if isinstance(x, (np.integer,)):    return int(x)
    if isinstance(x, (np.floating,)):   return float(x)
    if isinstance(x, (Path, dt.date, dt.datetime)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_primitives(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_primitives(v) for v in x]
    return str(x)

def _pkg_version(dist_name: str, fallback="NA"):
    try:
        return ilmeta.version(dist_name)  # 永远返回 str
    except Exception:
        return fallback

def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "NA"

def detect_parallel_info(torch):
    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    return {
        "ddp_initialized": bool(ddp),
        "backend": (torch.distributed.get_backend() if ddp else "none"),
        "world_size": int(torch.distributed.get_world_size() if ddp else 1),
        # "rank": int(torch.distributed.get_rank() if ddp else 0),
        # "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        # "node_rank": int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", -1))),
        "n_gpus_visible": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
    }

def inject_runtime(cfg):
    import torch  # 放在函数内，避免其他模块提前污染
    # 1) 用包元数据拿 **字符串** 版本，彻底绕开 TorchVersion
    torch_ver = _pkg_version("torch", getattr(torch, "__version__", "NA"))
    torchaudio_ver = _pkg_version("torchaudio", "NA")
    torchvision_ver = _pkg_version("torchvision", "NA")

    run_meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "workdir": os.getcwd(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch_ver,                    # 确保是 str
        "torchvision": torchvision_ver,        # 可选
        "torchaudio": torchaudio_ver,          # 可选
        "cuda": getattr(torch.version, "cuda", "NA"),
        "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "NA"),
    }

    # 2) 防御：无论外部是否注入过对象版本，这里都**强制覆盖**成字符串
    run_meta.pop("TorchVersion", None)

    # 3) 构建 payload 前，把所有东西做递归原生化
    payload = _to_primitives({
        # "checkpoint": OmegaConf.to_container(cfg.get("checkpoint", {}), resolve=True),
        # "sampling":   OmegaConf.to_container(cfg.get("sampling",   {}), resolve=True),
        "parallel":   detect_parallel_info(torch),
        "run_meta":   run_meta,
    })

    # 4) 最后再做一次强校验，防止任何非原生混进来
    assert isinstance(payload["run_meta"]["torch"], str), type(payload["run_meta"]["torch"])

    with open_dict(cfg):
        cfg.runtime = OmegaConf.create(payload)
