import os
import json
from datetime import datetime
import torch

def get_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    cuda_version = torch.version.cuda if cuda_available else None
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "device": str(device),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name
    }

def update_run_info(run_id, config, start_time, end_time=None, final_train_loss=None, final_val_loss=None, error=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diffusion_runs_folder = os.path.join(current_dir, '..', 'diffusion_runs')
    run_folder = os.path.join(diffusion_runs_folder, run_id)
    run_info_path = os.path.join(run_folder, 'run_info.json')

    run_info = {
        "run_id": run_id,
        "config": {
            "dataset": config.dataset.__dict__,
            "model": config.model.__dict__,
            "trainer": config.trainer.__dict__
        },
        "start_time": start_time.isoformat(),
        "device_info": get_device_info(),
    }

    if end_time:
        run_info["end_time"] = end_time.isoformat()
        run_info["total_time"] = (end_time - start_time).total_seconds()

    if final_train_loss is not None:
        run_info["final_train_loss"] = final_train_loss

    if final_val_loss is not None:
        run_info["final_val_loss"] = final_val_loss

    if error:
        run_info["error"] = str(error)

    os.makedirs(os.path.dirname(run_info_path), exist_ok=True)
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=4)

def update_all_runs(run_id, run_info):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diffusion_runs_folder = os.path.join(current_dir, '..', 'diffusion_runs')
    all_runs_path = os.path.join(diffusion_runs_folder, 'all_runs.json')

    if os.path.exists(all_runs_path):
        with open(all_runs_path, 'r') as f:
            all_runs = json.load(f)
    else:
        all_runs = {}

    all_runs[run_id] = run_info

    with open(all_runs_path, 'w') as f:
        json.dump(all_runs, f, indent=4)

def update_finished_runs(run_id, run_info):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diffusion_runs_folder = os.path.join(current_dir, '..', 'diffusion_runs')
    finished_runs_path = os.path.join(diffusion_runs_folder, 'finished_runs.json')

    if os.path.exists(finished_runs_path):
        with open(finished_runs_path, 'r') as f:
            finished_runs = json.load(f)
    else:
        finished_runs = {}

    finished_runs[run_id] = {
        "config": run_info["config"],
        "start_time": run_info["start_time"],
        "end_time": run_info["end_time"],
        "total_time": run_info["total_time"],
        "final_train_loss": run_info["final_train_loss"],
        "final_val_loss": run_info["final_val_loss"],
        "device_info": run_info["device_info"],
        "best_checkpoint": {
            "epoch": run_info["best_checkpoint"]["epoch"],
            "loss": run_info["best_checkpoint"]["loss"],
            "path": run_info["best_checkpoint"]["path"]
        },
        "latest_checkpoint": {
            "epoch": run_info["latest_checkpoint"]["epoch"],
            "loss": run_info["latest_checkpoint"]["loss"],
            "path": run_info["latest_checkpoint"]["path"]
        }
    }

    with open(finished_runs_path, 'w') as f:
        json.dump(finished_runs, f, indent=4)
