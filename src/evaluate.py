import os
import logging
import PIL
import torch
import time
from tqdm import tqdm

from src.ddp import distrib
from src.data.datasets import match_signal
from src.enhance import save_wavs, save_specs
from src.metrics import run_metrics
from src.utils import LogProgress, bold

# from src.wandb_logger import log_data_to_wandb
from src.models.spec import spectro
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def evaluate_lr_hr_pr_data(data, wandb_n_files_to_log, files_to_log, epoch, args):
    lr, hr, pr, filename = data
    filename = filename[0]
    hr_sr = args.experiment.hr_sr if "experiment" in args else args.hr_sr
    if args.experiment.upsample:
        lr_sr = hr_sr
    else:
        lr_sr = args.experiment.lr_sr if "experiment" in args else args.lr_sr

    if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
        files_to_log.append(filename)

    if args.device != "cpu":
        hr = hr.cpu()
        pr = pr.cpu()

    # hr_spec_path = os.path.join(args.samples_dir, filename + '_hr_spec.png')
    # pr_spec_path = os.path.join(args.samples_dir, filename + '_pr_spec.png')
    # lr_spec_path = os.path.join(args.samples_dir, filename + '_lr_spec.png')

    # hr_spec = PIL.Image.open(hr_spec_path) if os.path.exists(hr_spec_path) else None
    # pr_spec = PIL.Image.open(pr_spec_path) if os.path.exists(pr_spec_path) else None
    # lr_spec = PIL.Image.open(lr_spec_path) if os.path.exists(lr_spec_path) else None

    lsd_i, lsd_hf_i, lsd_lf_i, visqol_i = run_metrics(hr, pr, args, filename)

    return {"lsd": lsd_i, "visqol": visqol_i, "filename": filename}


from pathlib import Path

"""
This is for saving intermediate spectrogram output as well as final time signal output of model.
"""


def evaluate_lr_hr_data(
    data, model, wandb_n_files_to_log, files_to_log, epoch, args, enhance=True
):
    (lr, lr_path), (hr, hr_path) = data
    lr, hr = lr.to(args.device), hr.to(args.device)
    hr_sr = args.experiment.hr_sr if "experiment" in args else args.hr_sr
    if args.experiment.upsample:
        lr_sr = hr_sr
    else:
        lr_sr = args.experiment.lr_sr if "experiment" in args else args.lr_sr
    model.eval()
    start_time = time.time()
    if args.experiment.model == "aero":
        with torch.no_grad():
            pr, pr_spec, lr_spec = model(lr, return_spec=True, return_lr_spec=True)
        pr = match_signal(pr, hr.shape[-1])
        hr_spec = model._spec(hr, scale=True)
    else:
        nfft = args.experiment.nfft
        win_length = nfft // 4
        pr = model(lr)
        pr_spec = spectro(pr, n_fft=nfft, win_length=win_length)
        lr_spec = spectro(lr, n_fft=nfft, win_length=win_length)
        hr_spec = spectro(hr, n_fft=nfft, win_length=win_length)
    run_time = time.time() - start_time
    model.train()
    filename = Path(hr_path[0]).stem

    if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
        files_to_log.append(filename)

    if args.device != "cpu":
        hr = hr.cpu()
        pr = pr.cpu()
        lr = lr.cpu()

    lsd_i, lsd_hf_i, lsd_lf_i, visqol_i = run_metrics(hr, pr, args, filename)
    base_lsd_i, base_lsd_hf_i, base_lsd_lf_i, base_visqol_i = run_metrics(
        hr,
        torch.tensor(resample_poly(lr, hr_sr, lr_sr, axis=-1)),
        args,
        filename,
    )
    rtf = run_time / hr.shape[-1] * hr_sr

    if enhance:
        output_dir = (
            f"../../../test_samples/{args.experiment.hr_sr}/{args.experiment.lr_sr}"
        )
        os.makedirs(output_dir, exist_ok=True)
        lr_sr = (
            args.experiment.hr_sr if args.experiment.upsample else args.experiment.lr_sr
        )
        save_wavs(
            pr,
            lr,
            hr,
            [os.path.join(output_dir, filename)],
            lr_sr,
            args.experiment.hr_sr,
        )

    return {
        "lsd": lsd_i,
        "lsd_hf": lsd_hf_i,
        "lsd_lf": lsd_lf_i,
        "visqol": visqol_i,
        "base_lsd": base_lsd_i,
        "base_lsd_hf": base_lsd_hf_i,
        "base_lsd_lf": base_lsd_lf_i,
        "base_visqol": base_visqol_i,
        "rtf": rtf,
        "rtf_reciprocal": 1 / rtf,
        "filename": filename,
    }


def evaluate_on_saved_data(args, data_loader, epoch):

    total_lsd = 0
    total_visqol = 0

    lsd_count = 0
    visqol_count = 0

    total_cnt = 0

    files_to_log = []
    wandb_n_files_to_log = (
        args.wandb.n_files_to_log if "wandb" in args else args.wandb_n_files_to_log
    )

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):
            metrics_i = evaluate_lr_hr_pr_data(
                data, wandb_n_files_to_log, files_to_log, epoch, args
            )

            total_lsd += metrics_i["lsd"]
            total_visqol += metrics_i["visqol"]

            lsd_count += 1 if metrics_i["lsd"] != 0 else 0
            visqol_count += 1 if metrics_i["visqol"] != 0 else 0

            total_cnt += 1

    if lsd_count != 0:
        (avg_lsd,) = [total_lsd / lsd_count]
    else:
        avg_lsd = 0

    if visqol_count != 0:
        (avg_visqol,) = [total_visqol / visqol_count]
    else:
        avg_visqol = 0

    logger.info(
        bold(
            f"{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:"
            f"LSD={avg_lsd} ({lsd_count}/{total_cnt}), VISQOL={avg_visqol} ({visqol_count}/{total_cnt})."
        )
    )

    return avg_lsd, avg_visqol


def evaluate(args, data_loader, epoch, model):
    lsd_list = []
    lsd_hf_list = []
    lsd_lf_list = []
    visqol_list = []
    base_lsd_list = []
    base_lsd_hf_list = []
    base_lsd_lf_list = []
    base_visqol_list = []
    rtf_list = []
    rtf_reciprocal_list = []
    filenames_list = []

    files_to_log = []
    wandb_n_files_to_log = (
        args.wandb.n_files_to_log if "wandb" in args else args.wandb_n_files_to_log
    )

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(tqdm(data_loader)):

            metrics_i = evaluate_lr_hr_data(
                data, model, wandb_n_files_to_log, files_to_log, epoch, args
            )

            lsd_list.append(metrics_i["lsd"])
            lsd_hf_list.append(metrics_i["lsd_hf"])
            lsd_lf_list.append(metrics_i["lsd_lf"])
            visqol_list.append(metrics_i["visqol"])
            base_lsd_list.append(metrics_i["base_lsd"])
            base_lsd_hf_list.append(metrics_i["base_lsd_hf"])
            base_lsd_lf_list.append(metrics_i["base_lsd_lf"])
            base_visqol_list.append(metrics_i["base_visqol"])
            rtf_list.append(metrics_i["rtf"])
            rtf_reciprocal_list.append(metrics_i["rtf_reciprocal"])

            filenames_list.append(metrics_i["filename"])

            lsd_count = len(lsd_list)
            visqol_count = len(visqol_list)

            total_cnt = lsd_count

    if lsd_count != 0:
        avg_lsd = torch.stack(lsd_list, dim=0).mean()
        avg_base_lsd = torch.stack(base_lsd_list, dim=0).mean()
        avg_lsd_hf = torch.stack(lsd_hf_list, dim=0).mean()
        avg_base_lsd_hf = torch.stack(base_lsd_hf_list, dim=0).mean()
        avg_lsd_lf = torch.stack(lsd_lf_list, dim=0).mean()
        avg_base_lsd_lf = torch.stack(base_lsd_lf_list, dim=0).mean()
        avg_rtf = torch.tensor(rtf_list).mean()
        avg_rtf_reciprocal = 1 / avg_rtf
    else:
        avg_lsd = 0
    if visqol_count != 0:
        avg_visqol = sum(visqol_list) / visqol_count
        avg_base_visqol = sum(base_visqol_list) / visqol_count
    else:
        avg_visqol = 0
        avg_base_visqol = 0

    return (
        avg_lsd,
        avg_lsd_hf,
        avg_lsd_lf,
        avg_visqol,
        avg_base_lsd,
        avg_base_lsd_hf,
        avg_base_lsd_lf,
        avg_base_visqol,
        avg_rtf,
        avg_rtf_reciprocal,
        filenames_list,
    )
