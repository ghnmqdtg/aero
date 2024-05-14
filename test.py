"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""

import logging
import os
import csv

import torch
from pathlib import Path
import hydra

from src.data.datasets import LrHrSet
from src.ddp import distrib
from src.evaluate import evaluate
from src.models import modelFactory
from src.utils import bold

logger = logging.getLogger(__name__)

SERIALIZE_KEY_MODELS = "models"
SERIALIZE_KEY_BEST_STATES = "best_states"
SERIALIZE_KEY_STATE = "state"


def save_results_to_csv(results, filename="results.csv"):
    # Change directory to root of the project
    os.chdir(hydra.utils.get_original_cwd())
    # Check if file exists. If not, create it and write headers
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Sample Rate",
                    "LSD (mean)",
                    "LSD (std)",
                    "Base LSD (mean)",
                    "Base LSD (std)",
                    "LSD HF (mean)",
                    "LSD HF (std)",
                    "Base LSD HF (mean)",
                    "Base LSD HF (std)",
                    "LSD LF (mean)",
                    "LSD LF (std)",
                    "Base LSD LF (mean)",
                    "Base LSD LF (std)",
                    "RTF (mean)",
                    "RTF (std)",
                    "RTF Reciprocal (mean)",
                    "RTF Reciprocal (std)",
                ]
            )

        writer.writerow(results)
    logger.info(f"Results saved to {filename}")


def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)["generator"]
    package = torch.load(checkpoint_file, "cpu")
    load_best = args.continue_best
    if load_best:
        logger.info(bold(f"Loading model {model_name} from best state."))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]["generator"][
                SERIALIZE_KEY_STATE
            ]
        )
    else:
        logger.info(bold(f"Loading model {model_name} from last state."))
        model.load_state_dict(
            package[SERIALIZE_KEY_MODELS]["generator"][SERIALIZE_KEY_STATE]
        )

    return model


def run(args):
    results = []

    tt_dataset = LrHrSet(
        args.dset.test,
        args.experiment.lr_sr,
        args.experiment.hr_sr,
        stride=None,
        segment=None,
        with_path=True,
        upsample=args.experiment.upsample,
    )
    tt_loader = distrib.loader(
        tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    model = _load_model(args)
    model.cuda()

    print(model.flops(tt_loader))

    (
        lsd,
        lsd_hf,
        lsd_lf,
        visqol,
        base_lsd,
        base_lsd_hf,
        base_lsd_lf,
        base_visqol,
        rtf,
        rtf_reciprocal,
        enhanced_filenames,
    ) = evaluate(args, tt_loader, 0, model)
    logger.info(f"Done evaluation.")
    logger.info(
        f"LSD={lsd}, BASE LSD={base_lsd}, LSD-HF={lsd_hf}, BASE LSD-HF={base_lsd_hf}, LSD-LF={lsd_lf}, BASE LSD-LF={base_lsd_lf}, VISQOL={visqol}, RTF={rtf}, RTF_RECIPROCAL={rtf_reciprocal}"
    )

    results.append(
        torch.stack(
            [
                lsd,
                base_lsd,
                lsd_hf,
                base_lsd_hf,
                lsd_lf,
                base_lsd_lf,
                rtf,
                rtf_reciprocal,
            ],
            dim=0,
        ).unsqueeze(-1)
    )

    results = torch.cat(results, dim=1)
    # Get mean and std in [[mean, std], [mean, std], ...] format
    results = torch.stack([results.mean(dim=1), results.std(dim=1)], dim=1)
    # Convert to [mean, std, mean, std]
    results = results.flatten().tolist()
    # Add sample rate to the beginning of the list
    results.insert(0, args.experiment.lr_sr)

    if args.experiment.hr_sr == 16000:
        save_results_to_csv(results, "results_16kHz.csv")
    elif args.experiment.hr_sr == 48000:
        save_results_to_csv(results, "results_48kHz.csv")


def _main(args):
    global __file__
    print(args)
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)

    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    run(args)


@hydra.main(
    config_path="conf", config_name="main_config"
)  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
