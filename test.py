"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import logging
import os

import torch
from pathlib import Path
import hydra
import wandb

from src.data.datasets import LrHrSet
from src.ddp import distrib
from src.evaluate import evaluate
from src.models import modelFactory
from src.utils import bold

logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Spectral Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'  # TODO: more to args/ user input

SERIALIZE_KEY_MODELS = 'models'
SERIALIZE_KEY_BEST_STATES = 'best_states'
SERIALIZE_KEY_STATE = 'state'


def _load_model(args):
    model_name = args.experiment.model
    checkpoint_file = Path(args.checkpoint_file)
    model = modelFactory.get_model(args)['generator']
    package = torch.load(checkpoint_file, 'cpu')
    load_best = args.continue_best
    if load_best:
        logger.info(bold(f'Loading model {model_name} from best state.'))
        model.load_state_dict(
            package[SERIALIZE_KEY_BEST_STATES][SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])
    else:
        logger.info(bold(f'Loading model {model_name} from last state.'))
        model.load_state_dict(package[SERIALIZE_KEY_MODELS]['generator'][SERIALIZE_KEY_STATE])

    return model

def run(args):
    tt_dataset = LrHrSet(args.dset.test, args.experiment.lr_sr, args.experiment.hr_sr,
                         stride=None, segment=None, with_path=True, upsample=args.experiment.upsample)
    tt_loader = distrib.loader(tt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = _load_model(args)
    model.cuda()

    lsd, visqol, enhanced_filenames = evaluate(args, tt_loader, 0, model)
    logger.info(f'Done evaluation.')



def _get_wandb_config(args):
    included_keys = ['eval_every', 'optim', 'lr', 'losses', 'epochs']
    wandb_config = {k: args[k] for k in included_keys}
    wandb_config.update(**args.experiment)
    wandb_config.update({'train': args.dset.train, 'test': args.dset.test})
    return wandb_config


def _init_wandb_run(args):
    tags = args.wandb.tags
    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    logger.info(f'current path: {os.getcwd()}, rank: {args.rank}')
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=_get_wandb_config(args),
               group=os.path.basename(args.dset.name), resume=args.wandb.resume, name=args.experiment.name,
               tags=tags)


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

    _init_wandb_run(args)
    run(args)
    wandb.finish()


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
