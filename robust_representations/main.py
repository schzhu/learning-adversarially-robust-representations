from argparse import ArgumentParser
import os
from cox import utils
import cox.store

from robustness.datasets import DATASETS
from robustness.defaults import check_and_fill_args
from robustness.tools import helpers
from robustness import defaults
from unsup_models import make_and_restore_model as make_and_restore_model_unsup
from train_unsup import train_model, eval_model


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser.add_argument('--no-store', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('--task',
                    choices=['estimate-mi',
                             'train-model',
                             'train-encoder',
                             'train-classifier'])
parser.add_argument('--representation-type',
                    choices=['layer',
                             'neuron-asbatch',
                             'neuron-crossproduct'],
                    default='layer')
parser.add_argument('--estimator-loss', choices=['normal', 'worst'],
                    default='normal')
parser.add_argument('--classifier-loss', choices=['standard', 'robust'])
parser.add_argument('--classifier-arch', choices=['mlp', 'linear'])
parser.add_argument('--share-arch', action='store_true', default=False)

parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
parser.add_argument('--va-fd-measure', default='JSD')
parser.add_argument('--va-hsize', type=int, default=2048)

args = parser.parse_args()

# torch.manual_seed(41)
# torch.cuda.manual_seed(41)
# np.random.seed(41)
# random.seed(41)
# torch.backends.cudnn.deterministic=True


def main(args):
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    store = None if args.no_store else setup_store_with_metadata(args)

    if args.debug:
        args.workers = 0

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    attacker_model, checkpoint = make_and_restore_model_unsup(args=args,
                                                              dataset=dataset)
    if args.eval_only:
        eval_model(args, attacker_model, val_loader,
                   checkpoint=checkpoint, store=store)
    else:
        train_model(args, attacker_model, loaders,
                    checkpoint=checkpoint, store=store)


def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    args.adv_train = (args.classifier_loss == 'robust') or \
                     (args.estimator_loss == 'worst')
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args


def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
    return store


if __name__ == "__main__":
    args = cox.utils.Parameters(args.__dict__)
    args = setup_args(args)

    args.workers = 0 if args.debug else args.workers
    print(args)

    main(args)


