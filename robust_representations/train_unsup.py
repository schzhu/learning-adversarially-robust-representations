import torch as ch
from torch.optim import SGD, lr_scheduler, Adam

from robustness.tools.helpers import AverageMeter, calc_fadein_eps, \
    save_checkpoint, ckpt_at_epoch, has_attr
from robustness.tools import constants as consts
from functools import partial


import dill
import time

import os

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir",# "adv_train",
                           "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint", "use_best",
                         "eps_fadein_epochs", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only:
        check_args(required_args_train)
    else:
        check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")


def make_optimizer_and_schedule(args, model, checkpoint, lr, step_lr):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    optimizer = Adam(model.parameters(), lr)
    # Make schedule
    schedule = None
    if step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=step_lr)
    elif args.custom_schedule:
        cs = args.custom_schedule
        periods = eval(cs) if type(cs) is str else cs

        def lr_func(ep):
            for (milestone, _lr) in reversed(periods):
                if ep > milestone: return _lr / lr
            return lr

        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint and args.task not in ['train-classifier', 'estimate-mi']:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

    return optimizer, schedule


def eval_model(args, atm, loader, *, checkpoint=None, store=None):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    if checkpoint:
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint[f"{'adv' if args.adv_train else 'nat'}_prec1"]

    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA_EXP2)
    writer = store.tensorboard if store else None

    # model = ch.nn.DataParallel(model)

    prec1, nat_loss = _model_loop(args, 'val', loader,
                                  atm, None, start_epoch, (False,), writer)

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args,
                                                               'attack_lr') else None

        prec1, nat_loss = _model_loop(args, 'val', loader,
                                                  atm, None, start_epoch, (True,), writer)
    log_info = {
        'epoch': 0,
        'nat_prec1': prec1,
        'adv_prec1': adv_prec1,
        'nat_loss': nat_loss,
        'adv_loss': adv_loss,
        'train_prec1': float('nan'),
        'train_loss': float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)


def train_model(args, atm, loaders, *, checkpoint=None, store=None):
    """
    Main function for training a model.

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_schedule (str)
                If given, use a custom LR schedule (format: [(epoch, LR),...])
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            eps_fadein_epochs (int, *required if adv_train or adv_eval*)
                If greater than 0, fade in epsilon along this many epochs
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            regularizer (function, optional)
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)`
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        store (cox.Store) : a cox store for logging training progress
    """
    big_encoder = atm.attacker.model

    # Logging setup
    writer = store.tensorboard if store else None
    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA_EXP2)
        store.add_table(consts.CKPTS_TABLE, consts.CKPTS_SCHEMA)

    # Reformat and read arguments
    check_required_args(args)  # Argument sanity check
    args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
    args.attack_lr = eval(str(args.attack_lr)) if has_attr(args,
                                                           'attack_lr') else None

    # Initial setup
    train_loader, val_loader = loaders

    opts, schedules = [], []
    for i, submodel in enumerate(big_encoder.models):
        if submodel is None:
            opts.append(None)
            schedules.append(None)
        else:
            lr = args.lr
            opt, schedule = make_optimizer_and_schedule(args, submodel, checkpoint,
                                                        lr, args.step_lr)
            opts.append(opt)
            schedules.append(schedule)

    best_prec1, best_loss, start_epoch = (0, float('inf'), 0)
    if checkpoint and args.task not in ['train-classifier', 'estimate-mi']:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[f"{'adv' if args.adv_train else 'nat'}_prec1"]

    # Put the model into parallel mode
    # assert not hasattr(model, "module"), "model is already in DataParallel."
    # model = ch.nn.DataParallel(model).cuda()

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        # if args.exp2_neuronest_mode == -1:
        train_prec1, train_loss = \
            _model_loop(args, 'train', train_loader, atm,
                        opts, epoch, (None,), writer) # None

        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model': atm.state_dict(),
            'optimizer': -1,
            'schedule': -1,
            # 'optimizer': opt.state_dict(),
            # 'schedule': (schedule and schedule.state_dict()),
            'epoch': epoch + 1
        }

        def save_checkpoint(filename):
            if not store:
                return
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                              store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            with ch.no_grad():
                prec1, nat_loss = _model_loop(args, 'val', val_loader, atm,
                                              None, epoch, (False,), writer)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            if should_adv_eval:
                adv_prec1, adv_loss = _model_loop(args, 'val', val_loader,
                                                  atm, None, epoch, (True,), writer)
            else:
                adv_prec1, adv_loss = -1.0, -1.0

            # remember best prec@1 and save checkpoint
            if args.task == 'train-model' or args.task == 'train-classifier':
                our_prec1 = adv_prec1 if args.adv_train else prec1
                is_best = our_prec1 > best_prec1
                best_prec1 = max(our_prec1, best_prec1)
            else:
                our_loss = adv_loss if args.adv_train else nat_loss
                is_best = our_loss < best_loss
                best_loss = min(our_loss, best_loss)

            # log every checkpoint
            log_info = {
                'epoch': epoch + 1,
                'nat_prec1': prec1,
                'adv_prec1': adv_prec1,
                'nat_loss': nat_loss,
                'adv_loss': adv_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(
                ckpt_at_epoch(epoch))
            # If  store exists and this is the last epoch, save a checkpoint
            if last_epoch and store: store[consts.CKPTS_TABLE].append_row(sd_info)

            # Update the latest and best checkpoints (overrides old one)
            # if not (args.exp2_neuronest_mode == 0):
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(atm, log_info)

    return atm


def _model_loop(args, loop_type, loader, atm, opts, epoch, advs, writer):

    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)

    is_train = (loop_type == 'train')

    adv_eval, = advs

    prec = 'NatPrec' if not adv_eval else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    atm = atm.train() if is_train else atm.eval() # 操！

    # If adv training (or evaling), set eps and random_restarts appropriately
    eps = calc_fadein_eps(epoch, args.eps_fadein_epochs, args.eps) \
        if is_train else args.eps
    random_restarts = 0 if is_train else args.random_restarts

    attack_kwargs = {
        'constraint': args.constraint,
        'eps': eps,
        'step_size': args.attack_lr,
        'iterations': args.attack_steps,
        'random_start': False,
        'random_restarts': random_restarts,
        'use_best': bool(args.use_best)
    }

    if is_train:
        opt_enc, opt_dim_local, opt_dim_global, opt_cla = opts
    else:
        opt_enc, opt_dim_local, opt_dim_global, opt_cla = None, None, None, None

    losses_cla = AverageMeter()
    precs_cla = AverageMeter()
    losses_enc_dim = AverageMeter()

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (input, target) in iterator:
        target = target.cuda(non_blocking=True)

        # Compute Loss: eval
        if not is_train:
            # if adv_mi_type == 'lo':
            attack_kwargs['custom_loss'] = partial(
                atm.attacker.model.custom_loss_func,
                loss_type='dim')
            # elif adv_mi_type == 'up':
            #     attack_kwargs['custom_loss'] = atm.attacker.model.cal_adv_mi_up_loss_dim
            loss_enc_dim, _, _ = atm.forward_custom(
                input=input,
                target=None,   # no need for target in computing mi
                loss_type='dim',
                make_adv=adv_eval,
                detach=True,    # whatever in eval mode
                enc_in_eval=True,
                **attack_kwargs)

            attack_kwargs['custom_loss'] = partial(
                atm.attacker.model.custom_loss_func,
                loss_type='cla')
            _, loss_cla, prec_cla = atm.forward_custom(
                input=input,
                target=target,
                loss_type='cla',
                make_adv=adv_eval,
                detach=True,
                enc_in_eval=True,
                **attack_kwargs)

        # Compute Loss: train
        else:
            if args.task == 'estimate-mi':
                target = None
                loss_type = 'dim'
                make_adv = True if args.estimator_loss == 'worst' else False
                detach = True
                enc_in_eval = True

            elif args.task == 'train-encoder':
                target = None
                loss_type = 'dim'
                make_adv = True if args.estimator_loss == 'worst' else False
                detach = True
                enc_in_eval = False

            elif args.task == 'train-classifier':
                target = target
                loss_type = 'cla'
                make_adv = True if args.classifier_loss == 'robust' else False
                detach = True
                enc_in_eval = True

            elif args.task == 'train-model':
                target = target
                loss_type = 'cla'
                make_adv = True if args.classifier_loss == 'robust' else False
                detach = False
                enc_in_eval = False

            else:
                raise NotImplementedError

            attack_kwargs['custom_loss'] = partial(
                atm.attacker.model.custom_loss_func,
                loss_type=loss_type)
            loss_enc_dim, loss_cla, prec_cla = atm.forward_custom(
                input=input,
                loss_type=loss_type,
                target=target,
                make_adv=make_adv,
                detach=detach,
                enc_in_eval=enc_in_eval,
                **attack_kwargs)

        # Compute gradient and do SGD step
        if is_train:

            if args.task == 'estimate-mi':
                opt_dim_local.zero_grad()
                opt_dim_global.zero_grad()
                loss_enc_dim.backward()
                opt_dim_local.step()
                opt_dim_global.step()

            elif args.task == 'train-encoder':
                opt_enc.zero_grad()
                opt_dim_local.zero_grad()
                opt_dim_global.zero_grad()
                loss_enc_dim.backward()
                opt_enc.step()
                opt_dim_local.step()
                opt_dim_global.step()

            elif args.task == 'train-classifier':
                opt_cla.zero_grad()
                loss_cla.backward() #retain_graph=True
                opt_cla.step()

            elif args.task == 'train-model':
                opt_enc.zero_grad()
                # opt_cla.zero_grad()
                loss_cla.backward() #retain_graph=True
                opt_enc.step()
                # opt_cla.step()

            else:
                raise NotImplementedError

        losses_cla.update(loss_cla.item(), input.size(0))
        precs_cla.update(prec_cla.item(), input.size(0))
        losses_enc_dim.update(loss_enc_dim.item(), input.size(0))


        # ITERATOR
        desc = ('{2} Epoch:{0} | '
                'Loss_dim {Loss_dim:.4f} | '
                'Loss_cla {Loss_cla:.4f} | '
                'prec_cla {prec_cla:.3f} |'
                .format(
                epoch, prec, loop_msg,
                Loss_dim=losses_enc_dim.avg,
                Loss_cla=losses_cla.avg,
                prec_cla=precs_cla.avg))

        # USER-DEFINED HOOK
        # if has_attr(args, 'iteration_hook'):
        #     args.iteration_hook(testee, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    return precs_cla.avg, losses_enc_dim.avg

