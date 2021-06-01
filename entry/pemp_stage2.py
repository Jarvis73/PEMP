import sys
from pathlib import Path

import torch
import torch.nn as nn
from sacred import Experiment

from config import experiments_setup
from config import global_ingredient, device_ingredient
from core import solver
from core.base_trainer import BaseTrainer, BaseEvaluator, evaluate_and_save
from data_kits import datasets
from networks.pemp_stage2 import PriorNet, ModelClass, net_ingredient
from utils import loggers
from utils import misc

NAME = "PEMP"
ROOT = Path(__file__).parent
ex = Experiment(name=NAME, ingredients=[global_ingredient, device_ingredient, datasets.data_ingredient,
                                        net_ingredient, solver.train_ingredient, solver.test_ingredient],
                save_git_info=False, base_dir=Path(__file__).parents[1])
experiments_setup(ex)


@ex.config
def ex_config():
    """ Experiment configuration """
    tag = "pemp_stage2"         # str, Configuration tag
    shot = 1                    # int, number of support samples per episode
    query = 1                   # int, number of query samples per episode
    split = -1                  # int, split number [0, 1, 2, 3], required
    seed = 1234                 # int, random seed
    ckpt = "bestckpt.pth"       # str, checkpoint file
    exp_id = -1                 # experiment id to load checkpoint. -1 means `ckpt` is full path.
    loss = "ce"                 # str, loss type. ce: xentropy; cedt: xentropy with a weight map. [ce/cedt]
    sigma = 5.                  # float, sigma value used in DT loss

    # Multi-stage training
    s1 = {
        "ckpt": "bestckpt.pth", # checkpoint of stage 1.
        "id": -1,               # specify experiment id of stage 1.
    }

    p = {
        "cls": -1,
        "sup": "",
        "qry": ""
    }

misc.post_hook(ex, NAME)


class Evaluator(BaseEvaluator):
    def __init__(self, cfg, stage1, model, mode, logger_name):
        super(Evaluator, self).__init__(cfg, model, mode, logger_name)
        self.stage1 = stage1

    def test_step(self, inputs, qry_msk, **kwargs):
        s1_logits = self.stage1(*[x.cuda() for x in inputs])                                # [BQ, 2, H, W]
        s1_pred = torch.argmax(s1_logits, dim=1, keepdim=True)                              # [BQ, 1, H, W]
        qry_pred = self.model(*[x.cuda() for x in inputs], s1_pred, qry_msk.shape[-2:])     # [BQ, 2, H, W]
        qry_msk = qry_msk.view(-1, *qry_msk.shape[-2:])
        loss = self.loss_obj(qry_pred, qry_msk.cuda()).item()
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()                            # [B, H, W]
        return qry_pred, loss


class Trainer(BaseTrainer):
    def __init__(self, cfg, stage1, model, _run):
        super(Trainer, self).__init__(cfg, model, _run, NAME)
        self.stage1 = stage1

    def train_step(self, *inputs, qry_msk):
        self.optimizer.zero_grad()
        s1_logits = self.stage1(*[x.cuda() for x in inputs])                                # [BQ, 2, H, W]
        qry_prior = torch.argmax(s1_logits, dim=1, keepdim=True)                            # [BQ, 1, H, W]
        qry_pred = self.model(*[x.cuda() for x in inputs], qry_prior, qry_msk.shape[-2:])   # [BQ, 2, H, W]
        loss = self.loss_obj(qry_pred, qry_msk.view(-1, *qry_msk.shape[-2:]).cuda())
        loss.backward()
        if self.cfg.net.backbone == "vgg16":
            # Clip gradients norm to 1.1 which gives a better performance
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.1)
        self.optimizer.step()
        return loss


@ex.command
def train(_run, _config,
          seed, split, shot, query, s1):
    """
    Start an experiment by training and testing PEMP_Stage1.

    Usage:

        CUDA_VISIBLE_DEVICES="0" PYTHONPATH=./ python entry/pemp_stage1.py train with <UPDATE>

    <UPDATE>:

        The user can update parameters. Here we list some useful parameters:

        split=0                 # (Required) The split number. [0, 1, 2, 3]
        shot=1                  # The number of support samples per episode. Default to 1. [1, 5]
        data.dataset=PASCAL     # Dataset name. Default to "PASCAL". [PASCAL, COCO]
        data.height=401         # Resize images to a fixed size
        data.width=401          # Resize images to a fixed size
        loss=cedt               # Loss type. Default to "ce". [ce, cedt]
                                #     ce: cross-entropy; cedt: cross-entropy with a weight map
        s1.id=-1                # The expriment id of PEMP_Stage1.
        s1.ckpt=bestckpt.pth    # The checkpoint path of PEMP_Stage1.
        tr.lr=0.001             # Learning rate. Default to 0.001.
        tr.total_epochs=3       # Total training epochs. Default to 3.
        -u                      # Disable Sacred observers
        -p                      # Print configurations before running experiment
    """
    cfg = misc.MapConfig(_config)
    s1 = misc.MapConfig(s1)
    logger = loggers.get_global_logger(name=NAME)
    logger.info(f"Run:" + " ".join(sys.argv))
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(seed)

    _, data_loader, _ = datasets.load(cfg.data, "train", split, shot, query, logger=logger, first=True)
    _, data_loader_val, num_classes = datasets.load(cfg.data, "eval_online", split, shot, query, logger=logger)
    logger.info(f"{' ' * 10} ==> Settings: split={split} shot={shot} stage=2")

    stage1 = PriorNet(logger).cuda().eval()   # Make sure the eval model
    stage1_ckpt, _ = misc.find_snapshot(cfg, s1.id, s1.ckpt)
    stage1.load_weights(stage1_ckpt, logger)
    stage1.maybe_fix_params(fix=True)

    model = ModelClass(shot, query, logger).cuda()

    trainer = Trainer(cfg, stage1, model, _run)
    evaluator = Evaluator(cfg, stage1, model, "EVAL_ONLINE", NAME)
    logger.info("Start training.")
    trainer.start_training_loop(data_loader, evaluator, data_loader_val, num_classes)

    logger.info(f"==================== Ending training with id {_run._id} ====================")
    if _run._id is not None:
        return test(_run, _config, exp_id=_run._id)


@ex.command
def test(_run, _config,
         seed, split, shot, query, ckpt, exp_id, s1):
    """
    Start an experiment by testing PEMP_Stage1.

    Usage:

        CUDA_VISIBLE_DEVICES="0" PYTHONPATH=./ python entry/pemp_stage1.py test with <UPDATE>

    <UPDATE>:

        The user can update parameters. Here we list some useful parameters:

        split=0                 # (Required) The split number. [0, 1, 2, 3]
        shot=1                  # The number of support samples per episode. Default to 1. [1, 5]
        data.dataset=PASCAL     # Dataset name. Default to "PASCAL". [PASCAL, COCO]
        data.height=401         # Resize images to a fixed size
        data.width=401          # Resize images to a fixed size
        s1.id=-1                # The expriment id of PEMP_Stage1.
        s1.ckpt=bestckpt.pth    # The checkpoint path of PEMP_Stage1.
        exp_id=-1               # The experiment id of current model. Used when testing the model.
        ckpt=bestckpt.pth       # The checkpoint path of current model. Used when testing the model.
        -u                      # Disable Sacred observers
        -p                      # Print configurations before running experiment

    Note:

        During testing, there are several ways to specify a checkpoint:

        1. use 'exp_id': test with exp_id=1
            --> ./model_dir/pemp_stage1/1/bestckpt.pth
        2. use 'ckpt': test with ckpt=/path/to/ckpt.pth
            --> /path/to/ckpt.pth
        3. use both 'exp_id' and 'ckpt': test with exp_id=1 ckpt=ckpt.pth
            --> ./model_dir/pemp_stage1/1/ckpt.pth

        It is the same to the 's1.id' and 's1.ckpt' when specifying the checkpoint of PEMP_Stage1.
    """
    cfg = misc.MapConfig(_config)
    s1 = misc.MapConfig(s1)
    logger = loggers.get_global_logger(name=NAME)
    misc.try_snapshot_files(_run)
    misc.set_seed(seed)

    _, data_loader, num_classes = datasets.load(cfg.data, "test", split, shot, query, logger=logger, first=True)
    logger.info(f"{' ' * 10} ==> Settings: split={split} shot={shot} stage=2")

    stage1 = PriorNet(logger).cuda().eval()   # Make sure the eval model
    stage1_ckpt, _ = misc.find_snapshot(cfg, s1.id, s1.ckpt)
    stage1.load_weights(stage1_ckpt, logger)

    model = ModelClass(shot, query, logger).cuda()
    model_ckpt, _ = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    tester = Evaluator(cfg, stage1, model, "EVAL", NAME)
    logger.info("Start testing.")
    loss, miou, biou = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {miou * 100:.2f}, bIoU: {biou * 100:.2f}"


@ex.command
def visualize(_run, _config,
              tag, split, shot, query, seed, ckpt, exp_id, s1, p):
    """
    Visualizing prediction results and response maps of PEMP_Stage2.
    This command will not be observed by default. Therefore, there is no need to use the '-u'.

    Usage:

        CUDA_VISIBLE_DEVICES="0" PYTHONPATH=./ python entry/pemp_stage1.py visualize with <UPDATE>

    <UPDATE>:

        The user can update parameters. Here we list some useful parameters:

        split=0                 # (Required) The split number. [0, 1, 2, 3]
        shot=1                  # The number of support samples per episode. Default to 1. [1, 5]
        data.dataset=PASCAL     # Dataset name. Default to "PASCAL". [PASCAL, COCO]
        data.height=401         # Resize images to a fixed size
        data.width=401          # Resize images to a fixed size
        s1.id=-1                # The expriment id of PEMP_Stage1.
        s1.ckpt=bestckpt.pth    # The checkpoint path of PEMP_Stage1.
        exp_id=-1               # The experiment id of current model. Used when testing the model.
        ckpt=bestckpt.pth       # The checkpoint path of current model. Used when testing the model.
        data.test_n=1000        # The number of evaluation and saving. Default to 1000.
    """
    cfg = misc.MapConfig(_config)
    s1 = misc.MapConfig(s1)
    p = misc.MapConfig(p)
    logger = loggers.get_global_logger(name=NAME)
    misc.set_seed(seed)

    if p.cls > 0:
        dataset = datasets.OneExampleLoader(cfg.data, split, shot, query)
        (sup_img, sup_msk, qry_img), qry_msk, cls = dataset.load(p.cls, p.sup, p.qry)
        data_loader = [((sup_img, sup_msk, qry_img), qry_msk, cls, [p.sup], [p.qry])]
    else:
        dataset, data_loader, _ = datasets.load(cfg.data, "test", split, shot, query, logger=logger,
                                                ret_name=True, first=True)

    stage1 = PriorNet(logger).cuda().eval()   # Make sure the eval model
    stage1_ckpt, _ = misc.find_snapshot(cfg, s1.id, s1.ckpt)
    stage1.load_weights(stage1_ckpt, logger)

    model = ModelClass(shot, query, logger).cuda().eval()
    model_ckpt, eid = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    def forward(inputs, qry_msk):
        s1_logits = stage1(*inputs)  # [BQ, 2, H, W]
        s1_pred = torch.argmax(s1_logits, dim=1, keepdim=True)  # [BQ, 1, H, W]
        logits, indices = model(*inputs, s1_pred, out_shape=qry_msk.shape[-2:], ret_ind=True)
        pred = logits.argmax(dim=1).detach().cpu().float()
        return pred, indices

    evaluate_and_save(cfg, dataset, data_loader, forward, eid)


if __name__ == '__main__':
    ex.run_commandline()
