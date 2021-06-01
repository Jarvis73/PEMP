import sys
from pathlib import Path

import torch.nn as nn
from sacred import Experiment

from config import experiments_setup
from config import global_ingredient, device_ingredient
from core import solver
from core.base_trainer import BaseTrainer, BaseEvaluator, evaluate_and_save
from data_kits import datasets
from networks.pemp_stage1 import net_ingredient, ModelClass
from utils import loggers
from utils import misc

NAME = "PEMP"
ROOT = Path(__file__).parent
ex = Experiment(name=NAME,
                ingredients=[global_ingredient, device_ingredient, datasets.data_ingredient,
                             net_ingredient, solver.train_ingredient, solver.test_ingredient],
                save_git_info=False, base_dir=Path(__file__).parents[1])
experiments_setup(ex)


@ex.config
def ex_config():
    """ Experiment configuration """
    tag = "pemp_stage1"         # str, Configuration tag
    shot = 1                    # int, number of support samples per episode
    query = 1                   # int, number of query samples per episode [Default to 1. Don't change!!!]
    split = -1                  # int, split number [0, 1, 2, 3], required
    seed = 1234                 # int, random seed
    ckpt = "bestckpt.pth"       # str, checkpoint file
    exp_id = -1                 # experiment id to load checkpoint. -1 means `ckpt` is full path.
    loss = "ce"                 # str, loss type [ce/cedt]
    sigma = 5.                  # float, sigma value used in DT loss

    p = {
        "cls": -1,
        "sup": "",
        "qry": ""
    }

misc.post_hook(ex, NAME)


class Evaluator(BaseEvaluator):
    def test_step(self, inputs, qry_msk, **kwargs):
        qry_pred = self.model(*[x.cuda() for x in inputs], qry_msk.shape[-2:])
        qry_msk = qry_msk.view(-1, *qry_msk.shape[-2:])
        loss = self.loss_obj(qry_pred, qry_msk.cuda()).item()
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()    # [B, H, W]
        return qry_pred, loss


class Trainer(BaseTrainer):
    def train_step(self, *inputs, qry_msk=None):
        self.optimizer.zero_grad()
        qry_pred = self.model(*[x.cuda() for x in inputs], qry_msk.shape[-2:])             # [BQ, 2, H, W]
        loss = self.loss_obj(qry_pred, qry_msk.view(-1, *qry_msk.shape[-2:]).cuda())
        loss.backward()
        # Clip gradients norm to 1.1 which gives a better performance
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.1)
        self.optimizer.step()
        return loss


@ex.command
def train(_run, _config,
          seed, split, shot, query):
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
        tr.lr=0.001             # Learning rate. Default to 0.001.
        tr.total_epochs=3       # Total training epochs. Default to 3.
        -u                      # Disable Sacred observers
        -p                      # Print configurations before running experiment
    """
    cfg = misc.MapConfig(_config)
    logger = loggers.get_global_logger(name=NAME)
    logger.info(f"Run:" + " ".join(sys.argv))
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(cfg.seed)

    _, data_loader, _ = datasets.load(cfg.data, "train", split, shot, query, logger=logger, first=True)
    _, data_loader_val, num_classes = datasets.load(cfg.data, "eval_online", split, shot, query, logger=logger)
    logger.info(f"{' ' * 10} ==> Settings: split={split} shot={shot} stage=1")

    model = ModelClass(logger).cuda()

    trainer = Trainer(cfg, model, _run, NAME)
    evaluator = Evaluator(cfg, model, "EVAL_ONLINE", NAME)
    logger.info("Start training.")
    trainer.start_training_loop(data_loader, evaluator, data_loader_val, num_classes)

    logger.info(f"==================== Ending training with id {_run._id} ====================")
    if _run._id is not None:
        return test(_run, _config, exp_id=_run._id)


@ex.command
def test(_run, _config,
         seed, split, shot, query, ckpt, exp_id):
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
    """
    cfg = misc.MapConfig(_config)
    logger = loggers.get_global_logger(name=NAME)
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(seed)

    _, data_loader, num_classes = datasets.load(cfg.data, "test", split, shot, query, logger=logger, first=True)
    logger.info(f"{' ' * 10} ==> Settings: split={split} shot={shot} stage=1")

    model = ModelClass(logger).cuda()
    model_ckpt, _ = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    tester = Evaluator(cfg, model, "EVAL", NAME)
    logger.info("Start testing.")
    loss, miou, biou = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {miou * 100:.2f}, bIoU: {biou * 100:.2f}"


@ex.command
def visualize(_run, _config,
              tag, seed, split, shot, query, ckpt, exp_id, p):
    """
    Visualizing prediction results and response maps of PEMP_Stage1.
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
        exp_id=-1               # The experiment id of current model. Used when testing the model.
        ckpt=bestckpt.pth       # The checkpoint path of current model. Used when testing the model.
        data.test_n=1000        # The number of evaluation and saving. Default to 1000.
    """
    cfg = misc.MapConfig(_config)
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

    model = ModelClass().cuda().eval()
    logger.info(f"           ==> Model {model.__class__.__name__} created")

    # Load checkpoints
    model_ckpt, eid = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    def forward(inputs, qry_msk):
        logits, indices = model(*[x.cuda() for x in inputs], out_shape=qry_msk.shape[-2:], ret_ind=True)
        pred = logits.argmax(dim=1).detach().cpu().float()
        return pred, indices

    evaluate_and_save(cfg, dataset, data_loader, forward, eid)


if __name__ == '__main__':
    ex.run_commandline()
