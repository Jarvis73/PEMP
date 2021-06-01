import sys
from pathlib import Path

import tqdm
from sacred import Experiment

from config import experiments_setup
from config import global_ingredient, device_ingredient
from core import solver
from core.base_trainer import BaseTrainer, BaseEvaluator
from data_kits import datasets
from networks.pfenet import ModelClass
from utils import loggers
from utils import misc
from utils import timer as timer_utils

NAME = "PEMP"
ROOT = Path(__file__).parent
ex = Experiment(name=NAME,
                ingredients=[global_ingredient, device_ingredient, datasets.data_ingredient,
                             solver.train_ingredient, solver.test_ingredient],
                save_git_info=False, base_dir=Path(__file__).parents[1])
experiments_setup(ex)


@ex.config
def ex_config():
    """ FSS configuration """
    tag = "pfenet"              # str, Configuration tag
    shot = 1                    # int, number of support samples per episode
    query = 1                   # int, number of query samples per episode
    split = -1                  # int, split number [0, 1, 2, 3], required
    seed = 1234                 # int, random seed
    ckpt = "bestckpt.pth"       # str, checkpoint file
    exp_id = -1                 # experiment id to load checkpoint. -1 means `ckpt` is full path.
    loss = "ce"                 # str, loss type [ce/cedt]
    sigma = 5.                  # float, sigma value used in DT loss
    loss_coef = 1.              # float, coefficient of the auxillary loss

    p = {
        "cls": -1,
        "sup": "",
        "qry": ""
    }

@ex.config_hook
def config_hook(config, command_name, logger):
    if config["split"] == -1:
        raise ValueError("`split` is required!")

misc.post_hook(ex, NAME)


class Evaluator(BaseEvaluator):
    def test_step(self, inputs, qry_msk, **kwargs):
        qry_pred = self.model(*[x.cuda() for x in inputs], qry_msk, qry_msk.shape[-2:])
        qry_msk = qry_msk.view(-1, *qry_msk.shape[-2:])
        loss = self.loss_obj(qry_pred, qry_msk.cuda()).item()
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()    # [B, H, W]
        return qry_pred, loss


class Trainer(BaseTrainer):
    def train_step(self, *inputs, qry_msk=None):
        self.optimizer.zero_grad()
        qry_pred, aux_loss = self.model(*[x.cuda() for x in inputs], qry_msk.cuda(), qry_msk.shape[-2:])
        loss = self.loss_obj(qry_pred, qry_msk.view(-1, *qry_msk.shape[-2:]).cuda())
        loss2 = loss + aux_loss * self.cfg.loss_coef
        loss2.backward()
        self.optimizer.step()
        return loss, aux_loss

    def start_training_loop(self, data_loader, evaluator, data_loader_val, num_classes):
        timer = timer_utils.Timer()
        self.prepare_snapshot()

        for epoch in range(1, self.cfg.tr.total_epochs + 1):
            # 1. Training
            self.model.train()
            total_loss = 0.
            total_aux_loss = 0.
            data_loader.dataset.sample_tasks()

            tqdm_gen = tqdm.tqdm(data_loader, leave=False)
            for inputs, labels, cls in tqdm_gen:
                with timer.start():
                    lr = self.optimizer.param_groups[0]['lr']
                    loss, aux_loss = self.train_step(*inputs, qry_msk=labels)
                    total_loss += loss.item()
                    total_aux_loss += aux_loss.item()
                    tqdm_gen.set_description(f'[TRAIN] loss: {loss:.5f} - aux_loss: {aux_loss:.5f} - lr: {lr:g}')
                self.step_lr()

            # 2. Evaluation
            self.try_snapshot(epoch)
            mloss, miou, biou, best = self.evaluation(epoch, evaluator, data_loader_val, num_classes)
            self.log_result(epoch, total_loss / timer.calls, mloss, miou, biou, best, timer.cps)

            # 3. Prepare for next epoch
            timer.reset()

        self.try_snapshot(final=True)


@ex.command
def train(_run, _config,
          seed, split, shot, query):
    cfg = misc.MapConfig(_config)
    logger = loggers.get_global_logger(name=NAME)
    logger.info(f"Run:" + " ".join(sys.argv))
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(cfg.seed)

    _, data_loader, _ = datasets.load(cfg.data, "train", split, shot, query, logger=logger, first=True)
    _, data_loader_val, num_classes = datasets.load(cfg.data, "eval_online", split, shot, query, logger=logger)

    model = ModelClass(shot, logger).cuda()

    trainer = Trainer(cfg, model, _run, NAME)
    evaluator = Evaluator(cfg, model, "EVAL_ONLINE", NAME)
    logger.info("Start training.")
    trainer.start_training_loop(data_loader, evaluator, data_loader_val, num_classes)

    logger.info(f"==================== Ending training with id {_run._id} ====================")
    return test(_run, _config, exp_id=_run._id)


@ex.command
def test(_run, _config,
         seed, split, shot, query, ckpt, exp_id):
    cfg = misc.MapConfig(_config)
    logger = loggers.get_global_logger(name=NAME)
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(seed)

    _, data_loader, num_classes = datasets.load(cfg.data, "test", split, shot, query, logger=logger, first=True)

    model = ModelClass(shot, logger).cuda()
    model_ckpt, _ = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    tester = Evaluator(cfg, model, "EVAL", NAME)
    logger.info("Start testing.")
    loss, miou, biou = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {miou * 100:.2f}, bIoU: {biou * 100:.2f}"


if __name__ == '__main__':
    ex.run_commandline()
