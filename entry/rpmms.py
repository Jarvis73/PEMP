import sys
from pathlib import Path

import torch
import tqdm
from sacred import Experiment

from config import experiments_setup
from config import global_ingredient, device_ingredient
from core import solver
from core.base_trainer import BaseTrainer, BaseEvaluator
from core.metrics import FewShotMetric, Accumulator
from data_kits import datasets
from networks.rpmms import net_ingredient, ModelClass
from utils import loggers
from utils import misc
from utils import timer as timer_utils

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
    tag = "rpmms"               # str, Configuration tag
    shot = 1                    # int, number of support samples per episode
    query = 1                   # int, number of query samples per episode [Default to 1. Don't change!!!]
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

misc.post_hook(ex, NAME)


class Evaluator(BaseEvaluator):
    def test_step(self, inputs, qry_msk, **kwargs):
        logits = self.model(*[x.cuda() for x in inputs])
        loss, loss_part1, loss_part2 = self.model.get_loss(
            logits, qry_msk.view(-1, 1, *qry_msk.shape[-2:]).cuda())
        qry_pred = self.model.get_pred(logits, qry_msk)[1].detach().cpu().numpy()    # [B, H, W]
        return qry_pred, loss.item(), loss_part1.item(), loss_part2.item()

    def start_eval_loop(self, data_loader, num_classes):
        # Set model to evaluation mode (for specific layers, such as batchnorm, dropout, dropblock)
        self.model.eval()
        # Fix sampling order of the test set.
        data_loader.dataset.reset_sampler()
        timer = timer_utils.Timer()
        accum = Accumulator(loss=[], loss_p1=[], loss_p2=[], miou=[], biou=[])
        val_labels = datasets.get_val_labels(self.cfg.split)

        # Disable computing gradients for accelerating evaluation process
        with torch.no_grad():
            for epoch in range(1, self.cfg.te.epochs + 1):
                fs_metric = FewShotMetric(num_classes)
                accum_inner = Accumulator(loss=[], loss_p1=[], loss_p2=[])
                data_loader.dataset.sample_tasks()

                tqdm_gen = tqdm.tqdm(data_loader, leave=False)
                for inputs, qry_msk, classes in tqdm_gen:
                    with timer.start():
                        qry_pred, loss, loss_p1, loss_p2 = self.test_step(inputs, qry_msk)

                    tqdm_gen.set_description(f'[{self.mode}] [round {epoch}/{self.cfg.te.epochs}] loss: {loss:.5f} '
                                             f'loss_p1: {loss_p1:.5f} loss_p2: {loss_p2:.5f}')
                    accum_inner.update(loss=loss, loss_p1=loss_p1, loss_p2=loss_p2)
                    fs_metric.update(qry_pred, qry_msk, classes)

                mIoU, mIoU_mean = fs_metric.mIoU(val_labels)
                bIoU, bIoU_mean = fs_metric.mIoU(val_labels, binary=True)
                self.logger.info(f'[round {epoch}/{self.cfg.te.epochs}] '
                                 f'mIoU: {self.round(mIoU * 100)} -> {self.round(mIoU_mean * 100)}  |  '
                                 f'bIoU: {self.round(bIoU * 100)} -> {self.round(bIoU_mean * 100)}')
                mloss, mloss_p1, mloss_p2 = accum_inner.mean(["loss", "loss_p1", "loss_p2"])
                accum.update(loss=mloss, loss_p1=mloss_p1, loss_p2=mloss_p2, miou=mIoU, biou=bIoU)

        # 5 runs average
        if self.mode == "EVAL":
            miou_5r, biou_5r = accum.mean(["miou", "biou"], axis=0)
            miou_5r_avg, biou_5r_avg = accum.mean(["miou", "biou"])
            self.logger.info('--------------------- Final Results ---------------------')
            self.logger.info(f'| mIoU mean: {self.round(miou_5r * 100)} ==> {self.round(miou_5r_avg * 100)}')
            self.logger.info(f'| bIoU mean: {self.round(biou_5r * 100)} ==> {self.round(biou_5r_avg * 100)}')
            self.logger.info(f'| speed: {self.round(timer.cps)} FPS')
            self.logger.info('---------------------------------------------------------')

        loss, loss_p1, loss_p2, miou, biou = accum.mean(["loss", "loss_p1", "loss_p2", "miou", "biou"])
        return (loss, loss_p1, loss_p2), miou, biou


class Trainer(BaseTrainer):
    def train_step(self, *inputs, qry_msk):
        self.optimizer.zero_grad()
        logits = self.model(*[x.cuda() for x in inputs])             # [BQ, 2, H, W]
        loss, loss_part1, loss_part2 = self.model.get_loss(
            logits, qry_msk.view(-1, 1, *qry_msk.shape[-2:]).cuda())
        loss.backward()
        self.optimizer.step()
        return loss, loss_part1, loss_part2

    def start_training_loop(self, data_loader, evaluator, data_loader_val, num_classes):
        timer = timer_utils.Timer()
        self.prepare_snapshot()

        for epoch in range(1, self.cfg.tr.total_epochs + 1):
            # 1. Training
            self.model.train()
            accum = Accumulator(loss=[], loss_p1=[], loss_p2=[])
            data_loader.dataset.sample_tasks()

            tqdm_gen = tqdm.tqdm(data_loader, leave=False)
            for inputs, labels, cls in tqdm_gen:
                with timer.start():
                    lr = self.optimizer.param_groups[0]['lr']
                    loss, loss_p1, loss_p2 = self.train_step(*inputs, qry_msk=labels)
                    accum.update(loss=loss.item(), loss_p1=loss_p1.item(), loss_p2=loss_p2.item())
                    tqdm_gen.set_description(f'[TRAIN] loss: {loss:.5f} - loss_p1: {loss_p1:.5f} - '
                                             f'loss_p2: {loss_p2:.5f} - lr: {lr:g}')
                self.step_lr()

            # 2. Evaluation
            self.try_snapshot(epoch)
            (mloss_val, mloss_p1_val, mloss_p2_val), miou, biou, best = self.evaluation(
                epoch, evaluator, data_loader_val, num_classes)
            mloss, mloss_p1, mloss_p2 = accum.mean(["loss", "loss_p1", "loss_p2"])
            self.log_result(epoch, mloss, mloss_val, miou, biou, best, timer.cps,
                            train_loss_p1=mloss_p1, train_loss_p2=mloss_p2,
                            val_loss_p1=mloss_p1_val, val_loss_p2=mloss_p2_val)

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

    _, data_loader, _ = datasets.load(cfg.data, "train", split, shot, query, logger, first=True)
    _, data_loader_val, num_classes = datasets.load(cfg.data, "eval_online", split, shot, query, logger=logger)

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
    cfg = misc.MapConfig(_config)
    logger = loggers.get_global_logger(name=NAME)
    misc.try_snapshot_files(_run)   # snapshot source files if use FileStorageObserver
    misc.set_seed(seed)

    _, data_loader, num_classes = datasets.load(cfg.data, "test", split, shot, query, logger=logger, first=True)

    model = ModelClass(logger).cuda()
    model_ckpt, _ = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    tester = Evaluator(cfg, model, "EVAL", NAME)
    logger.info("Start testing.")
    (loss, loss_p1, loss_p2), miou, biou = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, Loss_p1: {loss_p1:.4f}, Loss_p2: {loss_p2:.4f}, " \
           f"mIoU: {miou * 100:.2f}, bIoU: {biou * 100:.2f}"


if __name__ == '__main__':
    ex.run_commandline()
