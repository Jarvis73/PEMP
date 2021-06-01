import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from sacred import Experiment

from config import experiments_setup
from config import global_ingredient, device_ingredient
from core import solver
from core.base_trainer import BaseTrainer, BaseEvaluator
from core.metrics import FewShotMetric, Accumulator
from data_kits import datasets
from networks.canet import net_ingredient, ModelClass
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
    tag = "canet"               # str, Configuration tag
    shot = 1                    # int, number of support samples per episode
    query = 1                   # int, number of query samples per episode
    split = -1                  # int, split number
    seed = 1234                 # int, random seed
    ckpt = "bestckpt.pth"       # str, checkpoint file
    exp_id = -1                 # experiment id to load checkpoint. -1 means `ckpt` is full path.
    loss = "ce"                 # str, loss type [ce/cedt]
    sigma = 5.                  # float, sigma value used in DT loss

misc.post_hook(ex, NAME)


class Evaluator(BaseEvaluator):
    def test_step(self, inputs, qry_msk, **kwargs):
        pred = self.model(*[x.cuda() for x in inputs], False, history_mask=kwargs["history_mask"].cuda())
        qry_pred = F.interpolate(pred, qry_msk.shape[-2:], mode='bilinear', align_corners=True)     # [BQ, 2, H, W]
        qry_msk = qry_msk.view(-1, *qry_msk.shape[-2:])
        loss = self.loss_obj(qry_pred, qry_msk.cuda()).item()
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()    # [B, H, W]
        pred_softmax = F.softmax(pred, dim=1).detach().cpu()
        return qry_pred, loss, pred_softmax

    def start_eval_loop(self, data_loader, num_classes):
        # Set model to evaluation mode (for specific layers, such as batchnorm, dropout, dropblock)
        self.model.eval()
        # Fix sampling order of the test set.
        data_loader.dataset.reset_sampler()
        timer = timer_utils.Timer()
        accum = Accumulator(loss=[], miou=[], biou=[])
        val_labels = datasets.get_val_labels(self.cfg.split)

        # Disable computing gradients for accelerating evaluation process
        with torch.no_grad():
            for epoch in range(1, self.cfg.te.epochs + 1):
                fs_metric = FewShotMetric(num_classes)
                accum_inner = Accumulator(loss=[])
                data_loader.dataset.sample_tasks()

                tqdm_gen = tqdm.tqdm(data_loader, leave=False)
                for inputs, qry_msk, classes, indices, history_mask in tqdm_gen:
                    with timer.start():
                        qry_pred, loss, pred_softmax = self.test_step(inputs, qry_msk, history_mask=history_mask)

                    tqdm_gen.set_description(f'[{self.mode}] [round {epoch}/{self.cfg.te.epochs}] loss: {loss:.5f}')
                    for j in range(pred_softmax.shape[0]):
                        sub_index = indices[j][0].item()    # Assume query=1
                        c = classes[j].item()
                        data_loader.dataset.history_mask_list[c][sub_index] = pred_softmax[j]

                    accum_inner.update(loss=loss)
                    fs_metric.update(qry_pred, qry_msk, classes)

                mIoU, mIoU_mean = fs_metric.mIoU(val_labels)
                bIoU, bIoU_mean = fs_metric.mIoU(val_labels, binary=True)
                self.logger.info(f'[round {epoch}/{self.cfg.te.epochs}] '
                                 f'mIoU: {self.round(mIoU * 100)} -> {self.round(mIoU_mean * 100)}  |  '
                                 f'bIoU: {self.round(bIoU * 100)} -> {self.round(bIoU_mean * 100)}')
                loss = accum_inner.mean(["loss"])
                accum.update(loss=loss, miou=mIoU, biou=bIoU)

        # 5 runs average
        if self.mode == "EVAL":
            miou_5r, biou_5r = accum.mean(["miou", "biou"], axis=0)
            miou_5r_avg, biou_5r_avg = accum.mean(["miou", "biou"])
            self.logger.info('--------------------- Final Results ---------------------')
            self.logger.info(f'| mIoU mean: {self.round(miou_5r * 100)} ==> {self.round(miou_5r_avg * 100)}')
            self.logger.info(f'| bIoU mean: {self.round(biou_5r * 100)} ==> {self.round(biou_5r_avg * 100)}')
            self.logger.info(f'| speed: {self.round(timer.cps)} FPS')
            self.logger.info('---------------------------------------------------------')

        return accum.mean(["loss", "miou", "biou"])


class Trainer(BaseTrainer):
    def train_step(self, *inputs, qry_msk, **kwargs):
        self.optimizer.zero_grad()
        pred = self.model(*[x.cuda() for x in inputs], False,
                          history_mask=kwargs["history_mask"].cuda())                               # [BQ, 2, h, w]
        qry_pred = F.interpolate(pred, qry_msk.shape[-2:], mode='bilinear', align_corners=True)     # [BQ, 2, H, W]
        loss = self.loss_obj(qry_pred, qry_msk.view(-1, *qry_msk.shape[-2:]).cuda())
        loss.backward()
        self.optimizer.step()
        pred_softmax = F.softmax(pred, dim=1).detach().cpu()
        return loss, pred_softmax

    @ex.capture
    def start_training_loop(self, data_loader, evaluator: Evaluator, data_loader_val, num_classes):
        timer = timer_utils.Timer()
        self.prepare_snapshot()

        for epoch in range(1, self.cfg.tr.total_epochs + 1):
            # 1. Training
            self.model.train()
            total_loss = 0.
            data_loader.dataset.sample_tasks()

            tqdm_gen = tqdm.tqdm(data_loader, leave=False)
            for inputs, labels, cls, indices, history_mask in tqdm_gen:
                with timer.start():
                    lr = self.optimizer.param_groups[0]['lr']
                    loss, pred_softmax = self.train_step(*inputs, qry_msk=labels, history_mask=history_mask)
                    loss = loss.item()
                    total_loss += loss
                    tqdm_gen.set_description(f'[TRAIN] loss: {loss:.5f} - lr: {lr:g}')
                    for j in range(pred_softmax.shape[0]):
                        sub_index = indices[j][0].item()    # Assume query=1
                        c = cls[j].item()
                        data_loader.dataset.history_mask_list[c][sub_index] = pred_softmax[j]
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
    misc.set_seed(seed)

    _, data_loader, _ = datasets.load(cfg.data, "train_canet", split, shot, query, logger=logger, first=True)
    _, data_loader_val, num_classes = datasets.load(cfg.data, "test_canet", split, shot, query, logger=logger)

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

    _, data_loader, num_classes = datasets.load(cfg.data, "test_canet", split, shot, query, logger=logger, first=True)

    model = ModelClass(logger).cuda()
    model_ckpt, _ = misc.find_snapshot(cfg, exp_id, ckpt)
    model.load_weights(model_ckpt, logger)

    tester = Evaluator(cfg, model, "EVAL", NAME)
    logger.info("Start testing.")
    loss, miou, biou = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {miou * 100:.2f}, bIoU: {biou * 100:.2f}"


if __name__ == '__main__':
    ex.run_commandline()
