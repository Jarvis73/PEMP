import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from core import losses as loss_utils
from core import solver
from core.metrics import FewShotMetric, Accumulator
from data_kits import datasets
from utils import loggers
from utils import timer as timer_utils

C = loggers.C


class BaseEvaluator(object):
    """
    Evaluator base class. Evaluator is used in the validation stage and testing stage.
    All the evaluators should inherit from this class and implement the `test_step()`
    function.

    Parameters
    ----------
    cfg: misc.MapConfig
        Experiment configuration.
    model: nn.Module
        PyTorch model instance.
    mode: str
        Evaluation mode. [EVAL_ONLINE, EVAL]
    logger_name: str
        Equals to the experiment name, which is used to call the global logger of
        current experiment.

    """
    def __init__(self, cfg, model, mode, logger_name):
        self.cfg = cfg
        self.mode = mode
        if mode not in ["EVAL_ONLINE", "EVAL"]:
            raise ValueError(f"Not supported evaluation mode {mode}. [EVAL_ONLINE, EVAL]")
        self.logger = loggers.get_global_logger(name=logger_name)

        self.model = model
        self.loss_obj = loss_utils.get(cfg)

    @staticmethod
    def round(array):
        if isinstance(array, float) or array.ndim == 0:
            return f"{array:5.2f}"
        if array.ndim == 1:
            return "[" + ", ".join([f"{x:5.2f}" for x in array]) + "]"

    def test_step(self, inputs, qry_msk, **kwargs):
        raise NotImplementedError

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
                for inputs, qry_msk, classes in tqdm_gen:
                    with timer.start():
                        qry_pred, loss = self.test_step(inputs, qry_msk)

                    tqdm_gen.set_description(f'[{self.mode}] [round {epoch}/{self.cfg.te.epochs}] loss: {loss:.5f}')
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


class BaseTrainer(object):
    """
    Trainer base class. Trainer is used in the training stage.
    All the trainers should inherit from this class and implement the `train_step()`
    function.

    Parameters
    ----------
    cfg: misc.MapConfig
        Experiment configuration.
    model: nn.Module
        PyTorch model instance.
    _run: sacred.run.Run
        The Run object of Sacred, which represents and manages a single run of an
        experiment
    logger_name: str
        Equals to the experiment name, which is used to call the global logger of
        current experiment.

    """
    def __init__(self, cfg, model, _run, logger_name):
        self.cfg = cfg
        self.run = _run
        self.logger = loggers.get_global_logger(name=logger_name)

        # Define model-related objects
        self.model = model
        self.loss_obj = loss_utils.get(cfg)
        self.model.maybe_fix_params()
        self.optimizer, self.scheduler = solver.get(
            self.model, max_steps=self.cfg.tr.total_epochs * (self.cfg.data.train_n // self.cfg.data.bs))

        # Define model_dir for saving checkpoints
        self.do_ckpt = False
        if self.run._id is not None:
            self.model_dir = Path(cfg.g.model_dir) / str(cfg.tag) / str(_run._id)
            self.do_ckpt = True
        else:
            # For unobserved experiments, we still make a checkpoint at the end of training.
            self.model_dir = Path(cfg.g.model_dir) / "None"

        # Define metrics and output templates
        self.best_iou = -1.
        self.best_epoch = -1
        ndigits = len(str(cfg.tr.epochs))
        self.template = f"Epoch: {{:{ndigits}d}}/{{:{ndigits}d}}" + \
                        " | LR: {:.2e} | Train {:7.5f} | Val {:7.5f} | mIoU {:5.2f} | bIoU {:5.2f}" + \
                        " | Speed: {:.2f}it/s"

    def train_step(self, *inputs, qry_msk, **kwargs):
        """
        Virtual function for training a step

        Parameters
        ----------
        inputs: tuple
            A tuple containing model inputs.
            sup_rgb: torch.Tensor
                Support images
            sup_mask: torch.Tensor
                Support masks
            qry_rgb: torch.Tensor
                Query images
        qry_msk: torch.Tensor
            Query masks (labels of current episode)
        kwargs: dict
            Other keyword parameters
            history_mask: torch.Tensor
                For CaNet implementation.

        Returns
        -------
        loss: torch.Tensor
            A scalar of the loss value in current training step.

        """
        raise NotImplementedError

    def start_training_loop(self, data_loader, evaluator, data_loader_val, num_classes):
        timer = timer_utils.Timer()
        self.prepare_snapshot()

        for epoch in range(1, self.cfg.tr.total_epochs + 1):
            # 1. Training
            self.model.train()
            total_loss = 0.
            data_loader.dataset.sample_tasks()

            tqdm_gen = tqdm.tqdm(data_loader, leave=False)
            for inputs, labels, cls in tqdm_gen:
                with timer.start():
                    lr = self.optimizer.param_groups[0]['lr']
                    loss = self.train_step(*inputs, qry_msk=labels).item()
                    total_loss += loss
                    tqdm_gen.set_description(f'[TRAIN] loss: {loss:.5f} - lr: {lr:g}')
                self.step_lr()

            # 2. Evaluation
            self.try_snapshot(epoch)
            mloss, miou, biou, best = self.evaluation(epoch, evaluator, data_loader_val, num_classes)
            self.log_result(epoch, total_loss / timer.calls, mloss, miou, biou, best, timer.cps)

            # 3. Prepare for next epoch
            timer.reset()

        self.try_snapshot(final=True)

    def prepare_snapshot(self):
        if not self.do_ckpt:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def step_lr(self):
        """
        Update learning rate by the specified learning rate policy.
        For 'cosine' and 'poly' policies, the learning rate is updated by steps.
        For other policies, the learning rate is updated by epochs.
        """
        if self.scheduler is None:
            return

        if not hasattr(self, "step_lr_counter"):
            self.step_lr_counter = 0
        self.step_lr_counter += 1

        if self.cfg.tr.lrp in ["cosine", "poly"]:   # forward per step
            self.scheduler.step()
        elif self.step_lr_counter == self.cfg.data.train_n // self.cfg.data.bs:      # forward per epoch
            self.scheduler.step()
            self.step_lr_counter = 0

    def try_snapshot(self, epoch=-1, final=False, verbose=0):
        if final:
            # Save a checkpoint at the end of training.
            if self.run._id is None:
                self.model_dir.mkdir(parents=True, exist_ok=True)    # First time(also the last time) making checkpoint
                postfix = time.strftime("%y%m%d-%H%M%S", time.localtime())
                save_path = self.model_dir / f"ckpt-{postfix}.pth"
                verbose = 1
            else:
                save_path = self.model_dir / "ckpt.pth"
            torch.save(self.model.state_dict(), str(save_path))
            if verbose:
                self.logger.info(C.c(f" \\_/ Save checkpoint to {save_path}", C.OKGREEN))
            return save_path

        # Save a checkpoint by the checkpoint interval.
        if self.do_ckpt and self.cfg.tr.ckpt_epoch > 0 and epoch % self.cfg.tr.ckpt_epoch == 0:
            save_path = str(self.model_dir / "ckpt.pth")
            torch.save(self.model.state_dict(), save_path)
            if verbose:
                self.logger.info(C.c(f" \\_/ Save checkpoint to {save_path}", C.OKGREEN))
            return save_path

    def evaluation(self, epoch, evaluator, *args, **kwargs):
        """
        Evaluate the model during training and save the best checkpoint.

        Parameters
        ----------
        epoch: int
            Current epoch number.
        evaluator: BaseEvaluator
            An Evaluator for evaluating the model
        args: tuple
            Passed to the `evaluator.start_eval_loop()`
        kwargs: dict
            Passed to the `evaluator.start_eval_loop()`

        Returns
        -------
        mloss: float
            Mean loss
        miou: float
            Mean-IoU
        biou: float
            Binary-IoU
        best: bool
            If this epoch generates the best results or not.

        """
        mloss, miou, biou = evaluator.start_eval_loop(*args, **kwargs)
        best = False
        if miou > self.best_iou:
            self.best_iou, self.best_epoch = miou, epoch
            if self.do_ckpt:
                save_path = str(self.model_dir / "bestckpt.pth")
                torch.save(self.model.state_dict(), save_path)
                best = True
        return mloss, miou, biou, best

    def log_result(self, epoch, train_loss, val_loss, val_mIoU, val_bIoU, best, speed, **kwargs):
        # Log results to the console
        log_str = self.template.format(
            epoch, self.cfg.tr.total_epochs, self.optimizer.param_groups[0]['lr'],
            train_loss, val_loss, val_mIoU * 100, val_bIoU * 100, speed) + " (best)" * best + "\n"
        self.logger.info(C.c(log_str, C.BOLD))
        # Log results to the sacred database
        self.run.log_scalar('train_loss', float(train_loss), epoch)
        self.run.log_scalar('val_loss', float(val_loss), epoch)
        self.run.log_scalar('val_mIoU', float(val_mIoU), epoch)
        self.run.log_scalar('val_bIoU', float(val_bIoU), epoch)
        for k, v in kwargs.items():
            self.run.log_scalar(k, float(v), epoch)


def evaluate_and_save(cfg, ds, data_loader, step_fn, eid):
    """
    For visualizing segmentation results and response maps. This function makes
    a evaluation run and saves all the images, masks, predictions and response
    maps to the directory '<PROJECT_DIR>/http/static/<experiment>'. The
    visualization is implementated by html. See the README.md for details.

    Parameters
    ----------
    cfg: misc.MapConfig
        Experiment configuration.
    ds: torch.utils.data.Dataset
        A PyTorch Dataset instance.
    data_loader: torch.utils.data.DataLoader
        A PyTorch DataLoaer instance.
    step_fn: callable
        A function to perform a inference step.
    eid: int
        Experiment id

    """
    http_dir = f"http/static/" \
               f"{eid}_{cfg.data.dataset.lower()}_{cfg.shot}shot_{cfg.tag}_s{cfg.split}{'_misc' if cfg.p.cls > 0 else ''}"
    if cfg.data.one_cls > 0:
        http_dir = http_dir + f"_c{cfg.data.one_cls}"

    ds.reset_sampler()  # Fix sampling order of the test set.
    ds.sample_tasks()
    with torch.no_grad():
        for i, ((sup_img, sup_msk, qry_img), qry_msk, cls, sup_names, qry_names) in enumerate(data_loader):
            cls = cls[0]
            cname = datasets.get_class_name(cls, cfg.data.dataset)
            inputs = [sup_img.cuda(), sup_msk.cuda(), qry_img.cuda()]

            pred, indices = step_fn(inputs, qry_msk)
            label = qry_msk.float()

            save = http_dir + f"/{i:03d}_{cls:02d}"
            Path(save).mkdir(parents=True, exist_ok=True)
            # save indices
            indices = indices.detach().cpu()
            colors = torch.from_numpy(np.array([[147, 70, 25], [179, 116, 30], [207, 172, 112],
                                                [12, 11, 100], [38, 32, 193], [78, 178, 247]]))  # BGR
            color_t = colors.index_select(0, indices.view(-1)).view(*indices.shape, -1)

            acc = (pred * label).sum() * 2 / (pred.sum() + label.sum())
            print(f"[{i:03d}][{cls:02d}] Accuracy: {acc.item():.3f}")

            if cfg.shot == 1:
                sup_name = sup_names[0][0]
                qry_name = qry_names[0][0]

                data = {"acc": str(round(acc.item(), 3)),
                        "cls_id": int(cls),
                        "cls_name": cname,
                        "sup": sup_name if isinstance(sup_name, str) else str(int(sup_name)),
                        "qry": qry_name if isinstance(qry_name, str) else str(int(qry_name))}
                with open(f"{save}/data.json", "w") as f:
                    json.dump(data, f)

                cv2.imwrite(f"{save}/{cname}_sup_img_{data['sup']}.jpg", np.array(ds.get_image(sup_name))[:, :, ::-1])
                cv2.imwrite(f"{save}/{cname}_sup_msk_{data['sup']}.png",
                            np.array(ds.get_label(cls, sup_name, cache=False, new_label=True)))
                cv2.imwrite(f"{save}/{cname}_qry_img_{data['qry']}.jpg", np.array(ds.get_image(qry_name))[:, :, ::-1])
                cv2.imwrite(f"{save}/{cname}_qry_msk_{data['qry']}.png",
                            np.array(ds.get_label(cls, qry_name, cache=False, new_label=True)))
                cv2.imwrite(f"{save}/{cname}_qry_pred_{data['qry']}.png", (pred[0].numpy() * 255).astype(np.uint8))
                cv2.imwrite(f"{save}/{cname}_qry_color_{data['qry']}.png", color_t[0].numpy().astype(np.uint8))
            else:  # shot == 5
                qry_name = qry_names[0][0]

                data = {"acc": str(round(acc.item(), 3)),
                        "cls_id": int(cls),
                        "cls_name": cname,
                        "qry": qry_name if isinstance(qry_name, str) else str(int(qry_name))}
                for j, sup_name in enumerate(sup_names):
                    sup_name = sup_name[0]
                    data[f"sup{j + 1}"] = sup_name if isinstance(sup_name, str) else str(int(sup_name))
                with open(f"{save}/data.json", "w") as f:
                    json.dump(data, f)

                for j, sup_name in enumerate(sup_names):
                    sup_name = sup_name[0]
                    cv2.imwrite(f"{save}/{cname}_sup_img_{data[f'sup{j + 1}']}.jpg",
                                np.array(ds.get_image(sup_name))[:, :, ::-1])
                    cv2.imwrite(f"{save}/{cname}_sup_msk_{data[f'sup{j + 1}']}.png",
                                np.array(ds.get_label(cls, sup_name, cache=False, new_label=True)))

                cv2.imwrite(f"{save}/{cname}_qry_img_{data['qry']}.jpg", np.array(ds.get_image(qry_name))[:, :, ::-1])
                cv2.imwrite(f"{save}/{cname}_qry_msk_{data['qry']}.png",
                            np.array(ds.get_label(cls, qry_name, cache=False, new_label=True)))
                cv2.imwrite(f"{save}/{cname}_qry_pred_{data['qry']}.png", (pred[0].numpy() * 255).astype(np.uint8))
                cv2.imwrite(f"{save}/{cname}_qry_color_{data['qry']}.png", color_t[0].numpy().astype(np.uint8))
