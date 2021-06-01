import torch
from sacred import Ingredient

from utils import misc

train_ingredient = Ingredient("tr", save_git_info=False)
test_ingredient = Ingredient("te", save_git_info=False)


@train_ingredient.config
def train_config():
    """ Training Arguments """
    epochs = 0                              # int, Number of epochs for training
    total_epochs = 3                        # int, Number of total epochs for training

    lr = 1e-3                               # float, Base learning rate for model training
    lrp = "period_step"                     # str, Learning rate policy [custom_step/period_step/plateau]
    if lrp == "custom_step":
        lr_boundaries = []                  # list, [custom_step] Use the specified lr at the given boundaries
    if lrp == "period_step":
        lr_step = 999999999                 # int, [period_step] Decay the base learning rate at a fixed step
    if lrp in ["custom_step", "period_step", "plateau"]:
        lr_rate = 0.1                       # float, [period_step, plateau] Learning rate decay rate
    if lrp in ["plateau", "cosine", "poly"]:
        lr_end = 0.                         # float, [plateau, cosine, poly] The minimal end learning rate
    if lrp == "plateau":
        lr_patience = 30                    # int, [plateau] Learning rate patience for decay
        lr_min_delta = 1e-4                 # float, [plateau] Minimum delta to indicate improvement
        cool_down = 0                       # bool, [plateau]
        monitor = "val_loss"                # str, [plateau] Quantity to be monitored [val_loss/loss]
    if lrp == "poly":
        power = 0.9                         # float, [poly]

    opt = "sgd"                             # str, Optimizer for training [sgd/adam]
    if opt == "adam":
        adam_beta1 = 0.9                    # float, [adam] Parameter
        adam_beta2 = 0.999                  # float, [adam] Parameter
        adam_epsilon = 1e-8                 # float, [adam] Parameter
    if opt == "sgd":
        sgd_momentum = 0.9                  # float, [momentum] Parameter
        sgd_nesterov = False                # bool, [momentum] Parameter

    weight_decay = 0.0005                   # float, weight decay coefficient
    ckpt_epoch = 1                          # int, checkpoint interval, 0 to disable checkpoint


@test_ingredient.config
def test_config():
    """ Testing Arguments """
    epochs = 5


class PolyLR(object):
    def __init__(self, optimizer, max_iter, power=0.9, lr_end=0, last_step=0):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.power = power
        self.lr_end = lr_end
        self.last_step = last_step
        self.init_lr = optimizer.param_groups[0]['lr']

        self.step()

    def step(self, step=None):
        self.last_step += 1
        if step is None:
            step = self.last_step
        else:
            self.last_step = step

        lr = (self.init_lr - self.lr_end) * (1 - step / self.max_iter) ** self.power + self.lr_end
        self.optimizer.param_groups[0]['lr'] = lr


@train_ingredient.capture
def get(model,
        _config,
        max_steps=200001):
    cfg = misc.MapConfig(_config)
    if isinstance(model, list):
        params_group = model
    elif isinstance(model, torch.nn.Module):
        params_group = model.parameters()
    else:
        raise TypeError(f"`model` must be an nn.Model or a list, got {type(model)}")

    if cfg.opt == "sgd":
        optimizer_params = {"momentum": cfg.sgd_momentum,
                            "weight_decay": cfg.weight_decay,
                            "nesterov": cfg.sgd_nesterov}
        optimizer = torch.optim.SGD(params_group, cfg.lr, **optimizer_params)
    elif cfg.opt == "adam":
        optimizer_params = {"betas": (cfg.adam_beta1, cfg.adam_beta2),
                            "eps": cfg.adam_epsilon,
                            "weight_decay": cfg.weight_decay}
        optimizer = torch.optim.Adam(params_group, cfg.lr, **optimizer_params)
    else:
        raise ValueError("Not supported optimizer: " + cfg.opt)

    if cfg.lrp == "period_step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=cfg.lr_step,
                                                    gamma=cfg.lr_rate)
    elif cfg.lrp == "custom_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=cfg.lr_boundaries,
                                                         gamma=cfg.lr_rate)
    elif cfg.lrp == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=cfg.lr_rate,
                                                               patience=cfg.lr_patience,
                                                               threshold=cfg.lr_min_delta,
                                                               cooldown=cfg.cool_down,
                                                               min_lr=cfg.lr_end)
    elif cfg.lrp == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=max_steps,
                                                               eta_min=cfg.lr_end)
    elif cfg.lrp == "poly":
        scheduler = PolyLR(optimizer,
                           max_iter=max_steps,
                           power=cfg.power,
                           lr_end=cfg.lr_end)
    else:
        raise ValueError

    return optimizer, scheduler
