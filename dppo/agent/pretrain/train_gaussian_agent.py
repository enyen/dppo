"""
Pre-training Gaussian/GMM policy

"""

import logging
import wandb
import numpy as np

log = logging.getLogger(__name__)
from dppo.util.timer import Timer
from dppo.agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)

    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            ent_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train, infos_train = self.model.loss(
                    *batch_train,
                    ent_coef=self.ent_coef,
                )
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                ent_train_epoch.append(infos_train["entropy"].item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)
            ent_train = np.mean(ent_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(
                        *batch_val,
                        ent_coef=self.ent_coef,
                    )
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos_train.items()]
                )
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | {infos_str} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                            "entropy - train": ent_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
