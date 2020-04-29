import os
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import save_checkpoint
from loss_fun import My_loss
import torch.nn as nn
class model_trainer:
    def __init__(self,
                 model,
                 learning_rate = 2e-3,
                 num_epochs = 20,
                 batch_size = 32
                 ):
        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(
                                    filter(lambda p: p.requires_grad, self.model.parameters()),
                                    learning_rate)  # leave betas and eps by default
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.criterion = My_loss()
    def train(self, train_loader, valid_loader):
        image_num = len(train_loader.dataset.indices)
        niter = np.int(np.ceil(image_num / self.batch_size))


        train_loss_history = {'total_loss':[], 'cls_loss': [], 'reg_loss': [], 'centerness_loss': []}
        valid_loss_history = {'total_loss':[], 'cls_loss': [], 'reg_loss': [], 'centerness_loss': []}

        for i in range(self.num_epochs):
            self.model.train()
            start_t = time.time()
            acc_loss = 0
            acc_cls_loss = 0
            acc_reg_loss = 0
            acc_cen_loss = 0
            for iter_num, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                for key in batch:
                    assert isinstance(batch[key], type(batch['template'])) or isinstance(batch[key], type(batch['targets']))
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                    else:
                        for sub_key in batch[key]:
                            batch[key][sub_key] = batch[key][sub_key].to(self.device)

                batch_template = batch['template']
                batch_detection = batch['detection']
                GT_cls = batch['targets']['gt_class']
                GT_mask = batch['targets']['distances']

                batch_template = batch_template.to(self.device)
                batch_detection = batch_detection.to(self.device)
                GT_cls = GT_cls.to(self.device)
                GT_mask = GT_mask.to(self.device)

                cls, mask_reg, centerness = self.model.forward(batch_template, batch_detection)
                cls_loss, reg_loss, centerness_loss = self.criterion.forward(cls, mask_reg, centerness, GT_cls, GT_mask)
                loss = cls_loss + 1.5 * reg_loss + centerness_loss

                acc_loss += loss.item()
                acc_cls_loss += cls_loss.item()
                acc_reg_loss += reg_loss.item()
                acc_cen_loss += centerness_loss.item()

                loss.backward()
                self.optimizer.step()

                print('(Iter {} / {})'.format(iter_num + 1, niter))
                # self.writer.add_scalar('Loss/train', loss.item(), i * (image_num // batch.shape[0]) + iter_num)
            end_t = time.time()
            acc_loss = acc_loss / niter
            acc_cen_loss = acc_cen_loss / niter
            acc_reg_loss = acc_reg_loss / niter
            acc_cls_loss = acc_cls_loss / niter

            train_loss_history['total_loss'].append(acc_loss)
            train_loss_history['cls_loss'].append(acc_cls_loss)
            train_loss_history['reg_loss'].append(acc_reg_loss)
            train_loss_history['centerness_loss'].append(acc_cen_loss)
            val_tot_loss, val_cls_loss, val_reg_loss, val_cen_loss = self.valid(valid_loader= valid_loader)

            valid_loss_history['total_loss'].append(val_tot_loss)
            valid_loss_history['cls_loss'].append(val_cls_loss)
            valid_loss_history['reg_loss'].append(val_reg_loss)
            valid_loss_history['centerness_loss'].append(val_cen_loss)

            print('(Epoch {} / {}) train loss: {:.4f} valid loss: {:.4f} time per epoch: {:.1f}s current lr: {}'.format(
                i + 1, self.num_epochs, acc_loss, val_tot_loss, end_t - start_t, self.optimizer.param_groups[0]['lr']))
                       # save_checkpoint(self.model.module.state_dict(), is_best=True, checkpoint_dir=os.getcwd() + '/checkpoint/')
            if (i + 1) % 5 == 0:
                print('Save the current model to checkpoint!')
                save_checkpoint(self.model.module.state_dict(), is_best= False, checkpoint_dir= os.getcwd() + '/checkpoint/')
                torch.save(train_loss_history, os.getcwd() + '/checkpoint/train_loss.pt')
                torch.save(valid_loss_history, os.getcwd() + '/checkpoint/valid_loss.pt')
            if i == np.argmin(valid_loss_history['total_loss']):
                print('The current model is the best model! Save it!')
                save_checkpoint(self.model.module.state_dict(), is_best=True,
                                checkpoint_dir=os.getcwd() + '/checkpoint/')

            self.lr_scheduler.step(val_tot_loss)
    def valid(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            image_num = len(valid_loader.dataset.indices)
            niter = np.int(np.ceil(image_num / self.batch_size))
            acc_loss = 0
            acc_cls_loss = 0
            acc_reg_loss = 0
            acc_cen_loss = 0
            for iter_num, batch in enumerate(valid_loader):
                for key in batch:
                    assert isinstance(batch[key], type(batch['template'])) or isinstance(batch[key], type(batch['targets']))
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                    else:
                        for sub_key in batch[key]:
                            batch[key][sub_key] = batch[key][sub_key].to(self.device)
                batch_template = batch['template']
                batch_detection = batch['detection']
                GT_cls = batch['targets']['gt_class']
                GT_mask = batch['targets']['distances']

                cls, mask_reg, centerness = self.model.forward(batch_template, batch_detection)
                cls_loss, reg_loss, centerness_loss = self.criterion.forward(cls, mask_reg, centerness, GT_cls, GT_mask)
                loss = cls_loss + 1.5 * reg_loss + centerness_loss

                acc_loss += loss.item()
                acc_cls_loss += cls_loss.item()
                acc_reg_loss += reg_loss.item()
                acc_cen_loss += centerness_loss.item()

                print('(valid {} / {})'.format(iter_num + 1, niter))
            acc_loss = acc_loss / niter
            acc_cen_loss = acc_cen_loss / niter
            acc_reg_loss = acc_reg_loss / niter
            acc_cls_loss = acc_cls_loss / niter
        return acc_loss, acc_cls_loss, acc_reg_loss, acc_cen_loss