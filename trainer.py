import os
import torch
import numpy as np
import pandas as pd
import logging
logging.getLogger()
from tqdm import tqdm
from sklearn import metrics

from report_writer import ProgressReportWriter


class Trainer:
    def __init__(self,
            bert_punc_cap,
            optimizer,
            criterion,
            train_dataloader,
            valid_dataloader,
            save_path,
            batch_size,
            learning_rate,
            epochs,
            num_validations,
            alpha,
            patience,
            stop_metric
        ):
        self.model = torch.nn.DataParallel(bert_punc_cap)
        self.device = bert_punc_cap.device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.save_path = save_path
        self.hparams = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "num_validations": num_validations,
            "alpha": alpha,
            "patience": patience,
            "stop_metric": stop_metric
        }
        progress_filepath = os.path.join(self.save_path, "progress.tsv")
        self._progress_writer = ProgressReportWriter(
            progress_filepath,
            patience,
            stop_metric,
            bert_punc_cap.hparams["class_to_punc"],
            bert_punc_cap.hparams["class_to_case"]
        )

    def validate(self):
        losses, punc_losses, case_losses, punc_f1s, case_f1s = [],[],[],[],[]
        logging.info("Started validating the model on the validation set:")
        for inputs in tqdm(self.valid_dataloader, total=len(self.valid_dataloader)):
            with torch.no_grad():
                samples, punc_labels, case_labels = inputs
                # convert tensor to device
                samples = samples.to(self.device)
                punc_labels = punc_labels.to(self.device)
                case_labels = case_labels.to(self.device)
                punc_outputs, case_outputs = self.model.module(samples)
                # compute loss
                punc_loss = self.criterion(punc_outputs, punc_labels)
                case_loss = self.criterion(case_outputs, case_labels)
                alpha = self.hparams["alpha"]
                loss = (alpha * punc_loss) + (1-alpha) * case_loss
                # get predictions
                punc_labels = punc_labels.cpu().data.numpy().flatten()
                case_labels = case_labels.cpu().data.numpy().flatten()
                punc_preds = punc_outputs.argmax(dim=1).cpu().data.numpy().flatten()
                case_preds = case_outputs.argmax(dim=1).cpu().data.numpy().flatten()
                # compute other metrics
                punc_f1 = metrics.f1_score(punc_labels, punc_preds, average=None)
                case_f1 = metrics.f1_score(case_labels, case_preds, average=None)
                # append the info
                losses.append(loss.cpu().data.numpy())
                punc_losses.append(punc_loss.cpu().data.numpy())
                case_losses.append(case_loss.cpu().data.numpy())
                punc_f1s.append(punc_f1)
                case_f1s.append(case_f1)
        # average losses & other metrics over all validation set.
        val_loss, punc_loss, case_loss = \
            np.mean(losses), np.mean(punc_losses), np.mean(case_losses)
        punc_f1, case_f1 = np.mean(punc_f1s), np.mean(case_f1s)
        return val_loss, punc_loss, case_loss, punc_f1, case_f1

    def train(self):
        logging.debug("Changing the model's mode to `train`")
        self.model.train()
        print_every = (
            round(len(self.train_dataloader)/self.hparams["num_validations"])
        )
        pbar = tqdm(total=print_every)
        
        epoch = self._progress_writer._curr_epoch
        while(epoch < self.hparams["epochs"]):
            batch_count = 1
            validation_counter = 1

            for inputs in self.train_dataloader:
                samples, punc_labels, case_labels = inputs
                # convert tensor to device
                samples = samples.to(self.device)
                punc_labels = punc_labels.to(self.device)
                case_labels = case_labels.to(self.device)
                # predict
                punc_outputs, case_outputs = self.model.module(samples)
                # compute loss
                punc_loss = self.criterion(punc_outputs, punc_labels)
                case_loss = self.criterion(case_outputs, case_labels)
                alpha = self.hparams["alpha"]
                loss = (alpha * punc_loss) + (1-alpha) * case_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()      
                pbar.update()
                if batch_count % print_every == 0:
                    pbar.close()
                    # get loss values
                    train_loss = loss.cpu().data.numpy()
                    punc_train_loss = punc_loss.cpu().data.numpy()
                    case_train_loss = case_loss.cpu().data.numpy()
                    # evaluate model
                    logging.debug("Changing the model's mode to `eval`")
                    self.model.eval()
                    (valid_loss, punc_valid_loss, case_valid_loss,
                    punc_f1, case_f1) = self.validate()
                    # report results
                    self._progress_writer.write_results(
                        epoch, validation_counter,
                        train_loss, punc_train_loss, case_train_loss,
                        valid_loss, punc_valid_loss, case_valid_loss,
                        punc_f1, case_f1
                    )
                    # TODO save best model
                    # going back to train mode
                    logging.debug("Changing the model's mode to `train`")
                    self.model.train()
                    pbar = tqdm(total=print_every)
                    validation_counter += 1
                batch_count += 1

            pbar.close()
            if self._progress_writer.should_stop():
                logging.info(f"Early stopping at epoch {epoch}")
                logging.info("Saving the model's checkpoint")
            torch.save(self.model.state_dict(), self.save_path+f"/{epoch}.ckpt")
            if epoch < self.hparams["epochs"]-1:
                pbar = tqdm(total=print_every)
            epoch += 1

