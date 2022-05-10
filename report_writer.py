import os
import logging
logging.getLogger()
import  numpy   as np
import  pandas  as pd
from datetime import datetime
from collections import deque


class ProgressReportWriter:
    def __init__(self, progress_filepath, patience, stop_metric, class_to_punc, class_to_case):
        self._patience = patience
        self._stop_metric = stop_metric
        self._class_to_punc = class_to_punc
        self._class_to_case = class_to_case
        self._progress_tsv = open(progress_filepath, 'a')
        try:
            df = pd.read_csv(progress_filepath, sep='\t')
            logging.info("Reading progress report!")
            # load important variables from progress file
            self.curr_epoch = max(df["epoch"])
            self._best_valid = min(df[self._stop_metric]) \
                if 'loss' in stop_metric else max(df[self._stop_metric])
            best_valid_idx = df[self._stop_metric].index(self._best_valid)
            idx = max(best_valid_idx, len(df) - self._patience)
            self._last_few_valid_scores = df[self._stop_metric][-idx:]
        except pd.errors.EmptyDataError:
            logging.info("Couldn't load progress report, so creating one!")
            self._curr_epoch = 0
            self._best_valid = float("inf") \
                if 'loss' in stop_metric else float("-inf")
            self._last_few_valid_scores = deque(maxlen=self._patience)
            self._headers = ["time", "epoch", "validation_num",
                "train_loss", "punc_train_loss", "cap_train_loss",
                "valid_loss", "punc_valid_loss", "cap_valid_loss",
                "punc_overall_f1", "case_overall_f1"]
            self._headers += [punc+"_f1" for punc in class_to_punc.values()]
            self._headers += [case+"_f1" for case in class_to_case.values()]
            self._progress_tsv.write('\t'.join(self._headers)+'\n')
        # TODO handle the case where progress_filepath has only the header
    

    def write_results(self,
        epoch,
        validation_num,
        train_loss,
        punc_train_loss,
        case_train_loss,
        valid_loss,
        punc_valid_loss,
        case_valid_loss,
        punc_f1_scores,
        case_f1_scores
    ):
        results = {}
        results["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["epoch"] = epoch
        results["validation_num"] = validation_num
        results["train_loss"] = train_loss
        results["punc_train_loss"] = punc_train_loss
        results["case_train_loss"] = case_train_loss
        results["valid_loss"] = valid_loss
        results["punc_valid_loss"] = punc_valid_loss
        results["case_valid_loss"] = case_valid_loss
        punc_overall_f1 = np.mean(punc_f1_scores[1:]) #ignoring OTHER class
        case_overall_f1 = np.mean(case_f1_scores[1:]) #ignoring OTHER class
        results["punc_overall_f1"] = punc_overall_f1
        results["case_overall_f1"] = case_overall_f1
        results["overall_f1"] = np.mean(punc_overall_f1, case_overall_f1)
        # adding results for each punctuation
        results += {
            self._class_to_punc[k]+'_f1':punc_f1_scores 
                for k in range(len(punc_f1_scores))
        }
        # adding results for each case
        results += {
            self._class_to_case[k]+'_f1':case_f1_scores 
                for k in range(len(case_f1_scores))
        }
        logged_results = "\t".join([results[key] for key in self._headers])
        logging.info("Model validation results are:\n"+logged_results)
        self._progress_tsv.write(logged_results)
        # update member variables
        self._curr_epoch = epoch
        self._best_valid = min(results[self._stop_metric], self._best_valid) \
            if "loss" in self._stop_metric \
            else max(results[self._stop_metric], self._best_valid)
        self._last_few_valid_scores.append(results[self._stop_metric])
    
    def should_stop(self):
        # higher is better
        if 'f1' in self._stop_metric:
            if ((len(self._last_few_valid_scores) == self._patience)
                    and max(self._last_few_valid_scores) < self._best_valid):
                return True
            else:
                return False
        # lower is better
        elif 'loss' in self._stop_metric:
            if ((len(self._last_few_valid_scores) == self._patience)
                    and min(self._last_few_valid_scores) > self._best_valid):
                return True
            else:
                return False




