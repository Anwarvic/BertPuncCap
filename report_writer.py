import os
import datetime
import logging
logging.get_logger()
import  numpy   as np
import  pandas  as pd
from collections import deque


class ProgressReportWriter:
    def __init__(self, progress_tsv, patience, class_to_punc, class_to_case):
        self._patience = patience
        self._class_to_punc = class_to_punc
        self._class_to_case = class_to_case
        # load important variables from progress file
        if not os.path.exists(progress_tsv):
            logging.info("Couldn't load progress report, so creating one!")
            self._curr_epoch = 0
            self._best_valid_loss = float("inf")
            self._last_few_valid_losses = deque(maxlen=self._patience)
            self._headers = ["time", "epoch", "validation_num", "train_loss",
                "punc_train_loss", "cap_train_loss",
                "valid_loss", "punc_valid_loss", "cap_valid_loss",
                "punc_overall_f1", "case_overall_f1"]
            self._headers += [punc+"_f1" for punc in class_to_punc.values()]
            self._headers += [case+"_f1" for case in class_to_case.values()]
            self._progress_tsv = open(progress_tsv, 'a')
            self._progress_tsv.write('\t'.join(self._headers)+'\n')
        else:
            df = pd.read_csv(progress_tsv)
            #TODO do these
            # self.curr_epoch = 

    def write_results(self,
            epoch,
            train_loss,
            punc_train_loss,
            case_train_loss,
            valid_loss,
            punc_valid_loss,
            case_valid_loss,
            punc_f1_scores,
            case_f1_scores):
        results = {}
        results["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["epoch"] = epoch
        results["train_loss"] = train_loss
        results["punc_train_loss"] = punc_train_loss
        results["case_train_loss"] = case_train_loss
        results["valid_loss"] = valid_loss
        results["punc_valid_loss"] = punc_valid_loss
        results["case_valid_loss"] = case_valid_loss
        results["punc_overall_f1"] = np.mean(punc_f1_scores[1:]) #ignoring OTHER class
        results["case_overall_f1"] = np.mean(case_f1_scores[1:]) #ignoring OTHER class
        # adding results for each punctuation
        results += {
            self._class_to_punc[k]:punc_f1_scores 
                for k in range(len(punc_f1_scores))
        }
        # adding results for each case
        results += {
            self._class_to_case[k]:case_f1_scores 
                for k in range(len(case_f1_scores))
        }
        logged_results = "\t".join([results[key] for key in self._headers])
        logging.info("Model validation results:\n"+logged_results)
        self._progress_tsv.write(logged_results)

