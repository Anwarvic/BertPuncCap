import os
import logging
logging.getLogger()
import  numpy   as np
import  pandas  as pd
from datetime import datetime
from collections import deque


class ProgressReportWriter:
    def __init__(self,
            progress_filepath,
            patience,
            stop_metric,
            class_to_punc,
            class_to_case
        ):
        self.progress_filepath = progress_filepath
        self._patience = patience
        self._stop_metric = stop_metric
        self._class_to_punc = class_to_punc
        self._class_to_case = class_to_case
        logging.info("Initializing the ProgressReportWriter module")
        # create headers for the progress report file
        self._headers = ["time", "epoch", "validation_num",
            "train_loss", "punc_train_loss", "case_train_loss",
            "valid_loss", "punc_valid_loss", "case_valid_loss",
            "punc_overall_f1", "case_overall_f1", "overall_f1"]
        self._headers += [punc+"_f1" for punc in class_to_punc.values()]
        self._headers += [case+"_f1" for case in class_to_case.values()]
        if os.path.exists(progress_filepath):
            logging.info("Reading progress report!")
            df = pd.read_csv(progress_filepath, sep='\t')
            # load important variables from progress file
            self._curr_epoch = max(df["epoch"])
            self._best_valid = min(df[self._stop_metric]) \
                if 'loss' in stop_metric else max(df[self._stop_metric])
            valid_results = df[self._stop_metric].values
            best_valid_idx = int(np.where(valid_results == self._best_valid)[-1])
            idx = max(best_valid_idx, len(df) - self._patience)
            self._last_few_valid_scores = deque(df[self._stop_metric][idx:])
        else:
            logging.info("Couldn't load progress report, so creating one!")
            # initialize important variables
            self._curr_epoch = 0
            self._best_valid = float("inf") \
                if 'loss' in stop_metric else float("-inf")
            self._last_few_valid_scores = deque(maxlen=self._patience)
            with open(self.progress_filepath, 'a') as fout:
                fout.write('\t'.join(self._headers)+'\n')
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
        PRECISION = 5 #round precision
        results = {}
        results["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["epoch"] = epoch
        results["validation_num"] = validation_num
        results["train_loss"] = np.round_(train_loss, PRECISION)
        results["punc_train_loss"] = np.round_(punc_train_loss, PRECISION)
        results["case_train_loss"] = np.round_(case_train_loss, PRECISION)
        results["valid_loss"] = np.round_(valid_loss, PRECISION)
        results["punc_valid_loss"] = np.round_(punc_valid_loss, PRECISION)
        results["case_valid_loss"] = np.round_(case_valid_loss, PRECISION)
        punc_overall_f1 = np.mean(punc_f1_scores[1:]) #ignoring OTHER class
        case_overall_f1 = np.mean(case_f1_scores[1:]) #ignoring OTHER class
        results["punc_overall_f1"] = np.round_(punc_overall_f1, PRECISION)
        results["case_overall_f1"] = np.round_(case_overall_f1, PRECISION)
        results["overall_f1"] = \
            np.round_(np.mean([punc_overall_f1, case_overall_f1]), PRECISION)
        # adding results for each punctuation
        results.update({
            self._class_to_punc[i]+'_f1':np.round_(punc_f1_scores[i], PRECISION)
                for i in range(len(punc_f1_scores))
        })
        # adding results for each case
        results.update({
            self._class_to_case[i]+'_f1':np.round_(case_f1_scores[i], PRECISION)
                for i in range(len(case_f1_scores))
        })
        logged_results = "\t".join([str(results[key]) for key in self._headers])
        logging.info("Model validation results are:\n"
                    + "\t".join(self._headers) + '\n' + logged_results)
        with open(self.progress_filepath, 'a') as fout:
            fout.write(logged_results+'\n')
        # update member variables
        self._curr_epoch = epoch
        self._best_valid = min(results[self._stop_metric], self._best_valid) \
            if "loss" in self._stop_metric \
            else max(results[self._stop_metric], self._best_valid)
        self._last_few_valid_scores.append(results[self._stop_metric])
    
    def should_stop(self):
        """Checks the possibility of early-stopping."""
        logging.debug("Checking the possibility of early-stopping.")
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

