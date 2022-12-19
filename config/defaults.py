"""
Default configuration values for the experiments.
"""
import os
from yacs.config import CfgNode as CN

_C = CN()

# data
_C.DATA = CN()
_C.DATA.BASE_INPUT_PATH = os.path.join(os.getcwd(), 'input')
_C.DATA.DATASET_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'calcifications.csv')
_C.DATA.TRAIN_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'train.csv')
_C.DATA.TEST_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'external.csv')
_C.DATA.EXTERNAL_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'external.csv')
_C.DATA.APPLY_OHE = True
_C.DATA.SCALE = False
_C.DATA.TEST_SIZE = 0.2
_C.DATA.TARGET = "symptoms"
_C.DATA.CLASS_NAMES = ["Symptomatic", "Asymptomatic"]
_C.DATA.TO_DROP = ["site", "patient_ID", "infarct_side", "stroke", "TIA"]
_C.DATA.TO_SCALE = ["age", "stenosis_left", "stenosis_right"]
_C.DATA.TO_LABEL_ENCODE = ["hypertension", "cad", "gender",
                           "diabetes", "hyperlipidemia", "smoker_status"]
_C.DATA.TO_OHE = ["calcification_type_left", "calcification_type_right"]
_C.DATA.PREFIXES = ["calcification_left", "calcification_right"]
_C.DATA.CAT_FEATURES = ["hypertension", "cad", "gender", "diabetes", "hyperlipidemia",
                        "calcification_type_left", "calcification_type_right", "smoker_status"
                        ]
_C.DATA.SUBSET_DATA = False
_C.DATA.WHICH_SUBSET = "clinical"
_C.DATA.CLINICAL_FEATURE_SUBSET = ["hypertension", "cad", "smoker_status",
                                   "age", "gender", "diabetes", "hyperlipidemia",
                                   "stenosis_right", "stenosis_left",
                                   ]
_C.DATA.CALC_FEATURE_SUBSET = ["calcification_type_left", "calcification_type_right"] # 

# output
_C.OUTPUT = CN()
_C.OUTPUT.BASE_OUTPUT_PATH = os.path.join(os.getcwd(), "output")
_C.OUTPUT.FITTED_MODELS_PATH = os.path.join(os.getcwd(), "output/fitted_models")
_C.OUTPUT.RESULTS_PATH = os.path.join(os.getcwd(), "output/results")
_C.OUTPUT.PREDS_PATH = os.path.join(os.getcwd(), "output/predictions")
_C.OUTPUT.FITTED_MODEL_PATH = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "model.pkl")
_C.OUTPUT.BEST_MODEL = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "final_backup.pkl")
_C.OUTPUT.BEST_PARAMS = os.path.join(_C.OUTPUT.PARAMS_PATH, "best_params_backup.pkl")
_C.OUTPUT.TEST_RESULTS = os.path.join(_C.OUTPUT.RESULTS_PATH, "test_results_final_2000boots.pkl")

# evaluation
_C.EVAL = CN()
_C.EVAL.INNER_N_SPLITS = 5
_C.EVAL.OUTER_N_SPLITS = 10
_C.EVAL.N_REPEATS = 10  # 20
_C.EVAL.SHUFFLE = True
_C.EVAL.ALPHA = 0.05
_C.EVAL.BOOT_ROUNDS = 1000
_C.EVAL.THRESHOLD = 0.43496668724224646
_C.EVAL.THRESHOLD_ONLY_CLINICAL = 0
_C.EVAL.THRESHOLD_ONLY_CALCIFICATION = 0
_C.EVAL.SET_NAMES = ["Train", "Valid"]
_C.EVAL.ALGO_SHORT_NAMES = ["EBM"]
_C.EVAL.ALGO_LONG_NAMES = ["Explainable Boosting Classifier"]
_C.EVAL.METRIC_NAMES = ["Sensitivity", "Specificity", "FPR", "FNR",
                        "Precision", "PPV", "NPV", "F1",
                        "F2", "ROCAUC", "PRAUC", "Brier"]

# misc
_C.MISC = CN()
_C.MISC.SEED = 1234


def get_defaults():
    return _C.clone()
