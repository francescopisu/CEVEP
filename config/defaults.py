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
_C.DATA.TRAIN_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'train_no_ohe.csv')
_C.DATA.TEST_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'test_no_ohe.csv')
_C.DATA.APPLY_OHE = False
_C.DATA.SCALE = False
_C.DATA.TEST_SIZE = 0.2
_C.DATA.TARGET = "symptoms"
_C.DATA.CLASS_NAMES = ["Symptomatic", "Asymptomatic"]
_C.DATA.TO_DROP = ["site", "patient_ID", "infarct_side", "stroke", "tabulator"]
_C.DATA.TO_SCALE = ["age", "stenosis_left", "stenosis_right"]
_C.DATA.TO_LABEL_ENCODE = ["TIA", "hypertension", "cad", "gender", "diabetes", "hyperlipidemia"]
_C.DATA.TO_OHE = ["calcification_type_left", "calcification_type_right", "smoker_status"]
_C.DATA.PREFIXES = ["calcification_left", "calcification_right", "smoker"]
_C.DATA.CAT_FEATURES = ["TIA", "hypertension", "cad", "gender", "diabetes", "hyperlipidemia",
                        "calcification_type_left", "calcification_type_right", "smoker_status"]

# output
_C.OUTPUT = CN()
_C.OUTPUT.BASE_OUTPUT_PATH = os.path.join(os.getcwd(), "output")
_C.OUTPUT.FITTED_MODELS_PATH = os.path.join(os.getcwd(), "output/fitted_models")
_C.OUTPUT.RESULTS_PATH = os.path.join(os.getcwd(), "output/results")
_C.OUTPUT.PARAMS_PATH = os.path.join(os.getcwd(), "output/parameters")
_C.OUTPUT.FITTED_MODEL_PATH = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "model.pkl")
_C.OUTPUT.BEST_MODEL = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "final.pkl")
_C.OUTPUT.BEST_PARAMS = os.path.join(_C.OUTPUT.PARAMS_PATH, "best_params.pkl")
_C.OUTPUT.TEST_RESULTS = os.path.join(_C.OUTPUT.RESULTS_PATH, "test_results_final_2000boots.pkl")

# evaluation
_C.EVAL = CN()
_C.EVAL.N_SPLITS = 10
_C.EVAL.N_REPEATS = 10
_C.EVAL.SHUFFLE = True
_C.EVAL.ALPHA = 0.05
_C.EVAL.BOOT_ROUNDS = 2000
_C.EVAL.THRESHOLD = 0.5008254

# hyperparameter tuning
_C.TUNING = CN()
_C.TUNING.N_TRIALS = 250
_C.TUNING.EARLY_STOPPING_ROUNDS = 30
_C.TUNING.EVAL_SIZE = 0.2
_C.TUNING.DIRECTION = 'minimize'
_C.TUNING.METRIC = 'brier_loss'

# misc
_C.MISC = CN()
_C.MISC.SEED = 1303


def get_defaults():
    return _C.clone()
