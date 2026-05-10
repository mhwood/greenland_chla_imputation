
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import xgboost as xgb
from sklearn.metrics import r2_score
import shap
from sklearn.model_selection import train_test_split

def run_shap_analysis(project_dir, model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values.feature_names = ["SST", "SSS", "Sea Ice", "U Wind", "V Wind", "U", "V"]

    # need to figure out how to save this
    shap.plots.bar(shap_values)

    # fig = plt.figure()
    # plt.savefig(os.path.join(project_dir,'Figures','Remote Sensing Model','Chl_XGBoost_SHAP.png'))
    # plt.close(fig)
