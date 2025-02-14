import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, cross_val_score

from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer, mean_squared_error


from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from pyDOE3 import lhs

import os
import multiprocessing as mp
import random
import time
import sys
import win32com.client as win32  # For interacting with Aspen Plus
import pythoncom  # Required for COM operations
from contextlib import redirect_stdout, redirect_stderr

np.set_printoptions(linewidth=120, suppress=True, formatter={'float_kind': '{:.3f}'.format})



class BaseModel:
    def __init__(self, scaler_X=None, scaler_y=None):
        self.model = None  # Single model for all outputs
        self.scaler_X = scaler_X if scaler_X else MinMaxScaler()  # Default scaler for input
        self.scaler_y_list = []  # List of scalers for output variables
        self.X_train = None
        self.y_train = None

    def train(self, X, y, blackbox_idx=None, retrain=True, cv=5):
        if blackbox_idx is None or not isinstance(blackbox_idx, (list, np.ndarray)):
            raise ValueError("blackbox_idx must be a list or array of column indices.")

        self.X_train = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # Select target columns
        self.y_train = y[:, blackbox_idx]
        assert self.y_train.shape[0] == self.X_train.shape[0], (
            f"Mismatch: X has {self.X_train.shape[0]} rows but y_train has {self.y_train.shape[0]} rows."
        )

        # Scale input features
        X_scaled = self.scaler_X.fit_transform(self.X_train)
        X_transformed = self.transform_X(X_scaled)

        # Scale each column of y independently
        scaled_y_columns = []
        self.scaler_y_list = []  # Reset scalers for retraining
        for col in range(self.y_train.shape[1]):
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(self.y_train[:, col].reshape(-1, 1)).ravel()
            self.scaler_y_list.append(scaler_y)
            scaled_y_columns.append(y_scaled)

        Y_scaled = np.column_stack(scaled_y_columns)  # Stack all scaled columns into one array

        # Perform cross-validation
        if retrain or self.model is None:
            self.model = self.create_model()

            # Define KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            scorer = make_scorer(mean_squared_error, greater_is_better=False)

            # Cross-validate
            scores = cross_val_score(self.model, X_transformed, Y_scaled, cv=kf, scoring=scorer)
            print(f"Cross-Validation Scores (MSE): {scores}")
            print(f"Mean CV Score: {np.mean(scores)}")

            # Train final model on the full dataset
            self.model.fit(X_transformed, Y_scaled)

    def predict(self, vma):
        vma_scaled = self.scaler_X.transform(vma.reshape(1, -1))
        vma_transformed = self.transform_X(vma_scaled)

        # Predict all outputs together and inverse-transform each column
        Y_pred_scaled = self.model.predict(vma_transformed).reshape(1, -1)
        Y_pred = np.column_stack(
            [self.scaler_y_list[i].inverse_transform(Y_pred_scaled[:, i].reshape(-1, 1))[:, 0]
             for i in range(len(self.scaler_y_list))]
        )

        return np.maximum(Y_pred.flatten(), 1e-5)  # Ensure non-negative predictions

    def create_model(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def transform_X(self, X):
        # Default transformation is no transformation
        return X




class LinearModel(BaseModel):
    def __init__(self):
        super().__init__()

    def create_model(self):
        return LinearRegression()

    def transform_X(self, X):
        # No transformation for basic Linear Regression
        return X




class Poly2Model(BaseModel):
    def __init__(self, degree=2):
        super().__init__()
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)

    def create_model(self):
        return LinearRegression()

    def transform_X(self, X):
        return self.poly_features.fit_transform(X)




class Poly2RidgeModel(Poly2Model):
    def __init__(self, degree=2, alpha=0.1):
        super().__init__(degree)
        self.alpha = alpha

    def create_model(self):
        return Ridge(alpha=self.alpha)




class KrigingModel(BaseModel):
    def __init__(self, kernel=None, n_restarts_optimizer=10):
        super().__init__(scaler_X=MinMaxScaler())
        self.kernel = kernel if kernel else C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.n_restarts_optimizer = n_restarts_optimizer

    def create_model(self):
        return GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer
        )




class RigorousMLPModel:
    def __init__(self, hidden_layer_sizes=(100, 100), max_iter=500):
        self.models = []  # List to store individual MLP regressors
        self.scaler_X = MinMaxScaler()  # Scaler for input features
        self.scaler_y_list = []  # Separate scalers for each output column
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None

    def train(self, X, y, retrain=True):
        self.X_train = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        self.y_train = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

        # Scale input features
        X_scaled = self.scaler_X.fit_transform(self.X_train)

        if retrain or not self.models:
            self.models = []
            self.scaler_y_list = []
            for i in range(self.y_train.shape[1]):
                y_col = self.y_train[:, i].reshape(-1, 1)

                # Scale each output column
                scaler_y = MinMaxScaler()
                y_col_scaled = scaler_y.fit_transform(y_col).ravel()
                self.scaler_y_list.append(scaler_y)

                # Train MLPRegressor model
                model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)
                model.fit(X_scaled, y_col_scaled)
                self.models.append(model)

    def predict(self, vma):
        vma_scaled = self.scaler_X.transform(vma.reshape(1, -1))

        # Predict using each trained model and inverse-transform each output
        predicted_scaled = [model.predict(vma_scaled)[0] for model in self.models]
        predicted = [self.scaler_y_list[i].inverse_transform([[pred]])[0, 0] for i, pred in enumerate(predicted_scaled)]
        return np.maximum(predicted, 1e-5)

    def cross_validate(self, X, y, cv_splits=2):
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        results = []

        # Scale input features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled_list = []
        self.scaler_y_list = []
        for i in range(y.shape[1]):
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y.iloc[:, i].values.reshape(-1, 1))
            self.scaler_y_list.append(scaler_y)
            y_scaled_list.append(y_scaled)
        y_scaled = np.hstack(y_scaled_list)

        for i in range(y.shape[1]):
            y_variable = y_scaled[:, i]
            model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter)

            # Perform cross-validation
            y_pred_scaled = cross_val_predict(model, X_scaled, y_variable, cv=kf)

            # Inverse transform predictions
            y_pred = self.scaler_y_list[i].inverse_transform(y_pred_scaled.reshape(-1, 1))
            y_true = self.scaler_y_list[i].inverse_transform(y_variable.reshape(-1, 1))

            # Compute metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = (mean_squared_error(y_true, y_pred))**0.5
            r2 = r2_score(y_true, y_pred)

            # Append results
            results.append({
                'Model': 'MLP',
                'Variable': f'Output {i+1}',
                'mae': mae,
                'rmse': rmse,
                'R2': r2
            })

        return pd.DataFrame(results)
