import pandas as pd
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np


class FillingMissingCatValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna('NAN')


class FillingMissingQuatValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        values = {}
        for index in X.index:
            if pd.isnull(X.loc[index, 'GarageYrBlt']):
                X.loc[index, 'GarageYrBlt'] = X.loc[index, 'YearBuilt']
        sum_nan_series = X.select_dtypes(exclude='object').isna().sum()
        for col in sum_nan_series[sum_nan_series != 0].index:
            if col == 'LotFrontage':
                values[col] = X[col].median()
            elif col == 'MasVnrArea':
                values[col] = X[col].median()
            else:
                values[col] = -1
        return X.fillna(value=values)


class Encoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100000000)
        self.enc.fit(X.select_dtypes(include='object'))
        return self

    def transform(self, X, y=None):
        X_object_transformed = self.enc.transform(X.select_dtypes(include='object'))
        columns_object = list(X.select_dtypes(include='object').columns)
        X.loc[:, columns_object] = X_object_transformed
        return X


class FeatureCreation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        bool_year_liste = [X.loc[index, 'YearBuilt'] == X.loc[index, 'YearRemodAdd'] for index in X.index]
        X['HouseRemodelling'] = [0 if bool_year else 1 for bool_year in bool_year_liste]
        X['NbrPorch'] = 0
        for index in X.index:
            porch_nbr = 0
            for col in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']:
                if X.loc[index, col] != 0:
                    porch_nbr += 1
            X.loc[index, 'NbrPorch'] = porch_nbr
        X['GotPool'] = [0 if pd.isnull(cond_pool) else 1 for cond_pool in X['PoolQC']]
        X['GotWoodDeck'] = [0 if size_wood == 0 else 1 for size_wood in X['WoodDeckSF']]
        X['GotBsmt'] = [0 if pd.isnull(cond_bsmt) else 1 for cond_bsmt in X['BsmtCond']]
        X['GotFireplace'] = [0 if pd.isnull(cond_fireplace) else 1 for cond_fireplace in X['FireplaceQu']]
        return X.drop(columns = "Id")


class PreprocessorFeature:
    def __init__(self):
        self.model = make_pipeline(FillingMissingQuatValues(), FillingMissingCatValues(), FeatureCreation(), Encoder())

    def transform_train_set(self, X_sample, y=None):
        X_sample_transformed = self.model.fit_transform(X_sample, y)
        return X_sample_transformed

    def transform_test_set(self, X_sample, y=None):
        X_sample_transformed = self.model.transform(X_sample)
        return X_sample_transformed
    
    def save_model(self, path_to_saved_model):
        joblib.dump(self.model, path_to_saved_model)

def evaluation(model_prediction, training_sample, target_y, test_sample):
    model_prediction.fit(training_sample, target_y)
    return model_prediction.predict(test_sample)



if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(prog='Preprocessing module for house prices dataset', 
                                                description= 'Fit preprocessing pipeline and save model for training')
    parser.add_argument('filename', help = 'path to raw training data')
    parser.add_argument('--output_model', dest = 'output_model_path', default= 'preprocess_pipeline.gz', help = 'path to saved model')
    parser.add_argument('--output_training', dest = 'output_training_path', default= '', help = 'specify path to cleaned training data')
    args = parser.parse_args()
    df_train = pd.read_csv(args.filename)
    df_copy_train = df_train.copy()
    X_train = df_copy_train.drop(columns=['SalePrice'])
    y_train = df_copy_train['SalePrice']
    preprocessor_feature = PreprocessorFeature()
    X_train_transformed = preprocessor_feature.transform_train_set(X_train, y_train)
    preprocessor_feature.save_model(args.output_model_path)
    if args.output_training_path:
        df_train_transformed = pd.DataFrame(data=X_train_transformed)
        df_train_transformed.insert(loc=0, column='SalePrice', value=y_train)
        X_train_transformed_array = df_train_transformed.to_numpy()
        np.savetxt(args.output_training_path, X_train_transformed_array, delimiter=",",fmt = "%d")