import joblib
import requests
import pandas as pd
import io
import numpy as np
import json
from modules.cleaning import FillingMissingQuatValues, FillingMissingCatValues, FeatureCreation, Encoder


if __name__ == '__main__':
    import argparse 
    import os
    
    
    parser = argparse.ArgumentParser(prog= 'House prices Prediction App', description= 'Application for predicting house prices')
    parser.add_argument("filename", help = 'path to raw test data')
    parser.add_argument('-S', dest = 'saved_prediction_path', default = '', help = 'specify path to saved model prediction')
    args = parser.parse_args()
    preprocessing_features = joblib.load('models/preprocess_model/preprocess_pipeline.gz')
    df_test = pd.read_csv(args.filename)
    df_copy_test = df_test.copy()
    X_test_transformed_array = preprocessing_features.transform(df_copy_test)
    saved_string = io.StringIO()
    np.savetxt(saved_string, X_test_transformed_array, delimiter= ',', fmt = '%d')
    prediction_model_api_url = "https://dsbjugb667p2muymn4zitr32i40tplzl.lambda-url.us-east-2.on.aws/"
    headers = {'content-type': 'text/csv'}
    r = requests.post(url = prediction_model_api_url, headers = headers, data= saved_string.getvalue())
    prediction_dict = r.text
    print(prediction_dict)
    if args.saved_prediction_path:
        with open(args.saved_prediction_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(prediction_dict, jsonfile)
    