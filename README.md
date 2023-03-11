# House Prices Prediction App
This application predicts the prices of houses based on Kaggle Competition dataset. This app uses an AWS Sagemaker trained model and
the AWS Lambda interface (backend services). For more info about the implementation of the AWS sagemaker model used in this project, visit : https://github.com/physicist-boris/AWS-Sagemaker-sampletraining-house-prices


## Installation
Install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py <filename> -S <path_to_saved_prediction>
```
where <filename> is a path to the csv file representing a house prices dataset (Kaggle Competition Dataset), and <path_to_saved_prediction> the path where to save the prediction of the model (optional).

Avec Docker, 

```bash
docker image build -t <app-name>
docker run <filename> -S <path_to_saved_prediction> <app-name>
```

### Usage Preprocessing modules

```bash
python cleaning.py <filename> --output_model <path_to_saved_preprocessing_model> --output_training <path_to_clean_training_data>
```
where filename is a path to the csv file representing the training dataset for the preprocessing model.
