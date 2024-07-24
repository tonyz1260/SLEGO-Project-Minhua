from sklearn.model_selection import train_test_split
import pandas as pd
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from typing import Union
import joblib
# Import necessary libraries
from pycaret.time_series import *


def fetch_uci_dataset(uci_data_id:int=360, 
                      output_file_path:str='dataspace/AirQuality.csv'):


    # fetch dataset
    downloaded_data = fetch_ucirepo(id=uci_data_id)

    # data (as pandas dataframes)
    X = downloaded_data.data.features
    y = downloaded_data.data.targets

    # # metadata
    # print(air_quality.metadata)

    # # variable information
    # print(air_quality.variables)
    

    # Assuming `X` and `y` are both pandas DataFrames with the same index
    df = pd.concat([X, y], axis=1)
    df.to_csv(output_file_path)
    return df

def prepare_dataset_tabular_ml(input_file_path: str = 'dataspace/AirQuality.csv', 
                               index_col: Union[int, bool] = 0,
                               target_column_name: str = 'NO2(GT)', 
                               drop_columns: list = ['Date','NO2(GT)','Time'],
                               output_file_path:str = 'dataspace/prepared_dataset_tabular_ml.csv'):
    """
    Prepares a dataset for tabular machine learning by loading, cleaning, and optionally modifying it.
    """
    
    # Load the dataset
    df = pd.read_csv(input_file_path, index_col=index_col)
    
    # Drop any 'Unnamed' columns that may exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Set the target column if it exists
    if target_column_name in df.columns:
        df['target'] = df[target_column_name].values
    
    # Drop specified columns if any
    if drop_columns is not None:
        df = df.drop(columns=drop_columns, errors='ignore')
    
    # Save the cleaned dataset
    output_file_path = 'dataspace/prepared_dataset_tabular_ml.csv'
    df.to_csv(output_file_path)
    
    return df



def split_dataset_4ml(input_data_file:str = 'dataspace/prepared_dataset_tabular_ml.csv', 
                           index_col: Union[int, bool] = 0,
                           train_size:int =0.6, 
                           val_size:int =0.2, 
                           test_size:int =0.2,
                           output_train_file:str ='dataspace/train.csv', 
                           output_val_file:str ='dataspace/val.csv', 
                           output_test_file:str ='dataspace/test.csv'):
    """
    Split a dataset into training, validation, and test sets and save them as CSV files.
    
    Parameters:
    - data (DataFrame): The dataset to split.
    - train_size (float): The proportion of the dataset to include in the train split.
    - val_size (float): The proportion of the dataset to include in the validation split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - train_path (str): File path to save the training set CSV.
    - val_path (str): File path to save the validation set CSV.
    - test_path (str): File path to save the test set CSV.
    
    Returns:
    - train_data (DataFrame): The training set.
    - val_data (DataFrame): The validation set.
    - test_data (DataFrame): The test set.
    """

    # Load data
    data = pd.read_csv(input_data_file,index_col=index_col)

    if not (0 < train_size < 1) or not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("All size parameters must be between 0 and 1.")
        
    if train_size + val_size + test_size != 1:
        raise ValueError("The sum of train_size, val_size, and test_size must equal 1.")
        
    # First split to separate out the test set
    temp_train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Adjust val_size proportion to account for the initial split (for the remaining data)
    adjusted_val_size = val_size / (1 - test_size)
    
    # Second split to separate out the validation set from the temporary training set
    train_data, val_data = train_test_split(temp_train_data, test_size=adjusted_val_size, random_state=42)
    
    # Save to CSV
    train_data.to_csv(output_train_file, index=False)
    val_data.to_csv(output_val_file, index=False)
    test_data.to_csv(output_test_file, index=False)
    
    return train_data, val_data, test_data


def pycaret_train_ts_model(input_file_path:str= 'dataspace/train.csv',
                            # index_col: Union[int, bool] = False,
                            target_column:str = 'target', 
                            use_gpu:bool = False,
                            forecast_horizon:int = 3 , 
                            folds:int = 1, 
                            session_id:int= 123, 
                            output_file_path:str = 'dataspace/pycaret_ts_model.pkl'):
    """
    Trains a time series model using PyCaret based on the provided CSV data and saves the model.
    
    Parameters:
    - input_file_path: str, path to the CSV file containing the time series data.
    - target_column: str, the name of the target column for forecasting.
    - forecast_horizon: int, the number of periods to forecast into the future.
    - folds: int, the number of folds to be used for cross-validation.
    - session_id: int, a random seed for reproducibility.
    - output_file_path: str, path to save the trained model.
    
    Returns:
    - The path where the model was saved.
    """
    
    # Load data
    # df = pd.read_csv(input_file_path,index_col =index_col)
    df = pd.read_csv(input_file_path)
    # Ensure the DataFrame is in the correct format (optional, depending on your CSV structure)
    # df['date_column'] = pd.to_datetime(df['date_column'])
    # df.set_index('date_column', inplace=True)
    
    # Initialize PyCaret setup
    s = setup(data=df, target=target_column, fh=forecast_horizon, fold=folds, session_id=session_id,
              verbose=True, use_gpu = use_gpu)
    
    # Compare models and select the best one
    best_model = compare_models()
    
    # Finalize the model to make it ready for predictions
    final_model = finalize_model(best_model)
    
    # Save the model
    joblib.dump(final_model, model_save_path)
    
    return output_file_path
