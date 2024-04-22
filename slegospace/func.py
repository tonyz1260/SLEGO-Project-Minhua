import os
import ast
import panel as pn
import inspect

def test_function(input_string:str='Hello!', 
          output_file_path:str='dataspace/output.txt'):
    """
    A simple function to save the provided input string to a specified text file and return the string.

    Parameters:
    - input_string (str): The string to be saved.
    - output_file_path (str): The file path where the string should be saved.

    Returns:
    - str: The same input string.
    """

    # Open the file at the specified path in write mode
    with open(output_file_path, 'w') as file:
        # Write the input string to the file
        file.write(input_string)

    # Return the input string
    return input_string

def _compute(module_name, input):
    module = __import__(module_name)

    pipeline_dict = ast.literal_eval(input)
    output = ""
    for function_name, parameters in pipeline_dict.items():
        function = eval(f"module.{function_name}")
        result = function(**parameters)

        output += "\n===================="+function_name+"====================\n\n"
        output += str(result)

    return output

def _create_multi_select_combobox(target_module):
    """
    Creates a multi-select combobox with all functions from the target_module.
    """
    
    # Get the module name (e.g., "func" if your module is named func.py)
    module_name = target_module.__name__
    
    # Get a list of all functions defined in target_module
    functions = [name for name, obj in inspect.getmembers(target_module, inspect.isfunction)
                 if obj.__module__ == module_name and not name.startswith('_')]

    # Create a multi-select combobox using the list of functions
    multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions, height=150)

    return multi_combobox


# def _create_multi_select_combobox(func):
#   """
#   Creates a multi-select combobox with all functions from the func.py file.
#   """

#   # Get a list of all functions in the func.py file.
#   functions = [name for name, obj in inspect.getmembers(func)
#                 if inspect.isfunction(obj) and not name.startswith('_')]

#   # Create a multi-select combobox using the list of functions.
#   multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions,  height=150)

#   return multi_combobox


def _extract_parameter(func):
    """
    Extracts the names and default values of the parameters of a function as a dictionary.

    Args:
        func: The function to extract parameter names and default values from.

    Returns:
        A dictionary where the keys are parameter names and the values are the default values.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    parameter_dict = {}
    for name, param in parameters.items():
        if param.default != inspect.Parameter.empty:
            parameter_dict[name] = param.default
        else:
            parameter_dict[name] = None

    return parameter_dict




from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Union
import joblib
# Import necessary libraries



def merge_csv_by_index(input_csv_file1: str = 'dataspace/dataset1.csv',
                       input_csv_file2: str = 'dataspace/dataset2.csv',
                       output_csv_file: str = 'dataspace/merged_dataset.csv',
                       rename_dict1: dict = None,
                       rename_dict2: dict = None,
                       how: str = 'outer') -> pd.DataFrame:
    """
    Merges two CSV files horizontally based on their index after optionally renaming columns in each.

    Parameters:
    input_csv_file1 (str): Path to the first CSV file. Default path is provided.
    input_csv_file2 (str): Path to the second CSV file. Default path is provided.
    output_csv_file (str): Path where the merged CSV file will be saved. Default path is provided.
    rename_dict1 (dict): Dictionary to rename columns of the first CSV file.
    rename_dict2 (dict): Dictionary to rename columns of the second CSV file.
    how (str): Type of merge to perform ('outer', 'inner', 'left', 'right'). Default is 'outer'.

    Returns:
    pd.DataFrame: A dataframe containing the merged data.
    """
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(input_csv_file1)
    df2 = pd.read_csv(input_csv_file2)

    # Rename columns if dictionaries are provided
    if rename_dict1:
        df1.rename(columns=rename_dict1, inplace=True)
    if rename_dict2:
        df2.rename(columns=rename_dict2, inplace=True)

    # Merge the DataFrames based on the index
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how=how)
    # Drop any 'Unnamed' columns that may exist
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_csv_file, index=False)

    return merged_df




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




import ast
import pandas as pd
import datetime
from yfinance import Ticker as si
from typing import Union, Callable
import plotly.graph_objects as go
from typing import Union

'''
Guidelines for building microservices (python functions):

1. Create a python function
2. Make sure the function has a docstring that explain to user how to use the microservice, and what the service does
3. Make sure the function has a return statement
4. Make sure the function has a parameter
5. Make sure the function has a default value for the parameter
6. Make sure the function has a type hint for the parameter 

'''
def import_marketdata_yahoo_csv(ticker: str = 'msft', 
                                start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                end_date: str = datetime.datetime.now().strftime("%Y-%m-%d"), 
                                input_file_path: str = '',
                                output_file_path: str = 'dataspace/dataset.csv' ):
    """
    Imports market data from Yahoo using the yfinance Ticker API.
    
    Parameters:
    - ticker (str): The stock ticker symbol. Default is 'msft'.
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format. Default is one year ago from today.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format. Default is today.
    - output_file_path (str): The path to save the resulting CSV file. Default is 'yfinance_ohlc.csv'.
    
    Returns:
    - pd.DataFrame: A dataframe containing the imported market data.
    """

    # Fetch the market data using the yfinance Ticker
    df = si(ticker).history(start=start_date, end=end_date)
    df['Datetime'] = df.index
    df.to_csv(output_file_path)
    # Return the StringIO object
    return df

def preprocess_filling_missing_values(
    input_file_path: str = 'dataspace/dataset.csv',
    output_file_path: str = 'dataspace/dataset.csv',
    fill_strategy: Union[str, float, int, dict, Callable] = 'ffill',
):
    """
    Preprocesses a CSV dataset by filling missing values using various strategies and outputs the processed dataset to a CSV file.

    Parameters:
    - input_file_path (str): Path to the input CSV dataset. Defaults to 'dataspace/dataset.csv'.
    - output_file_path (str): Path for the output CSV dataset where processed data will be saved. Defaults to 'dataspace/dataset.csv'.
    - fill_strategy (Union[str, float, int, dict, Callable]): Defines the strategy for filling missing values. Can be a string ('ffill' for forward fill, 'bfill' for backward fill), a scalar value to fill missing entries, a dictionary to specify method or value per column, or a callable with custom logic. Defaults to 'ffill'.

    Returns:
    pandas.DataFrame: The processed DataFrame with missing values filled.

    This function reads a CSV file, fills missing values according to the specified strategy, and writes the processed DataFrame to the specified CSV file. It supports various filling strategies to accommodate different data preprocessing needs.
    """

    data = pd.read_csv(input_file_path)

    if callable(fill_strategy):
        # Apply a custom function to fill missing values
        data.apply(lambda x: x.fillna(fill_strategy(x)), axis=0)
    else:
        # Apply predefined pandas strategies or scalar values
        data.fillna(value=fill_strategy, inplace=True)

    data.to_csv(output_file_path)

    return data


def plot_chart_local(input_file_path: str = 'dataspace/dataset.csv', 
                     index_col: Union[None, int, str] = 0,
                     x_column: str = 'Date', 
                     y_column: str = 'SMA_Close', 
                     title: str = 'Data Plot', 
                     legend_title:str= 'Legend',
                     mode:str = 'lines',
                     output_html_file_path: str = 'dataspace/dataset_plot.html'):
    """
    Create a Plotly graph for data and save it as an HTML file locally.
    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset.csv'.
    :param x_column: Column to use for the X-axis. If None, the data index is used. Default is 'Date'.
    :param y_column: Column to use for the Y-axis. Default is 'SMA_Close'.
    :param title: Title of the plot. Default is 'Data Plot'.
    :param output_html_file_path: Local file path where the HTML file will be saved. Default is 'dataspace/dataset_plot.html'.
    """

    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=index_col)

    # Determine X-axis data
    x_data = data[x_column] if x_column else data.index

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=data[y_column], mode=mode, name=y_column))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=legend_title
    )

    # Save the figure as an HTML file locally
    fig.write_html(output_html_file_path)

    return "Plot saved locally."

def compute_simple_moving_average_local(input_file_path: str = 'dataspace/dataset.csv', 
                                        column_name: str = 'Close', 
                                        window_size: int = 20, 
                                        output_file_path: str = 'dataspace/dataset.csv'):
    """
    Compute simple moving average for a specified column in the data and save updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset.csv'.
    :param column_name: Name of the column to calculate the SMA on. Default is 'Close'.
    :param window_size: Window size for the moving average. Default is 20.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_sma.csv'.
    """
    # Load data from a local file
    
    data = pd.read_csv(input_file_path, index_col=0)

    # Calculate Simple Moving Average
    if column_name in data.columns:
        data[f'SMA_{column_name}'] = data[column_name].rolling(window=window_size).mean()
    else:
        raise ValueError(f"Column '{column_name}' not found in data")

    # Save updated data to a local file
    data.to_csv(output_file_path)

    return data




import json
import openai
import os
import PyPDF2
from docx import Document
from openpyxl import load_workbook
from PIL import Image
import pytesseract
import csv
import json
import openai
import os
import PyPDF2
from docx import Document
from openpyxl import load_workbook
from PIL import Image
import pytesseract
import csv


def chatgpt_chat(model:str='gpt-3.5-turbo',
                  user_input_file:str='dataspace/user_text_input.txt',
                  output_text_file:str='dataspace/gpt_output_text.txt',
                  output_json_file:str='dataspace/gpt_output_full.json',
                  temperature:int=1, 
                  max_tokens:int=256, 
                  top_p:int=1, 
                  frequency_penalty:int=0, 
                  presence_penalty:int=0,
                  api_key:str='sk-CiO5GzpXbxZQsMuKEQEkT3BlbkFJz4LS3FuI3f5NqmF1BXO', 
                  user_message:str='Summarize:',):
  
    '''
    This function interfaces with OpenAI's GPT model to process a text input and obtain a generated response.
    It reads an additional input from a file, merges it with the user's direct input, and sends the combined
    content to the API. The response is then saved to both text and JSON files.

    Parameters:
        api_key (str): Your OpenAI API key.
        model (str): Identifier for the model version to use.
        user_message (str): Direct user input as a string.
        user_input_file (str): Path to a file containing additional text to send to the API.
        output_file (str): Path to save the plain text response from the API.
        output_json_file (str): Path to save the full JSON response from the API.
        temperature (float): Controls randomness in response generation. Higher is more random.
        max_tokens (int): Maximum length of the response.
        top_p (float): Controls diversity via nucleus sampling: 0.1 means only top 10% probabilities considered.
        frequency_penalty (float): Decreases likelihood of repeating words.
        presence_penalty (float): Decreases likelihood of repeating topics.

    Returns:
        str: The text response from the API.
    '''

    # Initialize variables to store responses
    ans = None
    ans_dict = {}

    combined_message = user_message  # Start with the user's direct message

    # Determine the file type and read content accordingly
    if os.path.exists(user_input_file):
        file_extension = user_input_file.split('.')[-1].lower()
        if file_extension == 'txt':
            with open(user_input_file, 'r') as file:
                file_contents = file.read().strip()
            combined_message += f"\n\n==== Text File Input ====\n\n{file_contents}"
        elif file_extension == 'pdf':
            with open(user_input_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_contents = ' '.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            combined_message += f"\n\n==== PDF File Input ====\n\n{pdf_contents}"
        elif file_extension == 'docx':
            doc = Document(user_input_file)
            docx_contents = ' '.join([para.text for para in doc.paragraphs])
            combined_message += f"\n\n==== DOCX File Input ====\n\n{docx_contents}"
        elif file_extension == 'xlsx':
            workbook = load_workbook(filename=user_input_file)
            sheet = workbook.active
            xlsx_contents = ' '.join([str(cell.value) for row in sheet for cell in row if cell.value is not None])
            combined_message += f"\n\n==== XLSX File Input ====\n\n{xlsx_contents}"
        elif file_extension in ['png', 'jpg', 'jpeg']:
            img = Image.open(user_input_file)
            image_text = pytesseract.image_to_string(img)
            combined_message += f"\n\n==== Image File Input (OCR) ====\n\n{image_text}"
        elif file_extension == 'json':
            with open(user_input_file, 'r') as file:
                json_data = json.load(file)
                json_contents = json.dumps(json_data, indent=4)
            combined_message += f"\n\n==== JSON File Input ====\n\n{json_contents}"
        elif file_extension == 'csv':
            with open(user_input_file, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                csv_contents = ' '.join([','.join(row) for row in reader])
            combined_message += f"\n\n==== CSV File Input ====\n\n{csv_contents}"

    
    try:
        # Set the API key (consider using environment variables for security)
        openai.api_key = api_key
        
        # Create a chat completion request with the specified parameters
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_message}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Extract and process the response
        ans_dict = response.to_dict()
        if 'choices' in ans_dict and len(ans_dict['choices']) > 0:
            if 'message' in ans_dict['choices'][0]:
                ans = ans_dict['choices'][0]['message']['content']

        # Save the text response and the full JSON response
        if ans:
            with open(output_text_file, 'w') as f:
                f.write(ans)
        with open(output_json_file, 'w') as json_file:
            json.dump(ans_dict, json_file, indent=4)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return ans

import sweetviz as sv
import pandas as pd

from autoviz import AutoViz_Class
from typing import Union


def generate_sweetviz_report(input_csv: str='dataspace/dataset.csv', 
                             output_html: str = 'dataspace/sweetviz_report.html'):
    """
    Generates a Sweetviz report from a specified CSV file and saves it as an HTML file.

    Parameters:
    input_csv (str): Path to the input CSV dataset.
    output_html (str): Path where the HTML report will be saved, default is 'report.html'.

    Returns:
    None: The function saves the HTML report to the specified path and does not return any value.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)

    # Analyze the dataset
    report = sv.analyze(data)

    # Save the report to an HTML file
    report.show_html(output_html)

    return 'Report has been generated!'



def autoviz_plot(input_file_path: str = 'dataspace/AirQuality.csv', 
                 target_variable: Union[str, None] = 'CO(GT)', 
                 custom_plot_dir: str = 'dataspace',
                 max_rows_analyzed: int = 150000,
                 max_cols_analyzed: int = 30,
                 lowess: bool = False,
                 header: int = 0,
                 verbose: int = 2,
                 sep: str = "e"):
    """
    Generates visualizations for the dataset using AutoViz.

    Parameters:
    input_file_path (str): Path to the input CSV dataset.
    target_variable (Union[str, None]): Target variable for analysis. If None, no specific target.
    custom_plot_dir (str): Directory where plots will be saved.
    max_rows_analyzed (int): Maximum number of rows to analyze.
    max_cols_analyzed (int): Maximum number of columns to analyze.
    lowess (bool): Whether to use locally weighted scatterplot smoothing.
    header (int): Row number to use as the column names.
    verbose (int): Verbosity level.
    sep (str): Separator used in the CSV file.

    Returns:
    str: Message indicating the completion of the visualization process.
    """
    AV = AutoViz_Class()

    # Perform the AutoViz analysis and generate the plots
    dft = AV.AutoViz(
        filename=input_file_path,
        sep=sep,
        depVar=target_variable,
        dfte=None,
        header=header,
        verbose=verbose,
        lowess=lowess,
        chart_format="html",
        max_rows_analyzed=max_rows_analyzed,
        max_cols_analyzed=max_cols_analyzed,
        save_plot_dir=custom_plot_dir)

    return "Visualizations have been generated and saved!"
from ucimlrepo import fetch_ucirepo 
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

import requests
from bs4 import BeautifulSoup

def webscrape_to_txt(url: str = "https://au.finance.yahoo.com/", 
                            output_filename: str = "dataspace/output_webscrape.txt"):
    """
    Fetches the content from the specified URL and saves the textual content 
    into a text file, stripping out all HTML tags.

    Parameters:
    - url (str): The URL from which to fetch the content. Default is Yahoo Finance homepage.
    - output_filename (str): The path to the file where the text content will be saved.

    Returns:
    - None: Outputs a file with the extracted text content.
    """
    try:
        # Send a HTTP request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text using .get_text()
            text_content = soup.get_text(separator='\n', strip=True)
            # Open a file in write mode
            with open(output_filename, 'w', encoding='utf-8') as file:
                file.write(text_content)
            print(f"Text content saved successfully to {output_filename}")
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

        return text_content
    except Exception as e:
        print(f"An error occurred: {e}")


import arxiv
import csv
from typing import List, Dict
import requests

def download_papers_from_arxiv_csv(filename: str = "dataspace/latest_papers.csv", 
                                    download_folder: str = "dataspace/papers/") -> None:
    """
    Downloads papers listed in a CSV file from arXiv.

    Args:
    filename (str): The path to the CSV file containing paper metadata.
    download_folder (str): The directory where the downloaded papers will be saved.

    The CSV file must contain at least the 'url' and 'title' fields.
    """

    try:
        # Create a directory to save downloaded papers
        os.makedirs(download_folder, exist_ok=True)
        
        # Read the CSV file to get the URLs
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                url = row['url']
                paper_title = row['title'].replace('/', '_')  # Replace any slashes in title to avoid file path errors
                file_path = f"{download_folder}{paper_title}.pdf"
                
                # Download the paper
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded '{paper_title}' successfully.")
                else:
                    print(f"Failed to download '{paper_title}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def search_arxiv_papers(search_query: str='machine learning', 
                           filename: str = "dataspace/latest_papers.csv", 
                           max_results: int = 5):

    """
    Searches for papers on arXiv and saves the results to a CSV file.

    Args:
    search_query (str): The query term to search for on arXiv.
    filename (str): Path to save the CSV file containing the search results.
    max_results (int): Maximum number of results to fetch and save.

    Returns:
    list: A list of dictionaries, each containing paper details.

    The function saves a CSV with fields: title, authors, abstract, published, url.
    """

    # Create a search object
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # Fetch the results
    results = list(search.results())
    
    # Format the results into a readable list and write to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ["title", "authors", "abstract", "published", "url"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            paper_info = {
                "title": result.title,
                "authors": ', '.join(author.name for author in result.authors),
                "abstract": result.summary.replace('\n', ' '),
                "published": result.published.strftime('%Y-%m-%d'),
                "url": result.entry_id
            }
            writer.writerow(paper_info)
    return results

import pandas as pd
import vectorbt as vbt
from typing import Union
import quantstats as qs
import yfinance as yf

def moving_avg_cross_signal(input_file_path: str = 'dataspace/dataset.csv', 
                            column: str = 'Close',
                            index_col: Union[int, bool] = 0,
                            short_ma_window: int = 10,
                            long_ma_window: int = 50,
                            output_file_path:str ='dataspace/moving_avg_cross_signal.csv'

                            ):
    """Calculate moving average cross signals from a CSV file and return as a DataFrame.
    
    Args:
        input_file_path (str): Path to the CSV file containing the stock data.
        column (str): The name of the column to use for calculating moving averages.
    
    Returns:
        DataFrame: A DataFrame containing the entry and exit signals.
    """
    # Load the dataset
    df = pd.read_csv(input_file_path, index_col=index_col)
    
    # Drop any 'Unnamed' columns that may exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Calculate short-term and long-term moving averages
    short_ma = vbt.MA.run(df[column], window= short_ma_window)
    long_ma = vbt.MA.run(df[column], window= long_ma_window  )

    # Generate entry and exit signals
    entries = short_ma.ma_crossed_above(long_ma)
    exits = short_ma.ma_crossed_below(long_ma)

    # Convert boolean arrays to DataFrame for better handling and visualization
    signals = pd.DataFrame({
        'Entries': entries,
        'Exits': exits
    })
    #combine signal and df
    signals = pd.concat([df, signals], axis=1)

    # save csv
    signals.to_csv(output_file_path)
    return signals


def vbt_sginal_backtest(input_signal_file:str='dataspace/moving_avg_cross_signal.csv',
                             price_col:str='Close',
                             entries_col:str='Entries',
                             exits_col:str='Exits',
                             freq:str='D',
                             output_stats_file:str='dataspace/backtest_stats.csv',
                             output_return_file:str='dataspace/backtest_returns.csv'):
    """
    Create and evaluate a trading portfolio based on moving average crossover signals.
    
    This function uses vectorbt to create a portfolio from entry and exit signals
    based on moving average crossovers. It calculates the portfolio statistics and
    returns, and saves them to CSV files.

    Parameters:
        input_signal_file (str): Path to the CSV file containing price data and signals.
        price_col (str): Column name for the price data in the CSV file.
        entries_col (str): Column name for entry signals in the CSV file.
        exits_col (str): Column name for exit signals in the CSV file.
        freq (str): Frequency of the data, used for modeling in vectorbt.
        output_stats_file (str): Path where portfolio statistics will be saved.
        output_return_file (str): Path where portfolio returns will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the statistics of the portfolio.
    """

    # Load data from CSV
    data = pd.read_csv(input_signal_file)

    # Extract price and signals from the data
    price = data[price_col]
    entries = data[entries_col].astype(bool)
    exits = data[exits_col].astype(bool)

    # Create a portfolio from the signals
    portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq=freq)
    
    # Save portfolio statistics to a CSV file
    stats = portfolio.stats()
    stats.to_csv(output_stats_file)

    # Compute returns and save them to a CSV file
    ret = portfolio.returns()
    ret_df = ret.to_frame(name='returns')  # Convert Series to DataFrame and name the column 'return'
    
    # Combine the original data with the returns for comprehensive output
    result = pd.concat([data, ret_df], axis=1)
    result.to_csv(output_return_file)

    # Return the statistics DataFrame
    return stats,portfolio


def backtest_viz_with_quantstats(input_file: str = 'dataspace/backtest_returns.csv', 
                                 output_file: str = 'dataspace/quantstats_results.html',
                                 benchmark_file_path:str = 'None',
                                 benchmark_col:str = 'None',
                                 return_col: str = 'returns', 
                                 time_col: str = 'Date',
                                 periods_per_year: int = 252, 
                                 compounded: str = 'True', 
                                 rf: float = 0.02,
                                 mode:str='full',
                                 title: str = 'Backtest Report Comparing Against SPY Benchmark'):
    """
    Perform a backtest visualization using QuantStats on prepared return data, optionally comparing
    to a benchmark. Defaults to SPY if no benchmark provided.

    Parameters:
        input_file (str): Path to the CSV file containing the returns data.
        output_file (str): Path to save the HTML report of the backtest results.
        return_col (str): Name of the column containing return data in the input file.
        benchmark_file_path (str, optional): Path to the CSV file containing the benchmark data.
        benchmark_col (str, optional): Column name for the benchmark data.
        periods_per_year (int): Number of periods per year for annualization (default 252 for daily data).
        compounded (bool): Whether the returns are to be treated as compounded (default True).
        rf (float): Risk-free rate used for calculating certain metrics like the Sharpe Ratio.
        time_col (str): Column containing datetime information in the input file.
        title (str): Title for the HTML report.

    Returns:
        str: Confirmation message indicating that the backtest report has been generated.
    """
    # Load historical price data from a CSV file
    data = pd.read_csv(input_file)
    data[time_col] = pd.to_datetime(data[time_col], utc=True)
    data.set_index(time_col, inplace=True)
    data.index = data.index.tz_localize(None)  # Assuming 'data' is your main DataFrame


    # Fetch SPY data from Yahoo Finance as default benchmark
    if benchmark_file_path == 'None' or benchmark_col == 'None':
        spy = yf.download('SPY', start=data.index.min(), end=data.index.max())
        benchmark = spy['Adj Close'].pct_change().dropna()
    else:
        benchmark_data = pd.read_csv(benchmark_file_path)
        benchmark_data[time_col] = pd.to_datetime(benchmark_data[time_col], utc=True)
        benchmark_data.set_index(time_col, inplace=True)
        benchmark = benchmark_data[benchmark_col]

    #rename benchamrk col
    

    # Use QuantStats to extend pandas functionality to financial series
    qs.extend_pandas()

    compounded = eval(compounded)
    # Analyze the strategy's returns and generate a report
    # Analyze the strategy's returns and generate a report
    qs.reports.html(data[return_col], 
                    benchmark=benchmark, 
                    rf=rf/periods_per_year, 
                    output=output_file, 
                    title=title, 
                    compounded=compounded, 
                    mode=mode,  # Adjust as needed
                    grayscale=False, 
                    display=False)

    return 'Backtest report generated!'

import pandas as pd
from autogluon.tabular import TabularPredictor
import pickle

def train_autogluon_tabular(input_file_path:str='dataspace/train.csv',
                               target:str='target',
                               train_frac:float=0.8,
                               random_state:int=42,
                               performance_output_path:str='dataspace/performance_autogluon.txt',
                               model_save_path:str='dataspace/autogluon_model.pkl'):
    """
    Trains a classification model using AutoGluon on the specified dataset and saves the trained model.

    Parameters:
        input_file_path (str): Path to the CSV file containing the dataset.
        target (str): Name of the column to predict.
        train_frac (float): Fraction of the dataset to use for training.
        random_state (int): Seed for the random number generator for reproducibility.
        performance_output_path (str): Path to save the text file containing model performance metrics.
        model_save_path (str): Path to save the trained AutoGluon model.

    Returns:
        Tuple[TabularPredictor, dict]: A tuple containing the trained AutoGluon TabularPredictor and 
                                       a dictionary with performance metrics.

    Saves the trained model using AutoGluon's built-in save mechanism and optionally as a pickle file.
    Evaluates the model's performance and writes metrics to the specified text file.
    """  
    # Load the dataset from a CSV file
    df = pd.read_csv(input_file_path)
    df = df.loc[:, ~df.columns.str.contains('Unnamed: 0', case=False)]
    train_data = df.sample(frac=train_frac, random_state=random_state)
    test_data = df.drop(train_data.index)

    # Train a classifier with AutoGluon
    predictor = TabularPredictor(label=target).fit(train_data)
    performance = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=True)

    # Write the performance metrics and leaderboard to a file
    with open(performance_output_path, 'w') as f:
        f.write(str(performance))
        f.write("\n")
        f.write(str(leaderboard))

    # Save the trained model using AutoGluon's method
    predictor.save(model_save_path)

    # Optionally, save the model using pickle
    pickle_path = model_save_path
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictor, f)

    return predictor, performance

def autogluon_model_predict(pickle_path:str= 'dataspace/autogluon_model.pkl',
                        input_data_path:str='dataspace/test.csv',
                        output_data_path:str='dataspace/autogluon_predict.csv' ):
    """
    Loads a pickled AutoGluon model from the specified path.

    Parameters:
        pickle_path (str): Path to the pickled model file.

    Returns:
        TabularPredictor: The loaded AutoGluon model.

    Note:
        Loading models via pickle can lead to issues if there are mismatches in library versions or 
        if the saved model includes elements that pickle cannot handle properly. It is generally 
        recommended to use AutoGluon's native load functionality unless there is a specific need 
        for pickle.
    """

    with open(pickle_path, 'rb') as f:
        loaded_model = pickle.load(f)

    predictions = loaded_model.predict(input_data_path)
        # Save the predictions to a CSV file
    predictions.to_csv(output_data_path, index=False)


    return predictions


