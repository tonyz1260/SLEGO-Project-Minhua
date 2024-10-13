import pandas as pd

def convert_to_long_format(input_file: str = "dataspace/World_Development_Indicators.csv", output_file: str = "dataspace/world_dev_indicator_long.csv", id_vars: list = ['Country Name', 'Country Code', 'Series Name', 'Series Code'], key_name: str = "Year", value_name: str = "Money"):
    df = pd.read_csv(input_file)
    df_long = pd.melt(df, id_vars=id_vars, var_name=key_name, value_name=value_name)
    df_long.to_csv(output_file, index=False)
    return df_long

def convert_to_integer(input_file: str = "dataspace/world_dev_indicator_long.csv", output_file: str = "dataspace/world_dev_indicator_long.csv", columns: list = ['Year']):
    df = pd.read_csv(input_file)
    for column in columns:
        df[column] = df[column].astype(int)
    df.to_csv(output_file, index=False)
    return df

def filter_by_multiple_val(input_file: str = "dataspace/world_dev_indicator_long.csv", output_file: str = "dataspace/gdp_series_all_2019.csv", columns: list = ['Series Name', 'Year'], values: list = ['GDP (current US$)', 2019]):
    df = pd.read_csv(input_file)
    for column, value in zip(columns, values):
        df = df[df[column] == value]
    df.to_csv(output_file, index=False)
    return df

def remove_row_with_na_specified_columns(input_file: str = "dataspace/gdp_series_all_2019.csv", output_file: str = "dataspace/gdp_series_all_2019.csv", columns: list = ['Money']):
    df = pd.read_csv(input_file)
    for column in columns:
        df = df.dropna(subset=[column])
    df.to_csv(output_file, index=False)
    return df

def get_top_n_rows(input_file: str = "dataspace/gdp_series_all_2019.csv", output_file: str = "dataspace/gdp_series_top_5_2019.csv", n: int = 5):
    df = pd.read_csv(input_file)
    df_top_n = df.head(n)
    df_top_n.to_csv(output_file, index=False)
    return df_top_n

def get_bottom_n_rows(input_file: str = "dataspace/gdp_series_all_2019.csv", output_file: str = "dataspace/gdp_series_bottom_5_2019.csv", n: int = 5):
    df = pd.read_csv(input_file)
    df_bottom_n = df.tail(n)
    df_bottom_n.to_csv(output_file, index=False)
    return df_bottom_n

def concatenate_dataframes(input_files: list = ['dataspace/gdp_series_top_5_2019.csv', 'dataspace/gdp_series_bottom_5_2019.csv'], output_file: str = "dataspace/gdp_series_2019.csv"):
    df_list = [pd.read_csv(file) for file in input_files]
    df_concatenated = pd.concat(df_list)
    df_concatenated.to_csv(output_file, index=False)
    return df_concatenated

def filter_by_val_list(input_file: str = "dataspace/world_dev_indicator_long.csv", output_file: str = "dataspace/gdp_series_all.csv", column: str = 'Country Name', value_path: str = "dataspace/gdp_series_2019.csv"):
    df = pd.read_csv(input_file)
    value_df = pd.read_csv(value_path)
    value_list = value_df[column].tolist()
    df = df[df[column].isin(value_list)]
    df.to_csv(output_file, index=False)
    return df

def filter_by_single_val(input_file: str = "dataspace/gdp_series_all.csv", output_file: str = "dataspace/gdp_series_all.csv", column: str = 'Series Name', value: str = 'GDP (current US$)'):
    df = pd.read_csv(input_file)
    df = df[df[column] == value]
    df.to_csv(output_file, index=False)
    return df