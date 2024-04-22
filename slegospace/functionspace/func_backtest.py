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
