import glob
import re
from datetime import datetime, date
import pandas as pd


tickerList = ['AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'PG', 'TSLA', 'V', 'WMT']

df_priceBound = pd.DataFrame()

for ticker in tickerList:
    file_pattern = f'tmp_results/priceBoundData_{ticker}_*.csv'
    files = glob.glob(file_pattern)

    # Dictionary to store dates and file names
    date_file_mapping = {}

    # Extract dates from file names
    for file in files:
        match = re.search(r'_(\d{6})\.csv$', file)  # Find the date in DDMMYY format
        if match:
            date_str = match.group(1)  # Extract the date part
            file_date = datetime.strptime(date_str, '%d%m%y')  # Convert to datetime
            date_file_mapping[file_date] = file

    # Find the newest file for the ticker
    if date_file_mapping:
        newest_date = max(date_file_mapping.keys())
        newest_file = date_file_mapping[newest_date]

        df = pd.read_csv(newest_file, index_col=0)
        df_priceBound = pd.concat([df_priceBound, df])
    else:
        continue

df_priceBound.columns = df_priceBound.columns.astype(int)
df_priceBound = df_priceBound.sort_index(axis=1).copy()
df_priceBound.to_csv(f'priceBoundData_{date.today().strftime("%d%m%y")}.csv')

