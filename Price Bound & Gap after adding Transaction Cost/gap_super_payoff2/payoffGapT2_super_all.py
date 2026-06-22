import glob
import re
from datetime import datetime, date
import pandas as pd

tickerList = ['AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'PG', 'TSLA', 'V', 'WMT']

df_all = []

for ticker in tickerList:
    file_pattern = f'tmp_results/gapData_super_{ticker}_*.csv'
    files = glob.glob(file_pattern)

    date_file_mapping = {}
    for file in files:
        m = re.search(r'_(\d{6})\.csv$', file)
        if m:
            d = datetime.strptime(m.group(1), '%d%m%y')
            date_file_mapping[d] = file

    if not date_file_mapping:
        continue

    newest_date = max(date_file_mapping.keys())
    newest_file = date_file_mapping[newest_date]

    df = pd.read_csv(newest_file)
    df_all.append(df)

if not df_all:
    raise SystemExit("No gapData_super_* files found under tmp_results/")

out = pd.concat(df_all, ignore_index=True)

# sort for readability
sort_cols = [c for c in ['ticker','contract_id','t2','t1','t0','dist_to_t1'] if c in out.columns]
if sort_cols:
    out = out.sort_values(sort_cols).reset_index(drop=True)

out_file = f'gapData_super_{date.today().strftime("%d%m%y")}.csv'
out.to_csv(out_file, index=False)
print(f"Saved: {out_file}  (rows={len(out)})")
