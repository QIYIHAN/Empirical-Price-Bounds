import glob
import re
from datetime import datetime, date
import pandas as pd

# Same ticker universe as the super all script
TICKERS = ['AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'PG', 'TSLA', 'V', 'WMT']

frames = []

for ticker in TICKERS:
    # Sub version reads the output of gap_sub_amzn.py
    file_pattern = f'tmp_results/gapData_sub_{ticker}_*.csv'
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
    frames.append(df)

if not frames:
    raise SystemExit("No gapData_sub_* files found under tmp_results/")

out = pd.concat(frames, ignore_index=True)

# sort for readability (only if those columns exist)
sort_cols = [c for c in ['ticker', 'contract_id', 't2', 't1', 't0', 'dist_to_t1'] if c in out.columns]
if sort_cols:
    out = out.sort_values(sort_cols).reset_index(drop=True)

out_file = f'gapData_sub_{date.today().strftime("%d%m%y")}.csv'
out.to_csv(out_file, index=False)
print(f"Saved: {out_file}  (rows={len(out)})")
