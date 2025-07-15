import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# 定义节假日（dtype 为 datetime64[D]）
nasdaq_holidays = [
    '2018-01-01', '2018-01-15', '2018-02-19', '2018-03-30', '2018-05-28',
    '2018-07-04', '2018-09-03', '2018-11-22', '2018-12-25',
    '2019-01-01', '2019-01-21', '2019-02-18', '2019-04-19', '2019-05-27',
    '2019-07-04', '2019-09-02', '2019-11-28', '2019-12-25',
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
    '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
    '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
    '2022-01-01', '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30',
    '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
    '2023-01-01', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
    '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25'
]
nasdaq_holidays = np.array(nasdaq_holidays, dtype='datetime64[D]')

def read_adjusted_stocks(adjusted_file):
    """
    read adjusted_stocks.csv，convert column Date to type datetime.date，and set as index。
    """
    df = pd.read_csv(adjusted_file)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)
    return df

def parse_filename(filename):
    parts = filename.split('_')
    ticker = parts[0]
    T1_str = parts[1]
    T2_str = parts[2]
    T1_date = datetime.datetime.strptime(T1_str, "%Y%m%d").date()
    T2_date = datetime.datetime.strptime(T2_str, "%Y%m%d").date()
    return ticker, T1_date, T2_date

def check_trading_conditions_to_dataframe(
    upper_file, lower_file, market_file,
    super_diff_file, sub_diff_file,
    ratio, adjusted_file
):
    """
    calculate payoff gap, price gap at T_2 and distance to boundaries for each derivative
    """

    df_upper  = pd.read_csv(upper_file, header=0, index_col=0) ##upper price bound
    df_lower  = pd.read_csv(lower_file, header=0, index_col=0) ##lower price bound
    df_market = pd.read_csv(market_file, header=0, index_col=0) ##simulated market prices
    df_super  = pd.read_csv(super_diff_file, header=0, index_col=0) ##super-hedge payoff gap at T2
    df_sub    = pd.read_csv(sub_diff_file, header=0, index_col=0) ##sub-hedge payoff gap at T2
    
    # 加载 adjusted_stocks 数据
    df_stocks = read_adjusted_stocks(adjusted_file)

    all_files = set(df_upper.index).intersection(
        df_lower.index,
        df_market.index,
        df_super.index,
        df_sub.index
    )
    
    details_list = []
    summary_list = []
    
    for fname in sorted(all_files):
        
        ticker, T1_date, T2_date = parse_filename(fname)
        T1_str = T1_date.strftime("%Y-%m-%d")
        T2_str = T2_date.strftime("%Y-%m-%d")
        
        row_upper  = df_upper.loc[fname]
        row_lower  = df_lower.loc[fname]
        row_market = df_market.loc[fname]
        row_super  = df_super.loc[fname]
        row_sub    = df_sub.loc[fname]
        
        details_for_file = []
        
        for gap_str in row_upper.index:
            try:
                gap = int(gap_str)
            except ValueError:
                continue
            
            try:
                ub = float(row_upper[gap_str])
                lb = float(row_lower[gap_str])
                mkt = float(row_market[gap_str])
                sdiff = float(row_super[gap_str])
                udiff = -float(row_sub[gap_str])
            except:
                continue
            
            # price distance to boundaries, at t_0
            dist_to_upper = ub - mkt if not np.isnan(ub) else np.nan
            dist_to_lower = mkt - lb if not np.isnan(lb) else np.nan
            
            # retrieve t_0 according to time_to_maturity
            try:
                t1_np = np.datetime64(T1_date.isoformat())
                t0_np = np.busday_offset(t1_np, -gap, roll='backward', holidays=nasdaq_holidays)
                t0_date = pd.to_datetime(str(t0_np)).date()
            except Exception as e:
                t0_date = T1_date - datetime.timedelta(days=gap)
            t0_str = t0_date.strftime("%Y-%m-%d")
            
            # 从 adjusted_stocks 中获取 t0 对应的 S0（若找不到，则为 NaN）
            try:
                S0 = float(df_stocks.at[t0_date, ticker])
            except KeyError:
                S0 = np.nan
            
            diff_val_upper = sdiff
            diff_val_lower = udiff

            timeToMaturity = np.busday_count(pd.to_datetime(t0_str).to_numpy().astype('datetime64[D]'),
                                             pd.to_datetime(T1_str).to_numpy().astype('datetime64[D]'),
                                             holidays=nasdaq_holidays)
            details_for_file.append({
                "FileName": fname,
                "T1": T1_str,
                "T2": T2_str,
                "gap": gap,
                "timeToMaturity": timeToMaturity,
                "t0": t0_str,
                "S0": S0,
                "ratio": ratio,
                "UB": ub,
                "LB": lb,
                "MKT": mkt,
                "dist_to_upper": dist_to_upper,
                "dist_to_lower": dist_to_lower,
                "diff_u": diff_val_upper,
                "diff_l": diff_val_lower,
            })
        details_list.extend(details_for_file)
    
    details_df = pd.DataFrame(details_list)
    return details_df


if __name__ == "__main__":
    upper_file = "priceBound_upper.csv"
    lower_file = "priceBound_lower.csv"
    market_file= "market_Price.csv"
    super_diff_file = "recovered_super_gap.csv"
    sub_diff_file   = "recovered_sub_gap.csv"
    adjusted_file   = "adjusted_stocks.csv"


    ratio = 0.005
    all_details_df= check_trading_conditions_to_dataframe(
        upper_file=upper_file,
        lower_file=lower_file,
        market_file=market_file,
        super_diff_file=super_diff_file,
        sub_diff_file=sub_diff_file,
        ratio=ratio,
        adjusted_file=adjusted_file
    )


all_details_df['net_profit_upper'] = all_details_df['diff_u'] - all_details_df['dist_to_upper']
all_details_df['net_profit_lower'] = all_details_df['diff_l'] - all_details_df['dist_to_lower']
all_details_df.to_csv('all_details_df.csv')