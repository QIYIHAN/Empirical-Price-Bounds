import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_details_df = pd.read_csv('all_details_df.csv', index_col=0)

def callSideColumnName(side_name):
    '''
    diff : hedgeValue - actualPayoff (upperBoundCase) or actualPayoff - hedgeValue (lowerBoundCase) at T_2
    distance : upperbound - actualPrice (upperBoundCase) or actualPrice - lowerBound (lowerBoundCase) at t_0
    r : diff(arbitrage payoff) - distance(cost)
    '''
    if side_name == "upper":
        diff_name = 'diff_u'
        r_name = 'net_profit_upper'
        distance_name = 'dist_to_upper'
    elif side_name == "lower":
        diff_name = 'diff_l'
        r_name = 'net_profit_lower'
        distance_name = 'dist_to_lower'
    return diff_name, r_name, distance_name

# === Table 1&2 :  calculate train set performance for given q(c) values ===
def fill_qc_table(df_side, table_df, side_name):

    diff_name, r_name, distance_name = callSideColumnName(side_name)
    c_array = (df_side[diff_name]/df_side['S0']).dropna().values
    c_array.sort()
    total_count = len(df_side)
    if total_count == 0:
        print(f"No valid c for side={side_name}.")
        return

    for q_val in q_list:
        # 1) calculate c value according to the quantiles P(c >= c_val) = q_val
        c_val = np.quantile(c_array, 1 - q_val)

        # 2) select the triggered trades
        subset = df_side[df_side[distance_name]/df_side['S0'] <= c_val]
        num_trades = len(subset)
        subset = subset[subset['timeToMaturity'] != 0]
        # 4) calculate the statistics (annualized per trade)
        avg_return = (subset[r_name]/subset['timeToMaturity']).mean()*252 if num_trades > 0 else np.nan
        std_return = (subset[r_name]/subset['timeToMaturity']).std()*math.sqrt(252) if num_trades > 0 else np.nan

        down_subset = subset[subset[r_name] < 0]
        down_dev = (down_subset[r_name]/down_subset['timeToMaturity']).std()*math.sqrt(252) if len(down_subset) > 0 else np.nan

        sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
        sortino_ratio = avg_return / down_dev if (down_dev != 0 and not np.isnan(down_dev)) else np.nan

        max_return = subset[r_name].max() if num_trades > 0 else np.nan
        min_return = subset[r_name].min() if num_trades > 0 else np.nan

        col_name = f"q(c)={q_val}"
        table_df.loc["c", col_name] = f"{c_val:.4f}"
        table_df.loc["Number of Trades", col_name] = num_trades
        table_df.loc["Avg return", col_name] = f"{avg_return:.4f}" if not pd.isnull(avg_return) else ""
        table_df.loc["Std dev", col_name] = f"{std_return:.4f}" if not pd.isnull(std_return) else ""
        table_df.loc["Down dev", col_name] = f"{down_dev:.4f}" if not pd.isnull(down_dev) else ""
        table_df.loc["Sharpe ratio", col_name] = f"{sharpe_ratio:.4f}" if not pd.isnull(sharpe_ratio) else ""
        table_df.loc["Sortino ratio", col_name] = f"{sortino_ratio:.4f}" if not pd.isnull(sortino_ratio) else ""
        table_df.loc["Max diff", col_name] = f"{max_return:.4f}" if not pd.isnull(max_return) else ""
        table_df.loc["Min diff", col_name] = f"{min_return:.4f}" if not pd.isnull(min_return) else ""
    return table_df

def plot_Qc_distribution(df_side, side_name):
    """
    plot q(c) and normalized average return for different c values
    """

    diff_name, r_name, distance_name = callSideColumnName(side_name)
    valid_side = df_side[~pd.isnull(df_side[diff_name])].copy()
    c_values = np.linspace(0, 0.2, 1000)
    q_vals = []
    avg_return_vals = []
    total_count = len(valid_side)

    for c_val in c_values:
        subset = valid_side[valid_side[distance_name]/valid_side['S0'] <= c_val]
        q_c = len(valid_side[valid_side[diff_name]/valid_side['S0'] >= c_val]) / total_count if total_count > 0 else 0
        q_vals.append(q_c)

        avg_return_vals.append((subset[r_name]/subset['S0']).mean() if len(subset) > 0 else np.nan)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8,6), sharex=True)
    fig.suptitle(f"C-based Distribution for side = {side_name}")

    ax1.plot(c_values, q_vals, color='blue', label="q(c) = P(diff >= c)")
    ax1.set_ylabel("q(c)")
    ax1.set_title("Probability q(c) vs. c")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(c_values, avg_return_vals, color='red', label="Normalized Avg return")
    # ax2.axhline(mean_return, color='blue', linestyle='--', label=f"Mean = {mean_return:.4f}")
    ax2.set_xlabel("c (non-negative)")
    ax2.set_ylabel("Average return")
    ax2.set_title("Average return vs. c")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"figures/Q(c)-Return_{side_name}.png")
    plt.show()

def test_period_results(valid_df, c_val=0.1):
    '''
    test period performance, both upperbound and lowerbound triggers are considered
    '''
    sides = ["upper", "lower"]
    df = pd.DataFrame()
    q_vals = []
    num_trades = 0
    for side in sides:
        diff_name, r_name, distance_name = callSideColumnName(side)
        total_count = len(valid_df)
        # select records with payoff-hedge <= c*S0 (successful trades)
        subset = valid_df[valid_df[distance_name]/valid_df['S0'] <= c_val]
        subset = subset[subset['timeToMaturity'] != 0]
        subset['net_profit'] = subset[r_name]

        df = pd.concat([df,subset])
        num_trades += len(subset)
        # percentage of success
        q_c = len(subset) / total_count
        q_vals.append(1-q_c)

    # calculate the profit
    avg_return = (df['net_profit']/df['timeToMaturity']).mean()*252 if num_trades > 0 else np.nan
    std_return = (df['net_profit']/df['timeToMaturity']).std()*math.sqrt(252) if num_trades > 0 else np.nan

    down_subset = df[df['net_profit'] < 0]
    down_dev = (down_subset['net_profit']/down_subset['timeToMaturity']).std()*math.sqrt(252) if len(down_subset) > 0 else np.nan

    sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
    sortino_ratio = avg_return / down_dev if (down_dev != 0 and not np.isnan(down_dev)) else np.nan

    max_return = df['net_profit'].max() if num_trades > 0 else np.nan
    min_return = df['net_profit'].min() if num_trades > 0 else np.nan

    results = {'q(c)_upper':round(q_vals[0],4),
               'q(c)_lower':round(q_vals[1],4),
               'Number of Trades':int(num_trades),
               'Avg return':round(avg_return,4),
               'Std return':round(std_return,4),
               'Down dev':round(down_dev,4),
               'Sharpe ratio':round(sharpe_ratio,4),
               'Sortino ratio':round(sortino_ratio,4),
               'Max diff':round(max_return,4),
               'Min diff':round(min_return,4)
               }
    col_name = f'testResults-(c={c_val})'
    table_df = pd.DataFrame.from_dict(results, orient='index')
    table_df.columns = [col_name]
    return table_df, df.groupby('T2')['net_profit'].sum()

def plotCumulatedValue(series,c):
    cumulative = series.cumsum()
    cumulative.index = pd.DatetimeIndex(cumulative.index)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative.index, cumulative.values, label='Cumulative Value')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Value')
    plt.title(f'Cumulative Net Value Over Time c={c}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/CumulativeValue_c={c}.png")
    plt.show()


if __name__ == "__main__":
    valid_df = all_details_df[(all_details_df['diff_u'].notna() & (all_details_df['diff_u'] != '')) |
                   (all_details_df['diff_l'].notna() & (all_details_df['diff_l'] != ''))]
    test_set = valid_df.sort_values(by='T2').iloc[int(0.5*valid_df.shape[0]):,:]
    valid_df = valid_df.sort_values(by='T2').iloc[:int(0.5*valid_df.shape[0]),:]

    metrics = [
        "c",
        "Number of Trades",
        "Avg return",
        "Std dev",
        "Down dev",
        "Sharpe ratio",
        "Sortino ratio",
        "Max diff",
        "Min diff"
    ]
    # # ===== fill the train set statistic table for upper and lower bound q(c) =====
    # q_list = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    # upper_table = pd.DataFrame(index=metrics, columns=[f"q(c)={q}" for q in q_list])
    # lower_table = pd.DataFrame(index=metrics, columns=[f"q(c)={q}" for q in q_list])
    # fill_qc_table(valid_df, upper_table, side_name="upper").to_csv('tables/upperTable.csv')
    # fill_qc_table(valid_df, lower_table, side_name="lower").to_csv('tables/lowerTable.csv')
    # #

    # # ===== plot q(c) with net profit =====
    # plot_c_distribution(valid_df)
    # plot_Qc_distribution(valid_df, side_name="upper")
    # plot_Qc_distribution(valid_df, side_name="lower")

    # ===== final result for test period
    table_result, netValue = test_period_results(test_set, c_val=0.01)
    table_df = pd.DataFrame()
    for c in [0, 0.01, 0.025, 0.05, 0.1]:
        table_result, netValue = test_period_results(test_set, c_val=c)
        table_df = pd.concat([table_df,table_result],axis=1)
    table_df.columns = ['c=0%', 'c=1%', 'c=2.5%', 'c=5%', 'c=10%']
    table_df.to_csv('tables/TestPeriodResults.csv')
    # plotCumulatedValue(netValue,0.025)
    # table_result.to_csv('tables/TestPeriodResults_0.025.csv')
    # table_result, netValue = test_period_results(test_set, c_val=0.01)
    # plotCumulatedValue(netValue,0.01)
    # table_result.to_csv('tables/TestPeriodResults_0.01.csv')
