import math
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import gurobipy as gp
from gurobipy import GRB
import os
import re
from datetime import date
import warnings
import seaborn as sns

import time


options = {
    "WLSACCESSID": "31598bc4-0b23-474f-86d4-af032d437137",
    "WLSSECRET": "bd7ad348-f8d6-452e-b615-ebc88d920e41",
    "LICENSEID": 2498404,
}

warnings.filterwarnings('ignore')


# load data
stocks = pd.read_csv('adjusted_stocks.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])


IRParams = pd.read_csv('interest_rates_parameters.csv', parse_dates=['Date'], dayfirst=True)
IRParams = IRParams.fillna(method='ffill')

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

# def function

def payoff(S1, S2):
    return np.maximum(S2 - S1, 0)


def IR_effect(init_date, start_date, end_date):
    beta0 = IRParams.loc[IRParams.Date==init_date, 'BETA0'].item()
    beta1 = IRParams.loc[IRParams.Date==init_date, 'BETA1'].item()
    beta2 = IRParams.loc[IRParams.Date==init_date, 'BETA2'].item()
    tau1 = IRParams.loc[IRParams.Date==init_date, 'TAU1'].item()

    # 将 start_date 和 end_date 转换为天精度
    start_date = np.datetime64(start_date, 'D')
    end_date = np.datetime64(end_date, 'D')

    tau = np.busday_count(start_date, end_date, holidays=nasdaq_holidays) / 252
    if tau == 0.0:
        r = 0.0
    else:
        r = beta0 + beta1*(1-math.exp(-tau/tau1))/(tau/tau1) + beta2*((1-math.exp(-tau/tau1))/(tau/tau1)-math.exp(-tau/tau1))
    return math.exp(r/100*tau)


def hedgeGapOnLargeGrid(df_t0, t0, t1, t2, S1_N, S2_N, parameters):
    K1 = df_t0['K1']
    K2 = df_t0['K2']
    S0 = df_t0['Adj_S0'].unique()

    M_S1 = np.zeros((len(S1_N), len(S2_N)))
    for i in range(len(S2_N)):
        M_S1[:, i] = S1_N

    M_S2 = np.zeros((len(S1_N), len(S2_N)))
    for i in range(len(S1_N)):
        M_S2[i, :] = S2_N

    M_S1_S0 = np.zeros((len(S1_N), len(S2_N)))
    for i in range(len(S2_N)):
        M_S1_S0[:, i] = S1_N - S0 * IR_effect(t0, t0, t1)

    M_S2_S1 = np.zeros((len(S1_N), len(S2_N)))
    for i in range(len(S2_N)):
        M_S2_S1[:, i] = S2_N[i] - S1_N * IR_effect(t0, t1, t2)

    M_hedge = (
            IR_effect(t0, t0, t2) * parameters['d']
            + sum(IR_effect(t0, t1, t2) * (parameters['theta1_ask'][r] - parameters['theta1_bid'][r]) * np.maximum(M_S1 - K1[r], 0) for r in df_t0.index)
            + sum(IR_effect(t0, t2, t2) * (parameters['theta2_ask'][r] - parameters['theta2_bid'][r]) * np.maximum(M_S2 - K2[r], 0) for r in df_t0.dropna().index)
            + parameters['delta0'] * M_S1_S0
            + parameters['delta1'](S1_N).reshape(len(S1_N), 1) * M_S2_S1
            # - 0.0005 * S1_N * np.abs(parameters['delta1'](S1_N) - parameters['delta0'])
    )

    M_payoff = np.zeros((len(S1_N), len(S2_N)))
    for i in range(len(S2_N)):
        M_payoff[:, i] = payoff(S1_N, S2_N[i])

    M_gap = M_hedge - M_payoff

    return M_hedge, M_gap


# parameters of an LP model
def params(df_t0, t0, t1, t2, S1, S2, init=None):
    C1_ask = df_t0['C1_ask']
    C1_bid = df_t0['C1_bid']
    C2_ask = df_t0['C2_ask']
    C2_bid = df_t0['C2_bid']
    K1 = df_t0['K1']
    K2 = df_t0['K2']
    S0 = df_t0['Adj_S0'].unique()

    m = gp.Model("Price Bounds")
    m.setParam('OutputFlag', 0)

    # cash position
    d = m.addVars(['d'], lb=-float('inf'), name="d")
    # options position: only long
    theta1_ask = m.addVars(df_t0.index, lb=0.0, name="theta1_ask")
    theta1_bid = m.addVars(df_t0.index, ub=0.0, name="theta1_bid")
    theta2_ask = m.addVars(df_t0.index, lb=0.0, name="theta2_ask")
    theta2_bid = m.addVars(df_t0.index, ub=0.0, name="theta2_bid")
    # stock position
    delta0 = m.addVars(['delta0'], lb=-float('inf'), name="delta0")
    delta1 = m.addVars(np.array(S1).flatten(), lb=-float('inf'), name="delta1")

    if init != None:
        d['d'].start = init['d']

        for r in df_t0.index:
            theta1_ask[r].start = init['theta1_ask'][r]
            theta1_bid[r].start = init['theta1_bid'][r]
            if r in init['theta2_ask'].keys():
                theta2_ask[r].start = init['theta2_ask'][r]
                theta2_bid[r].start = init['theta2_bid'][r]

        delta0['delta0'].start = init['delta0']

        for i in np.array(S1).flatten():
            delta1[i].start = init['delta1'](i)

        m.update()

    m.setObjective(d['d']
                   + sum(theta1_ask[r] * C1_ask[r] - theta1_bid[r] * C1_bid[r] for r in df_t0.index)
                   + sum(theta2_ask[r] * C2_ask[r] - theta2_bid[r] * C2_bid[r] for r in df_t0.dropna().index)
                   )
    m.ModelSense = GRB.MINIMIZE

    for S1_i in S1:
        for S2_j in S2:
            # diff_delta = m.addVar(name="diff_delta")
            # abs_diff_delta = m.addVar(lb=0.0, name="abs_diff_delta")
            # m.addConstr(diff_delta == delta1[S1_i] - delta0['delta0'], 'tmpConstr')
            # m.addGenConstrAbs(abs_diff_delta, diff_delta, "tmpAbsConstr")

            m.addConstr(IR_effect(t0, t0, t2) * d['d']
                        + sum(IR_effect(t0, t1, t2) * (theta1_ask[r] - theta1_bid[r]) * max(S1_i - K1[r], 0)
                            for r in df_t0.index)
                        + sum(IR_effect(t0, t2, t2) * (theta2_ask[r] - theta2_bid[r]) * max(S2_j - K2[r], 0)
                            for r in df_t0.dropna().index)
                        # in our cases the number of options at maturity T2 is less or equal to that at maturity T1
                        # thus apply dropna() on T2 exclusively to shorten the running time effectively
                        + delta0['delta0'] * (S1_i - S0 * IR_effect(t0, t0, t1))
                        + delta1[S1_i] * (S2_j - S1_i * IR_effect(t0, t1, t2))
                        # - 0.0005*S1_i*abs_diff_delta
                        >= payoff(S1_i, S2_j),
                        'C_{}{}'.format(S1_i, S2_j))
            m.update()

    m.optimize()

    if m.status == GRB.OPTIMAL:
        var = {v.varName: v.x for v in m.getVars()}

        delta1_values = np.array([var['delta1[{}]'.format(s1)] for s1 in S1])
        delta1_interpolator = CubicSpline(S1, delta1_values, bc_type='natural')

        parameters = {
            'd': var['d[d]'],
            'theta1_ask': {r: var['theta1_ask[{}]'.format(r)] for r in df_t0.index},
            'theta1_bid': {r: var['theta1_bid[{}]'.format(r)] for r in df_t0.index},
            'theta2_ask': {r: var['theta2_ask[{}]'.format(r)] for r in df_t0.dropna().index},
            'theta2_bid': {r: var['theta2_bid[{}]'.format(r)] for r in df_t0.dropna().index},
            'delta0': var['delta0[delta0]'],
            'delta1': delta1_interpolator
        }
        return parameters

    else:
        print("Optimization was not successful.")
        return False



# cutting plane algorithm
def cuttingPlane(df_t0, t0, t1, t2, N, n, realS1, realS2, TOL):
    # N: large grid
    # n: initial sub-grid
    # realS1, realS2: used to determine the range of grid
    # TOL: tolerance
    start_time = time.time() 

    K1 = df_t0['K1']
    K2 = df_t0['K2']    

    
    I_ref = 50
    d_S1, u_S1 = min(K1)-I_ref, max(K1)+I_ref
    d_S2, u_S2 = min(K2)-I_ref, max(K2)+I_ref
    while realS1 > u_S1:
        u_S1 += I_ref
    while realS1 < d_S1:
        d_S1 -= I_ref
    while realS2 > u_S2:
        u_S2 += I_ref
    while realS2 < d_S2:
        d_S2 -= I_ref
    
    S1_N = np.round(np.linspace(d_S1, u_S1, N),6)
    S2_N = np.round(np.linspace(d_S2, u_S2, N),6)
    
    index = [int(i*(N/n)) for i in range(n)]
    S1 = S1_N[index]
    S2 = S2_N[index]
    
    # 确保 S1 和 S2 没有重复的值
    S1 = np.unique(S1)
    S2 = np.unique(S2)

    delta = - np.infty
    k = 0
    while delta <= - TOL:
        
        # 检查是否超出最大运行时间
        elapsed_time = time.time() - start_time
        if elapsed_time > 600:
            print("Exceeded maximum runtime of 600 seconds. Stopping.")
            return None, k
        
        
        p = params(df_t0, t0, t1, t2, S1, S2)
        if p == False:
            print("Model is infeasible.")
            break
        else:
            _, test = hedgeGapOnLargeGrid(df_t0, t0, t1, t2, S1_N, S2_N, p)
            if np.min(test) == delta:
                print("Could not satisfy the tolerance level.")
                break
            else:
                delta = np.min(test)
                k += 1
                index = np.where(test < delta+1/k)
                S1 = np.unique(np.sort(np.append(S1, S1_N[index[0]])))
                S2 = np.unique(np.sort(np.append(S2, S2_N[index[1]])))
    
    print(f"Done after {k} iterations.")
        
    parameters = p
    return parameters, k

# calculate the payoff of a hedging strategy
def hedgingStrategy(parameters, df_t0, t0, t1, t2, s1, s2):
    K1 = df_t0['K1']
    K2 = df_t0['K2']
    S0 = df_t0['Adj_S0'].unique()

    h = (IR_effect(t0, t0, t2) * parameters['d']
         + sum(IR_effect(t0, t1, t2) * (parameters['theta1_ask'][r] - parameters['theta1_bid'][r]) * max(s1 - K1[r], 0)
               for r in df_t0.index)
         + sum(IR_effect(t0, t2, t2) * (parameters['theta2_ask'][r] - parameters['theta2_bid'][r]) * max(s2 - K2[r], 0)
               for r in df_t0.dropna().index)
         + parameters['delta0'] * (s1 - S0 * IR_effect(t0, t0, t1))
         + parameters['delta1'](s1) * (s2 - s1 * IR_effect(t0, t1, t2))
         # - 0.0005 * s1 * abs(parameters['delta1'](s1) - parameters['delta0'])
         )
    return h


# find best Paras

def find_best_parameters(df_t0, realS1, realS2, t0, t1, t2, TOL_values, n_values, N_values):
        
    S0 = df_t0['Adj_S0'].unique()[0]
    
    results = []
    

    for TOL in TOL_values:
        for n in n_values:
            for N in N_values:
                print("-------------------------------")
                print(f"Testing TOL={TOL}*S0, n={n}, N={N}")
                TOL_scale = TOL*S0
                
                # start time
                start_time = time.time()
                
                parameters, k = cuttingPlane(df_t0, t0, t1, t2, N, n, realS1, realS2, TOL_scale)
                
                if parameters is None:
                    continue
                
                if parameters:  
                    hedging_val = hedgingStrategy(parameters, df_t0, t0, t1, t2, realS1, realS2)
                    payoff_val = payoff(realS1, realS2)
                    gap = (hedging_val - payoff_val) / S0
                    gap_percentage = gap* 100 
                    

                    end_time = time.time()
                    runtime = end_time - start_time
                    
                    
                    results.append({
                        "TOL": TOL,
                        "n": n,
                        "N": N,
                        "gap_percentage": gap_percentage[0], 
                        "runtime": runtime,
                        "iterations": k
                    })
                    

    df_results = pd.DataFrame(results)
        
    return df_results



data_path = f'options_call_askbid'
data_files = os.listdir(data_path)
data_files = [file for file in data_files if file[-4:] == '.csv']
# AMZN files:
tickerList = ['AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'PG', 'TSLA', 'V', 'WMT']
ticker = tickerList[0]
data_files = [file for file in data_files if file.split('_')[0] == ticker]
pattern = r'^([A-Z]+)_(\d{8})_(\d{8})\.csv$'

for data_set in ticker:
   
    TOL_values = [0.05]
    n_values = list(range(10, 70, 5))    
    N_values = [500, 600, 700, 800, 900, 1000]
    
    
    if not os.path.exists('best_para_output'):
        os.makedirs('best_para_output')
        
    find_best_para = pd.DataFrame()
    
    for filename in sorted(data_files):
        print("++++++++++++++++++++++++++++++++++++")
        print(f"Processing {filename[:-4]}:")
        
        # 提取 ticker 和日期信息
        match = re.match(pattern, filename)
        ticker = match.group(1)
        date1 = match.group(2)
        date2 = match.group(3)
        
        # 生成 t1 和 t2 日期
        t1 = f'{date1[:4]}-{date1[4:6]}-{date1[6:]}'
        t2 = f'{date2[:4]}-{date2[4:6]}-{date2[6:]}'
        
        # 读取文件
        path = data_path + '/' + filename
        df = pd.read_csv(path, index_col=0)
        df['t0'] = pd.to_datetime(df['t0'])
        
        # 生成 t0 日期列表
        t0List = list(df['t0'].unique())
        
        # 从 stocks 中提取 realS1 和 realS2
        realS1 = stocks.loc[stocks['Date'] == pd.Timestamp(t1), ticker].squeeze()
        realS2 = stocks.loc[stocks['Date'] == pd.Timestamp(t2), ticker].squeeze()
        
        for t0 in t0List:
            print("===============================")
            print(f"t0 = {t0}:")
            
            # 过滤 df 只保留 t0 对应的数据
            df_t0 = df.loc[df['t0'] == pd.Timestamp(t0)].copy()
                    
            # 选择最优参数
            df_results = find_best_parameters(df_t0, realS1, realS2, t0, t1, t2, TOL_values, n_values, N_values)
            df_results['ticker'] = ticker
            df_results['t0'] = t0
            df_results['t1'] = t1
            df_results['t2'] = t2
            df_results = df_results[['ticker', 't0', 't1', 't2', 'TOL', 'n', 'N', 'runtime', 'iterations', 'gap_percentage']]
            
            # 将 df_results 追加到 find_best_para DataFrame
            find_best_para = pd.concat([find_best_para, df_results], ignore_index=True)
            
            t0 = pd.to_datetime(t0)
            output_filename = f'best_para_output/find_best_parameters_{ticker}_{t0.strftime("%Y%m%d")}_{date1}_{date2}.csv'
            df_results.to_csv(output_filename, index=False)
            print(f"Saved: {output_filename}")
