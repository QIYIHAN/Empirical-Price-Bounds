import math
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import gurobipy as gp
from gurobipy import GRB
import os
from datetime import date
import warnings
warnings.filterwarnings('ignore')


data_path = f'data/options_call_askbid'
data_files = os.listdir(data_path)
data_files = [file for file in data_files if file[-4:] == '.csv']
# WMT files:
tickerList = ['AMZN', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'PG', 'TSLA', 'V', 'WMT']
ticker = tickerList[8]
data_files = [file for file in data_files if file.split('_')[0] == ticker]

stocks = pd.read_csv('data/adjusted_stocks.csv')

IRParams = pd.read_csv('data/interest_rates_parameters.csv', parse_dates=['Date'], dayfirst=True)
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


def payoff(S1, S2):
    return np.maximum(S2 - S1, 0)


# compounding factor
def IR_effect(init_date, start_date, end_date):
    if start_date == end_date:
        return 1.0
    else:
        tau = np.busday_count(start_date, end_date, holidays=nasdaq_holidays) / 252
        if tau == 0.0:
            return 1.0
        else:
            beta0 = IRParams.loc[IRParams.Date == init_date, 'BETA0'].item()
            beta1 = IRParams.loc[IRParams.Date == init_date, 'BETA1'].item()
            beta2 = IRParams.loc[IRParams.Date == init_date, 'BETA2'].item()
            tau1 = IRParams.loc[IRParams.Date == init_date, 'TAU1'].item()

            r = beta0 + beta1*(1-math.exp(-tau/tau1))/(tau/tau1) + beta2*((1-math.exp(-tau/tau1))/(tau/tau1)-math.exp(-tau/tau1))
            return math.exp(r/100*tau)


# computation of hedge gaps on the large grid in cutting plane algorithm
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

    m = gp.Model("Price Bounds - SubHedge")
    m.setParam('OutputFlag', 0)

    # cash position
    d = m.addVars(['d'], lb=-float('inf'), name="d")
    # only short options
    theta1_ask = m.addVars(df_t0.index, ub=0.0, name="theta1_ask")
    theta1_bid = m.addVars(df_t0.index, lb=0.0, name="theta1_bid")
    theta2_ask = m.addVars(df_t0.index, ub=0.0, name="theta2_ask")
    theta2_bid = m.addVars(df_t0.index, lb=0.0, name="theta2_bid")
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
    m.ModelSense = GRB.MAXIMIZE

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
                        <= payoff(S1_i, S2_j),
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
        return m.ObjVal, parameters

    else:
        print("Optimization was not successful.")
        return False, False


def cuttingPlane(df_t0, t0, t1, t2, N, n, realS1, realS2, TOL):
    # N: large grid
    # n: initial sub-grid
    # realS1, realS2: used to determine the range of grid
    # TOL: tolerance

    K1 = df_t0['K1']
    K2 = df_t0['K2']

    I_ref = 50
    d_S1, u_S1 = max(min(K1) - I_ref, 1e-6), max(K1) + I_ref
    d_S2, u_S2 = max(min(K2) - I_ref, 1e-6), max(K2) + I_ref
    while realS1 > u_S1:
        u_S1 += I_ref
    while realS1 < d_S1:
        d_S1 -= I_ref
    while realS2 > u_S2:
        u_S2 += I_ref
    while realS2 < d_S2:
        d_S2 -= I_ref

    S1_N = np.round(np.linspace(d_S1, u_S1, N), 6)
    S2_N = np.round(np.linspace(d_S2, u_S2, N), 6)

    index = [int(i * (N / n)) for i in range(n)]
    S1 = S1_N[index]
    S2 = S2_N[index]

    delta = np.infty
    k = 0
    p = None
    while delta >= TOL:
        obj, p = params(df_t0, t0, t1, t2, S1, S2, init=p)
        if p == False:
            print("Model is infeasible.")
            break
        else:
            _, test = hedgeGapOnLargeGrid(df_t0, t0, t1, t2, S1_N, S2_N, p)
            if np.max(test) == delta:
                print("Could not satisfy the tolerance level.")
                break
            else:
                delta = np.max(test)
                k += 1
                index = np.where(test > delta - 1 / k / 100)
                S1 = np.unique(np.sort(np.append(S1, S1_N[index[0]])))
                S2 = np.unique(np.sort(np.append(S2, S2_N[index[1]])))
    print(f"Done after {k} iterations.")

    priceBound = obj
    return priceBound


df_priceBound = pd.DataFrame()
N, n, tol = 100, 20, 0.05
for filename in sorted(data_files):
    print(f"Processing {filename[:-4]}:")
    date1 = filename[:-4].split('_')[1]
    date2 = filename[:-4].split('_')[2]
    t1 = f'{date1[:4]}-{date1[4:6]}-{date1[6:]}'
    t2 = f'{date2[:4]}-{date2[4:6]}-{date2[6:]}'

    path = data_path + '/' + filename
    df = pd.read_csv(path, index_col=0)
    t0List = list(df.t0.unique())

    realS1 = stocks.loc[stocks.Date == t1, ticker].item()
    realS2 = stocks.loc[stocks.Date == t2, ticker].item()

    priceBoundList = []
    for t0 in t0List:
        print(f"t0 = {t0}:")
        df_t0 = df.loc[df.t0 == t0].copy()

        S0 = df_t0['Adj_S0'].unique()
        pb = cuttingPlane(df_t0, t0, t1, t2, N, n, realS1, realS2, tol*S0)
        if pb == False:
            priceBound = np.nan
        else:
            priceBound = pb

        priceBoundList.append(priceBound)

    distance_to_t1 = np.busday_count(pd.to_datetime(t0List).values.astype('datetime64[D]'),
                                     pd.to_datetime([t1]).values.astype('datetime64[D]'),
                                     holidays=nasdaq_holidays)
    tmp = pd.DataFrame(np.array(priceBoundList), columns=[filename[:-4]], index=distance_to_t1)
    df_priceBound = pd.concat([df_priceBound, tmp.T])
    df_priceBound = df_priceBound.sort_index(axis=1).copy()
    # output the data during the processing (keep the data of the newest date)
    df_priceBound.to_csv(f'tmp_results/priceBoundData_{ticker}_{date.today().strftime("%d%m%y")}.csv')

# overwrite with the full data
df_priceBound.to_csv(f'tmp_results/priceBoundData_{ticker}_{date.today().strftime("%d%m%y")}.csv')

