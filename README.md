# Empirical-Option-Pricing

## Data Processing
### a) Fixed $t_0$ and find $T_1$ and $T_2$
Look at data from date $t_0$, choose one expiry date as $T_1$ and $T_2 = T_1 + 3$ months:  
We use AMAZON data first, choose $t_0$ as 2018-01-02, $T_1$ as 2018-01-19, and $T_2$ as 2018-04-20. Then we proceed $t_0$ from 2018-01-02 to 2018-01-19.

### b) Choose 20 options with largest volume on date $t_0$ for expiry date $T_1$ and $T_2$

### c) Clean data with arbitrage repair
Compute interest rate $r$: use put-call parity $C_{t}-P_{t}=S_{t}-Ke^{-r\cdot(T-t)}$ to compute.  
Next, we use computed $r$ to compute the forward price ($F$) in order to use arbitrage repair to clean the data.

## Stock Price Simulation  
Temporarily we use uniform distribution to simulate stock prices. For further study, we can use $\frac{\partial^{2}C}{\partial K^{2}}=e^{-rT}\phi(T,K)$ to compute the marginal distribution.

## Pricing Boundary
The objective function we need to minimise is:

$$
\begin{align*}
    & d + \sum_{i=0}^{N_1} \lambda_{i,1} \Pi_{i,1} + \sum_{i=0}^{N_2} \lambda_{i,2} \Pi_{i,2} + \\
    & \Gamma\cdot\left[\left(\Phi(S_1,S_2)- \left(d + \sum_{i=0}^{N_1} \lambda_{i,1}(S_1-K_{i,1})^+ +\sum_{i=0}^{N_2} \lambda_{i,2} (S_2-K_{i,2})^+ +\Delta_1(S_1)(S_2-S_1)\right)\right)^+\right]^2
\end{align*}
$$

$\Pi_{i,1}$ / $\Pi_{i,2}$ is the $i$-th option among those with strike $K_{i,1}$ / $K_{i,2}$ and expiry date $T_1$ / $T_2$ who have $N_1$ / $N_2$ largest trading volume. Temporarily set $N_1$ and $N_2$ the same value 20. $\Phi$ is the payoff of the derivative we want to hedge. And $d, \lambda_{i,1}, \lambda_{i,2},$ and $\Delta_1(S_1)$ are the parameters we want to optimize.

First compute the hedge price part 

$$
d + \sum_{i=0}^{N_1} \lambda_{i,1} \Pi_{i,1} + \sum_{i=0}^{N_2} \lambda_{i,2} \Pi_{i,2}
$$

Then the hedge term part 

$$ 
d+\sum_{i=0}^{N_1}\lambda_{i,1}(S_1-K_{i,1})^++\sum_{i=0}^{N_2}\lambda_{i,2}(S_2-K_{i,2})^++\Delta_1(S_1)(S_2-S_1)
$$

Then combine them into a function (i.e., the objective function), which has two outputs: the value of the objective function and the hedge price.

For now, we utilize `tensorflow.keras.optimizers.Adam` to minimize the objective function, and output the hedge price after optimization.
