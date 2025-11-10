import numpy as np

import matplotlib.pyplot as plt

    

def solve_dp(V,policy,T, x_max, p_sell, h, order_p):
    for t in range(T - 1, -1, -1):
        p_d = (t + 1) / T  
        for x in range(x_max + 1):
            Q = []
            for a in [0, 1]: 
                if a == 0:
                    
                    stock_before = x
                    #if demand
                    sales_if_d1 = min(1, stock_before)
                    next_stock_if_d1 = stock_before - sales_if_d1
                    reward_if_d1 = p_sell * sales_if_d1 - h * next_stock_if_d1

                    #if no demand
                    sales_if_d0 = 0
                    next_stock_if_d0 = stock_before
                    reward_if_d0 = - h * next_stock_if_d0

                    expected_revenue = (p_d * (reward_if_d1 + V[next_stock_if_d1, t + 1]) +
                                (1 - p_d) * (reward_if_d0 + V[next_stock_if_d0, t + 1]))
                else:
                    #if order arrives
                    # if demand
                    stock_s = min(x + 1, x_max)
                    sales_s_1 = min(1, stock_s)
                    next_s_1 = stock_s - sales_s_1
                    reward_s_1 = p_sell * sales_s_1 - h * next_s_1
                    # if no demand
                    sales_s_0 = 0
                    next_s_0 = stock_s
                    reward_s_0 = - h * next_s_0

                    val_success = (p_d * (reward_s_1 + V[next_s_1, t + 1]) +
                                   (1 - p_d) * (reward_s_0 + V[next_s_0, t + 1]))

                    #if order fails
                    # if demand
                    stock_f = x
                    sales_f_1 = min(1, stock_f)
                    next_f_1 = stock_f - sales_f_1
                    reward_f_1 = p_sell * sales_f_1 - h * next_f_1
                    # if no demand
                    sales_f_0 = 0
                    next_f_0 = stock_f
                    reward_f_0 = p_sell * sales_f_0 - h * next_f_0

                    val_fail = (p_d * (reward_f_1 + V[next_f_1, t + 1]) +
                                (1 - p_d) * (reward_f_0 + V[next_f_0, t + 1]))

                    expected_revenue = order_p * val_success + (1 - order_p) * val_fail

                Q.append(expected_revenue)

            # choose best action (max expected reward)
            best_action = int(np.argmax(Q))
            V[x, t] = Q[best_action]
            policy[x, t] = best_action

    return V, policy


def makePlot2(V, alpha, days, x_max):

    plt.figure(figsize=(12, 6))
    plt.imshow(alpha, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Optimal Order Quantity')
    plt.xlabel('Time (days)')
    plt.ylabel('Inventory Level')
    plt.title('Optimal Policy α(x,t)')
    plt.show()
    plt.savefig('optimal_policy.png')



def simulate_process(runs, policy, initial_stock, days, order_prob, price_per_unit, holding_cost, x_max):
    profits_list = []

    for _ in range(runs):
        stock = initial_stock
        total_reward = 0.0

        for t in range(days):
            p_d = (t + 1) / days
            demand = np.random.rand() < p_d
            action = policy[stock, t]

            if action == 1 and np.random.rand() < order_prob:
                stock = min(stock + 1, x_max)

            if demand and stock > 0:
                stock -= 1
                total_reward += price_per_unit*demand  

            total_reward -= holding_cost * stock

        profits_list.append(total_reward)

    avg_profit = sum(profits_list) / runs
    print(f"Average realized profit over {runs} runs: {avg_profit:.2f}")

    return profits_list



def makeHistogram3(data):
    plt.hist(data, bins=20, edgecolor='black')
    plt.title('Histogram of Net Profits')
    plt.xlabel('Net Profit')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('histogram_net_profits.png')
    
    
    
def main():
    initial_stock = 5
    days = 150
    price_per_unit = 1
    holding_cost = 0.1
    order_prob = 0.5
    x_max = days + initial_stock
    # Question 2
    V = np.zeros((x_max + 1, days + 1))
    policy = np.zeros((x_max + 1, days), dtype=int) 
    V,optimal_policy  = solve_dp(V,policy, days, x_max, price_per_unit, holding_cost, order_prob)
    print("Maximal Expected Reward:", V[initial_stock, 0])
    makePlot2(V, optimal_policy, days, x_max)
    
    #Question 3
    no_runs = 1000
    profits = simulate_process(no_runs, optimal_policy, initial_stock, days, order_prob, price_per_unit, holding_cost, x_max)
    makeHistogram3(profits)
    
    

if __name__ == "__main__":
    main()
    