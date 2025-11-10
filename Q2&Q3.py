import numpy as np

import matplotlib.pyplot as plt

def makebestpolicy(V, alpha, days,order_p, p, h, demand_prob, x_max):
    for t in range(days-2,-1,-1):
        for x in range(x_max+1):
            Q = []
            if x < x_max :
                for a in [0,1]:
                    
                    stock_s = x + a
                    
                    sales_s_1 = min(1, stock_s)
                    
                    new_stock_s_1 = stock_s - sales_s_1
                    profit_s_1 = p * sales_s_1 - h * new_stock_s_1
                    
                    sales_s_0 = min(0, stock_s)
                    new_stock_s_0 = stock_s - sales_s_0
                    profit_s_0 = p * sales_s_0 - h * new_stock_s_0
                    
                
                    if a ==0:
                        expected_profit = (
                        ((demand_prob*t)*profit_s_1 + V[new_stock_s_1, t + 1])+((1-demand_prob*t)*profit_s_0 + V[new_stock_s_0, t + 1])) 
                    else:
                        #failure case
                        stock_f = x
                        sales_f_1 = min(1, stock_f)
                        new_stock_f_1 = stock_f - sales_f_1
                        profit_fail_1 = p * sales_f_1 - h * new_stock_f_1
                        
                        stock_f = x
                        sales_f_0 = min(0, stock_f)
                        new_stock_f_0 = stock_f- sales_f_0
                        profit_fail_0 = p * sales_f_0 - h * new_stock_f_0
                        
                        expected_profit = (
                        order_p * (((demand_prob*t)*profit_s_1 + V[new_stock_s_1, t + 1])
                                +((1-demand_prob*t)*profit_s_0 + V[new_stock_s_0, t + 1]))  +
                        (1 - order_p) * ((demand_prob*t*profit_fail_1 + V[new_stock_f_1, t + 1]) + (1-demand_prob*t)*(profit_fail_0 + V[new_stock_f_0, t + 1])))
                    Q.append(expected_profit)                    
                        
                a_min = Q.index(min(Q))
                V[x, t] = Q[a_min]
                alpha[x, t] = a_min
                for a in [0,1]:
                    if Q[a] > V[x,t]:
                        alpha[x,t] = a
                        V[x,t] = Q[a]
    print("Optimal Value Function V:\n", V)
    print("Optimal Policy alpha:\n", alpha)
    return V, alpha

                    
                    
def create_demand(t, demand_prob):
    D = []
    for t in range(t):
        if np.random.rand() < t*demand_prob:
            D.append(1)
        else:
            D.append(0)
    return D


def makePlot(V, alpha, days, x_max):
    # Plot value function
    # Plot optimal policy
    plt.figure(figsize=(12, 6))
    plt.imshow(alpha, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Optimal Order Quantity')
    plt.xlabel('Time (days)')
    plt.ylabel('Inventory Level')
    plt.title('Optimal Policy α(x,t)')
    plt.show()
    plt.savefig('optimal_policy.png')




def Simulation(runs,V, alpha,initial_stock, days,order_prob ,price_per_unit, holding_cost, x_max, demand_probability):
    total_profit = 0
    profit_list = []
    for _ in range(runs):
        V,alpha = makebestpolicy(V, alpha, days,order_prob ,price_per_unit, holding_cost, demand_probability, x_max)
        expected_profit = V[initial_stock, 0]
        total_profit += expected_profit
        print(f"Run {_+1}: Expected profit=${expected_profit:.2f}")
        profit_list.append(expected_profit)
    average_profit = total_profit / runs
    print(f"Average expected Profit over {runs} runs: ${average_profit:.2f}")
    return profit_list

def makeHistogram(data):
    plt.hist(data, bins=20, edgecolor='black')
    plt.title('Histogram of Net Profits')
    plt.xlabel('Net Profit')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('histogram_net_profits.png')
    
    
    
def main():
    initial_stock = 5
    demand_prob = 1/150
    days = 150
    price_per_unit = 1
    holding_cost = 0.1
    order_prob = 0.5
    x_max = days+initial_stock
    V = np.zeros((x_max+1,days))
    V[:, -1] = 0
    alpha = np.zeros((x_max+1,days))
    #Demand = create_demand(days, demand_prob)
    V,alpha = makebestpolicy(V, alpha, days,order_prob ,price_per_unit, holding_cost, demand_prob, x_max)
    #makePlot(V, alpha, days, x_max)
    # Maximal expected reward starting from initial stock
    print("Maximal Expected Reward:", V[initial_stock, 0])
    no_runs = 100
    #data_profit = Simulation(runs=no_runs,V=V,alpha=alpha,initial_stock=initial_stock, days=days,order_prob =order_prob ,price_per_unit=price_per_unit, holding_cost=holding_cost,x_max=x_max, demand_probability=demand_prob)
    #makeHistogram(data_profit)
    
    

if __name__ == "__main__":
    main()
    