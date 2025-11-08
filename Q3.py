import random

import matplotlib.pyplot as plt

def createDemand(t, demand_prob):
    if random.random() < t*demand_prob:
        return 1
    else:
        return 0
    
    
def calculateStockLevel(initial_stock, demand_list,threshold):
    stock_level = initial_stock
    stock_levels = []
    
    for demand in demand_list:
        if stock_level < threshold:
            stocklevel +=orderStock(current_stock=stock_level)
        stock_level -= demand
        stock_levels.append(stock_level)
    
    return stock_levels

def orderStock(current_stock):
    if random.random() < 0.5:
        #print("Ordering Stock")
        return 1
    else:
        #print("Not Ordering Stock")
        return 0


def simulateInventory(initial_stock, demand_prob, days, profit_per_unit, holding_cost_per_unit,threshold):
    demand_list = [createDemand(t+1, demand_prob) for t in range(days)]
    stock_level = initial_stock
    total_revenue = 0
    total_holding_cost = 0
    
    for day in range(days):
        demand = demand_list[day]
        if stock_level <= threshold:
            stock_level += orderStock(stock_level)
        if stock_level >= demand:
            #print("Stock sufficient to meet demand",stock_level,demand)
            stock_level -= demand
            total_revenue += demand * profit_per_unit - holding_cost_per_unit * max(0,stock_level)
        else:
            #print("Stock insufficient to meet demand",stock_level,demand)
            total_revenue += - holding_cost_per_unit * max(0,stock_level)
        #print(f"Day {day+1}: Demand={demand}, Stock Level={stock_level}, Total Revenue={total_revenue:.2f}\n\n")

    
    net_profit = total_revenue - total_holding_cost
    return net_profit

def Simulation(runs):
    total_profit = 0
    profit_list = []
    for _ in range(runs):
        profit = simulateInventory(initial_stock=5, demand_prob=1/150, days=150, profit_per_unit=1, holding_cost_per_unit=0.1,threshold=1)
        total_profit += profit
        print(f"Run {_+1}: Profit=${profit:.2f}")
        profit_list.append(profit)
    average_profit = total_profit / runs
    print(f"Average Profit over {runs} runs: ${average_profit:.2f}")
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
    holding_cost_per_unit = 0.1
    threshold = 1
    no_runs = 1000
    net_profit = simulateInventory(initial_stock, demand_prob, days, price_per_unit, holding_cost_per_unit,threshold)
    print(f"Net Profit over {days} days: ${net_profit:.2f}")
    data_profit = Simulation(runs=no_runs)
    makeHistogram(data_profit)
if __name__ == "__main__":
    main()