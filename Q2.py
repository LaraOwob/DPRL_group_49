import random



def createDemand(t, demand_prob):
    print(f"t: {t}, demand_prob: {t/demand_prob}")
    if random.random() < t/demand_prob:
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
        return 1
    else:
        return 0


def simulateInventory(initial_stock, demand_prob, days, price_per_unit, holding_cost_per_unit,threshold):
    demand_list = [createDemand(t+1, demand_prob) for t in range(days)]
    stock_level = initial_stock
    total_revenue = 0
    total_holding_cost = 0
    
    for day in range(days):
        demand = demand_list[day]
        if stock_level < threshold:
            stock_level += orderStock(stock_level)
        if stock_level >= demand:
            stock_level -= demand
            
        total_revenue += demand * price_per_unit - holding_cost_per_unit * max(0,stock_level)
        print(f"Day {day+1}: Demand={demand}, Stock Level={stock_level}, Total Revenue={total_revenue:.2f}")

    
    net_profit = total_revenue - total_holding_cost
    return net_profit


def main():
    initial_stock = 5
    demand_prob = 150
    days = 150
    price_per_unit = 1
    holding_cost_per_unit = 0.1
    threshold = 2
    net_profit = simulateInventory(initial_stock, demand_prob, days, price_per_unit, holding_cost_per_unit,threshold)
    print(f"Net Profit over {days} days: ${net_profit:.2f}")
    
    
if __name__ == "__main__":
    main()