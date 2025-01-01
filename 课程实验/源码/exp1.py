import mindspore as ms
from mindspore import Tensor
import numpy as np
import heapq

# 罗马尼亚问题的图数据
romania_map = {
    "Arad": [("Zerind", 75), ("Timisoara", 118), ("Sibiu", 140)],
    "Zerind": [("Arad", 75), ("Oradea", 71)],
    "Timisoara": [("Arad", 118), ("Lugoj", 111)],
    "Sibiu": [("Arad", 140), ("Fagaras", 99), ("Rimnicu Vilcea", 80)],
    "Oradea": [("Zerind", 71), ("Sibiu", 151)],
    "Lugoj": [("Timisoara", 111), ("Mehadia", 70)],
    "Mehadia": [("Lugoj", 70), ("Drobeta", 75)],
    "Drobeta": [("Mehadia", 75), ("Craiova", 120)],
    "Craiova": [("Drobeta", 120), ("Rimnicu Vilcea", 146), ("Pitesti", 138)],
    "Rimnicu Vilcea": [("Sibiu", 80), ("Craiova", 146), ("Pitesti", 97)],
    "Fagaras": [("Sibiu", 99), ("Bucharest", 211)],
    "Pitesti": [("Rimnicu Vilcea", 97), ("Craiova", 138), ("Bucharest", 101)],
    "Bucharest": [("Fagaras", 211), ("Pitesti", 101), ("Giurgiu", 90)],
    "Giurgiu": [("Bucharest", 90)],
}


# 启发函数 h(n)（估计值）
heuristic = {
    "Arad": 366, "Zerind": 374, "Timisoara": 329, "Sibiu": 253,
    "Oradea": 380, "Lugoj": 244, "Mehadia": 241, "Drobeta": 242,
    "Craiova": 160, "Rimnicu Vilcea": 193, "Fagaras": 176,
    "Pitesti": 100, "Bucharest": 0, "Giurgiu": 77,
}


def a_star_search(graph, start, goal, heuristic):
    # 优先队列存储 (f(n), g(n), 节点, 路径)
    open_list = []
    heapq.heappush(open_list, (heuristic[start], 0, start, [start]))

    visited = set()  # 记录已访问节点

    while open_list:
        # 取出优先级最高的节点
        f, g, current, path = heapq.heappop(open_list)

        if current in visited:
            continue
        visited.add(current)

        # 如果达到目标节点，返回路径和代价
        if current == goal:
            return path, g

        # 扩展当前节点
        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float("inf")  # 无法到达目标

# 求解从 Arad 到 Bucharest 的最短路径
start_city = "Arad"
goal_city = "Bucharest"

path, cost = a_star_search(romania_map, start_city, goal_city, heuristic)
print(f"路径: {path}")
print(f"总代价: {cost}")