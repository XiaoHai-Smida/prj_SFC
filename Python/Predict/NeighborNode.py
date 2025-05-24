import numpy as np
def select_start_node(distance_matrix):
    """选择平均距离最小的节点作为起始点"""
    n = len(distance_matrix)
    min_avg = float('inf')
    start_node = 0
    for i in range(n):
        avg = np.mean([distance_matrix[i][j] for j in range(n) if j != i])
        if avg < min_avg:
            min_avg = avg
            start_node = i
    return start_node

def nearest_neighbor_path(distance_matrix, start_node):
    """最近邻法生成初始路径"""
    n = len(distance_matrix)
    path = [start_node]
    visited = set([start_node])
    current = start_node
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if j not in visited and distance_matrix[current][j] < min_dist:
                min_dist = distance_matrix[current][j]
                next_node = j
        if next_node == -1:
            break  # 处理无法访问的情况
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path

def calculate_total_distance(distance_matrix, path):
    """计算路径总距离"""
    return sum(distance_matrix[path[i]][path[i+1]] for i in range(len(path)-1))

def two_opt_swap(path, i, k):
    """交换路径中的节点i到k"""
    return path[:i] + path[i:k+1][::-1] + path[k+1:]

def two_opt_optimize(distance_matrix, path, max_iterations=100):
    """2-opt算法优化路径"""
    best_path = path.copy()
    best_distance = calculate_total_distance(distance_matrix, best_path)
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, len(best_path) - 2):
            for k in range(i + 1, len(best_path) - 1):
                new_path = two_opt_swap(best_path, i, k)
                new_distance = calculate_total_distance(distance_matrix, new_path)
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True
                    break  # 找到改进后重新开始
            if improved:
                break
        iteration += 1
    return best_path

# 示例距离矩阵
distance_matrix = np.loadtxt('dis.csv', delimiter=',')

# 选择起始节点并生成路径
start_node = select_start_node(distance_matrix)
initial_path = nearest_neighbor_path(distance_matrix, start_node)
optimized_path = two_opt_optimize(distance_matrix, initial_path)
np.savetxt('optimized_path.csv', optimized_path, delimiter=',')

print("初始路径:", initial_path)
print("初始总距离:", calculate_total_distance(distance_matrix, initial_path))
print("优化后路径:", optimized_path)
print("优化后总距离:", calculate_total_distance(distance_matrix, optimized_path))
