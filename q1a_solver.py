#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

# ================================================================
# Q1a: A* Search — Single Food Dot
# Q1a: A* 搜索 —— 单一食物点路径规划
#
# 目标 / Goal:
#   使用 A* 算法找到 Pac-Man 到达唯一食物点的最短路径。
#   Use A* to find the shortest path from Pac-Man to the single food dot.
#
# 状态 / State:
#   (x, y) —— Pac-Man 当前位置坐标。
#   (x, y) —— Current position of Pac-Man.
#
# 启发函数 / Heuristic:
#   曼哈顿距离（可接受且一致），保证 A* 找到最优解。
#   Manhattan distance (admissible & consistent), guarantees optimal solution.
# ================================================================


class AStarData:
    # 存储 A* 运行时所需的所有数据结构
    # Holds all data structures needed during A* execution
    def __init__(self):
        self.frontier = util.PriorityQueue()  # 优先队列：f = g + h 升序 / Priority queue ordered by f = g + h
        self.g_scores = {}                    # g(x)：起点到x的最小成本（初始为空）/ g(x): min cost from start to x
        self.came_from = {}                   # 路径回溯：x → (父节点, 动作) / Path tracking: x → (parent, action)
        self.closed = set()                   # 已处理节点（出队后标记）/ Closed set: nodes already expanded
        self.goal = None                      # 目标食物位置（唯一）/ Target food position (single dot)
        self.start = None                     # Pac-Man初始位置 / Pac-Man's starting position

def astar_initialise(problem: q1a_problem):
    """
    初始化 A* 数据结构，设置起始节点并将其压入优先队列。
    Initialise A* data structures, set the start node and push it onto the priority queue.
    """
    astarData = AStarData()
    # 1. 起始位置：严格调用problem的官方方法，无自定义
    # 1. Start position: obtained strictly from the official problem interface
    astarData.start = problem.getStartState()
    # 提取目标点 / Extract the single food target position
    astarData.goal = problem.startingGameState.getFood().asList()[0]

    # 压入起点 / Push start node with f = 0 + h(start)
    astarData.g_scores[astarData.start] = 0
    h_score = astar_heuristic(astarData.start, astarData.goal)
    astarData.frontier.push(astarData.start, h_score) # f = g + h = 0 + h

    return astarData

def astar_loop_body(problem, astarData):
    """
    A* 主循环体，每次调用处理一个节点扩展。
    A* main loop body — processes one node expansion per call.

    返回 / Returns:
        (terminate: bool, result: list) —— 是否终止 + 动作序列
        (terminate: bool, result: list) —— whether to stop + action sequence
    """
    # 边界：队列空了都没找到，直接宣布无解
    # Boundary: if frontier is empty before reaching goal, no solution exists
    if astarData.frontier.isEmpty():
        return True, []

    current = astarData.frontier.pop()

    # 目标检测必须在Pop出列后！这是保证A*最优性的铁律
    # Goal check MUST happen after pop — this is the rule that guarantees A* optimality
    if problem.isGoalState(current):
        path = []
        curr = current
        # 顺藤摸瓜回溯路径 / Backtrack from goal to start via came_from
        while curr in astarData.came_from:
            parent, action = astarData.came_from[curr]
            path.append(action)
            curr = parent
        path.reverse()
        return True, path

    # 如果优先队列里有该节点的陈旧数据（代价更高），直接过滤
    # Skip stale entries with higher cost that may remain in the frontier
    if current in astarData.closed:
        return False, []

    # 盖章标记：此节点已达绝对最优，不用再管了
    # Mark as expanded: this node has been settled with its optimal g-score
    astarData.closed.add(current)

    # 扩展邻居 / Expand successors of the current node
    for successor, action, step_cost in problem.getSuccessors(current):
        if successor in astarData.closed:
            continue

        tentative_g = astarData.g_scores[current] + step_cost

        # 只有找到更短的路，才去更新数据和队列
        # Only update if a shorter path to successor is found
        if successor not in astarData.g_scores or tentative_g < astarData.g_scores[successor]:
            astarData.g_scores[successor] = tentative_g
            astarData.came_from[successor] = (current, action)

            f_score = tentative_g + astar_heuristic(successor, astarData.goal)
            # 使用 update 覆盖队列里的老数据，进一步减少冗余扩展
            # Use update to overwrite stale frontier entries, reducing redundant expansions
            astarData.frontier.update(successor, f_score)

    # 本轮没找到终点，继续 / Goal not found this iteration, continue loop
    return False, []

def astar_heuristic(current, goal):
    """
    可接受且一致的启发函数：曼哈顿距离。
    Admissible and consistent heuristic: Manhattan distance.

    曼哈顿距离不会高估实际步数（网格移动，无对角线），
    且满足三角不等式，保证 A* 找到最优路径。
    Manhattan distance never overestimates actual steps (grid movement, no diagonals),
    and satisfies the triangle inequality, ensuring A* finds the optimal path.
    """
    x1, y1 = current
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)
