#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
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
# Q1b: A* Search — Multiple Food Dots (TSP-style)
# Q1b: A* 搜索 —— 多食物收集（旅行商问题风格）
#
# 目标 / Goal:
#   使用 A* 算法找到收集所有食物的最短路径，最小化节点展开数。
#   Use A* to find the shortest path collecting all food dots,
#   minimising the number of node expansions.
#
# 状态 / State:
#   (pos, remaining) —— 当前位置 + 尚未收集的食物集合（frozenset）
#   (pos, remaining) —— Current position + frozenset of uncollected food dots
#
# 启发函数 / Heuristic:
#   TSP 精确解：穷举剩余食物（≤4!）所有访问顺序，取 BFS 真实距离最短路径。
#   TSP exact solution: enumerate all orderings of remaining food (≤4!),
#   using real BFS distances — tighter lower bound than MST.
# ================================================================

from itertools import permutations


class AStarData:
    # 存储 A* 运行时所需的全部状态
    # Holds all runtime state for A* execution
    def __init__(self):
        self.pq = util.PriorityQueue()   # 优先队列：按 (f, -g) 排序 / Priority queue ordered by (f, -g)
        self.visited = {}                # visited[state] = 已知最小 g 值 / Best known g-score per state
        self.path = []                   # 当前最优动作序列 / Current best action sequence
        self.terminate = False           # 是否终止标志 / Termination flag


def astar_initialise(problem: q1b_problem) -> AStarData:
    """
    初始化 A* 数据结构：将起始状态压入优先队列。
    Initialise A* data structures: push the start state onto the priority queue.
    """
    astarData = AStarData()
    start_state = problem.getStartState()
    h = _tsp_heuristic(start_state, problem)
    # priority = (f, -g)：f相同时优先展开g更大的节点（更近目标）
    # priority = (f, -g): tie-break by preferring nodes closer to goal when f is equal
    astarData.pq.push((0, start_state, []), (h, 0))
    astarData.visited[start_state] = 0
    return astarData


def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    """
    A* 主循环体，每次调用处理一个节点。
    A* main loop body — processes one node per call.

    返回 / Returns:
        (terminate: bool, path: list) —— 是否终止 + 动作序列
        (terminate: bool, path: list) —— whether done + action sequence
    """
    # 队列为空 → 无解 / Empty frontier → no solution
    if astarData.pq.isEmpty():
        astarData.terminate = True
        return astarData.terminate, []

    g_current, current_state, path = astarData.pq.pop()

    # 过滤过期节点：visited 中已有更优 g 值，跳过
    # Skip stale entries where a better g-score has already been recorded
    if astarData.visited.get(current_state, float('inf')) < g_current:
        return astarData.terminate, astarData.path

    # 目标检测：remaining 为空即为全部收集 / Goal: all food collected (remaining is empty)
    if problem.isGoalState(current_state):
        astarData.terminate = True
        astarData.path = path
        return astarData.terminate, astarData.path

    # 扩展后继节点 / Expand successors
    for successor, action, cost in problem.getSuccessors(current_state):
        new_g = g_current + cost
        # 只有发现更短路径才更新 / Only update if a shorter path is found
        if new_g < astarData.visited.get(successor, float('inf')):
            astarData.visited[successor] = new_g
            h = _tsp_heuristic(successor, problem)
            # tie-break：f相同时，-g更小（g更大）优先
            # Tie-break: when f is equal, prefer larger g (closer to goal)
            astarData.pq.push((new_g, successor, path + [action]),
                              (new_g + h, -new_g))

    return astarData.terminate, astarData.path


def _tsp_heuristic(state, problem: q1b_problem) -> float:
    """
    TSP 精确解启发式（BFS 真实距离）。

    对剩余食物（最多4个）穷举所有访问顺序（最多 4!=24 种），
    找到从当前位置出发访问所有食物的最短路径长度。

    可接受且一致的启发式，比 MST 更紧的下界。
    配合 tie-breaking（f相同时优先展开g大的节点），
    可大幅减少总节点展开数。

    TSP exact heuristic (BFS real distances).

    Enumerates all orderings of remaining food (up to 4! = 24),
    finds the shortest path from current pos visiting all food.

    Admissible and consistent; tighter lower bound than MST.
    Combined with tie-breaking (prefer larger g when f is equal),
    significantly reduces total node expansions.
    """
    pos, remaining = state
    if not remaining:
        return 0

    rem_list = list(remaining)

    # 只剩一个食物：直接查 BFS 真实距离 / Single food left: direct BFS lookup
    if len(rem_list) == 1:
        return problem.dist_to_food(pos, rem_list[0])

    # 多个食物：穷举全排列取最小总代价 / Multiple food: brute-force all permutations
    best = float('inf')
    for perm in permutations(rem_list):
        cost = problem.dist_to_food(pos, perm[0])
        for i in range(len(perm) - 1):
            cost += problem.food_to_food(perm[i], perm[i + 1])
        if cost < best:
            best = cost
    return best


def astar_heuristic(state) -> float:
    """
    曼哈顿距离 MST（fallback，实际使用 _tsp_heuristic）
    Manhattan distance MST (fallback; actual search uses _tsp_heuristic)
    """
    pos, remaining = state
    if not remaining:
        return 0
    nodes = [pos] + list(remaining)
    # Prim 算法构建最小生成树 / Prim's algorithm to build MST
    min_edge = {n: abs(nodes[0][0]-n[0])+abs(nodes[0][1]-n[1])
                for n in nodes[1:]}
    mst = 0
    while min_edge:
        best = min(min_edge, key=min_edge.__getitem__)
        mst += min_edge.pop(best)
        for n in min_edge:
            d = abs(best[0]-n[0]) + abs(best[1]-n[1])
            if d < min_edge[n]:
                min_edge[n] = d
    return mst
