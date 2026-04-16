import logging
from collections import deque

import util
from game import Actions, Directions
from logs.search_logger import log_function
from pacman import GameState


# ================================================================
# Q1b Problem: Multi-Food Collection Search Problem Definition
# Q1b 问题定义：收集所有食物的搜索问题
#
# 功能 / Purpose:
#   定义收集地图上所有食物点的搜索问题，为 A* 提供状态空间与代价查询。
#   Defines the search problem for collecting all food dots on the map,
#   providing state space and cost lookups for A*.
#
# 状态表示 / State Representation:
#   (pos, remaining) —— 当前位置 + 尚未收集的食物集合（frozenset）
#   (pos, remaining) —— Current position + frozenset of uncollected food dots
#
# 关键优化 / Key Optimisations:
#   - 预处理阶段对每个食物节点做 BFS，构建全局真实距离表。
#     Pre-compute BFS from each food node to build a global real-distance table.
#   - 启发式查询时直接 O(1) 查表，无需运行时 BFS，大幅减少节点展开数。
#     Heuristic queries use O(1) table lookups, avoiding runtime BFS overhead.
# ================================================================

class q1b_problem:
    """
    Q1b: 收集所有可达食物，最小化 A* 节点展开数。

    关键优化：
    - 对每个食物节点做 BFS，记录地图上所有位置到该食物的真实距离
    - MST 启发式中任意位置到任意食物都能查到真实距离
    - 比只记录食物节点间距离更准确（覆盖全部搜索路径）

    Q1b: Collect all reachable food dots, minimising A* node expansions.

    Key optimisations:
    - BFS from each food node records real distances from all positions to that food.
    - Any position-to-food distance is O(1) lookup during heuristic computation.
    - More accurate than storing only food-to-food distances (covers all search paths).
    """

    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        预处理阶段：构建全局 BFS 距离表。
        Pre-processing phase: build the global BFS distance table.

        步骤 / Steps:
        1. 从起始位置 BFS，过滤出所有可达食物。
           BFS from start to identify all reachable food dots.
        2. 对每个可达食物做 BFS，记录地图上任意位置到该食物的真实距离。
           BFS from each food to record real distances from every position.
        3. 构建食物节点之间的距离字典，供 TSP 启发式使用。
           Build a food-to-food distance dict for use in the TSP heuristic.
        """
        self.startingGameState = gameState
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height

        start_pos = gameState.getPacmanPosition()
        start_pos = (int(start_pos[0]), int(start_pos[1]))

        food_grid = gameState.getFood()
        all_food = frozenset(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if food_grid[x][y]
        )

        # 只保留可达食物 / Keep only food dots reachable from the start
        reachable = self._bfs_all_dists(start_pos)
        self.reachable_food = frozenset(f for f in all_food if f in reachable)
        self._start_pos = start_pos

        # 对每个食物节点做 BFS，记录所有位置到该食物的真实距离
        # dist_to_food[food][pos] = 从 pos 到 food 的最短路径长度
        # BFS from each food node: dist_to_food[food][pos] = shortest path from pos to food
        self._dist_to_food = {}
        for food in self.reachable_food:
            self._dist_to_food[food] = self._bfs_all_dists(food)

        # 食物节点间距离（用于 MST 中 food-to-food 边）
        # Food-to-food distances (used for food-to-food edges in TSP/MST heuristic)
        food_list = list(self.reachable_food)
        self._food_dist = {}
        for i, fi in enumerate(food_list):
            for fj in food_list:
                self._food_dist[(fi, fj)] = self._dist_to_food[fi].get(fj, 10**6)

    def _bfs_all_dists(self, start):
        """
        从 start 出发做 BFS，返回所有可达位置的最短距离字典。
        BFS from start; returns a dict mapping every reachable position to its shortest distance.
        """
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and not self.walls[nx][ny]
                        and npos not in dist):
                    dist[npos] = dist[(x, y)] + 1
                    queue.append(npos)
        return dist

    def dist_to_food(self, pos, food):
        """
        任意位置 pos 到某个食物节点的 BFS 真实距离。
        Real BFS distance from any position pos to a food dot.
        若无法查到则退化为曼哈顿距离 / Falls back to Manhattan distance if not in table.
        """
        return self._dist_to_food[food].get(
            pos, abs(pos[0]-food[0]) + abs(pos[1]-food[1]))

    def food_to_food(self, fi, fj):
        """
        食物节点间的 BFS 真实距离。
        Real BFS distance between two food nodes.
        """
        return self._food_dist.get((fi, fj),
               abs(fi[0]-fj[0]) + abs(fi[1]-fj[1]))

    @log_function
    def getStartState(self):
        """
        返回初始状态：(起始位置, 所有可达食物的集合)
        Returns the initial state: (start position, frozenset of all reachable food dots)
        若起点本身有食物，从集合中移除（视为已收集）。
        If the start position contains food, it is removed (treated as already collected).
        """
        pos = self._start_pos
        remaining = frozenset(f for f in self.reachable_food if f != pos)
        return (pos, remaining)

    @log_function
    def isGoalState(self, state):
        """
        目标测试：remaining 为空集合即为全部食物已收集。
        Goal test: remaining is empty — all food has been collected.
        """
        _, remaining = state
        return len(remaining) == 0

    @log_function
    def getSuccessors(self, state):
        """
        后继函数：从当前位置尝试四个方向，若不是墙壁则生成后继状态。
        Successor function: try all four directions; generate successor if cell is not a wall.

        后继状态中，若新位置是食物则从 remaining 中移除。
        If the new position contains food, remove it from remaining in the successor state.
        每步代价为 1 / Step cost is 1.
        """
        successors = []
        pos, remaining = state
        x, y = pos
        for action in (Directions.NORTH, Directions.SOUTH,
                       Directions.EAST, Directions.WEST):
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(x + dx), int(y + dy)
            if (0 <= nx < self.width and 0 <= ny < self.height
                    and not self.walls[nx][ny]):
                npos = (nx, ny)
                # 若新位置是食物则从 remaining 中移除 / Remove food at new position if present
                nrem = frozenset(f for f in remaining if f != npos)
                successors.append(((npos, nrem), action, 1))
        return successors
