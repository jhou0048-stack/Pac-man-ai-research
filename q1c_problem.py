import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


# ================================================================
# Q1c Problem: Full Map Food Collection Search Problem Definition
# Q1c 问题定义：大地图全量食物收集搜索问题
#
# 功能 / Purpose:
#   定义收集地图上所有食物点的搜索问题，供 Q1c 高级规划求解器使用。
#   Defines the problem of collecting all food on the map for the Q1c solver.
#
# 状态表示 / State Representation:
#   (pos, remaining) —— 当前位置 + 尚未收集的食物集合（frozenset）
#   (pos, remaining) —— Current position + frozenset of uncollected food dots
#
# 与 Q1b 的区别 / Difference from Q1b:
#   Q1b 在初始化时预计算 BFS 距离表（适合小地图）；
#   Q1c 不做预计算，将距离计算完全交由 solver 按需处理（适合大地图）。
#   Q1b pre-computes BFS distance tables at init (suitable for small maps);
#   Q1c does no pre-computation — all distance calculations are handled by the solver on demand.
# ================================================================

class q1c_problem:
    """
    A search problem associated with finding a path that collects all of the
    food (dots) in a Pacman game.
    Some useful data has been included here for you
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState

        初始化：存储地图尺寸、墙壁信息，以及所有食物坐标的 frozenset。
        Init: store map dimensions, wall grid, and frozenset of all food positions.
        initial_food 在 solver 中用于模拟得分计算。
        initial_food is used by the solver to simulate score computation.
        """
        self.startingGameState: GameState = gameState
        self.walls = gameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height

        # 构建初始食物集合（frozenset 便于哈希和集合运算）
        # Build the initial food set (frozenset enables hashing and set operations)
        food_grid = gameState.getFood()
        self.initial_food = frozenset(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if food_grid[x][y]
        )

    @log_function
    def getStartState(self):
        """
        返回初始状态：(Pac-Man 起始位置, 初始食物集合去掉起始位置处的食物)
        Returns the initial state: (Pac-Man start pos, initial food minus food at start pos)
        """
        pos = self.startingGameState.getPacmanPosition()
        pos = (int(pos[0]), int(pos[1]))
        # 若起点本身有食物，视为已收集 / If start pos has food, treat it as already eaten
        remaining = frozenset(f for f in self.initial_food if f != pos)
        return (pos, remaining)

    @log_function
    def isGoalState(self, state):
        """
        目标测试：remaining 为空即表示所有食物已收集。
        Goal test: remaining is empty — all food has been collected.
        """
        _, remaining = state
        return len(remaining) == 0

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor

        后继函数：尝试四个方向，若目标格不是墙则生成后继状态。
        Successor function: try all four directions; generate successor if cell is not a wall.

        若新位置含有食物则自动从 remaining 中移除。
        If the new position contains food, it is automatically removed from remaining.
        每步代价固定为 1 / Step cost is always 1.
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
                # 到达食物位置时自动收集 / Automatically collect food at the new position
                nrem = frozenset(f for f in remaining if f != npos)
                successors.append(((npos, nrem), action, 1))
        return successors
