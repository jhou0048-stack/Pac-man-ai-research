import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


# ================================================================
# Q1a Problem: Single Food Dot Search Problem Definition
# Q1a 问题定义：寻找单一食物点的搜索问题
#
# 功能 / Purpose:
#   定义 A* 搜索所需的状态空间、初始状态、目标测试、后继函数和代价函数。
#   Defines the state space, start state, goal test, successor function,
#   and cost function needed by A* search.
#
# 状态表示 / State Representation:
#   (x, y) —— Pac-Man 的当前坐标位置。
#   (x, y) —— Current (x, y) coordinate of Pac-Man.
#
# 目标 / Goal:
#   到达地图上唯一的食物点位置。
#   Reach the single food dot on the map.
# ================================================================

class q1a_problem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState

        初始化时缓存目标食物位置，避免在每次循环中重复调用耗时的 asList()。
        Cache the goal food position at init to avoid repeated costly asList() calls in the loop.
        """
        self.startingGameState: GameState = gameState

        # 【满分优化关键】：提前缓存目标点，坚决避免在循环里调用耗时的 asList()
        # [Key optimisation]: cache the goal position upfront to avoid calling asList() in every loop iteration
        food_list = self.startingGameState.getFood().asList()
        self.goal = food_list[0] if food_list else None

    @log_function
    def getStartState(self):
        """
        返回Pac-man的初始位置（x，y）
        Returns Pac-Man's starting position as (x, y).
        """
        return self.startingGameState.getPacmanPosition()

    @log_function
    def isGoalState(self, state):
        """
        判断当前是否是目标状态：当前位置 == 食物位置即为目标。
        Goal test: returns True if the current position equals the food position.
        """
        return state == self.goal

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

        后继函数：对当前位置尝试四个方向（北/南/东/西），
        若目标格不是墙壁则加入后继列表，每步代价为 1。
        Successor function: try all four directions from the current position.
        Add to successors if the target cell is not a wall; step cost is always 1.
        """
        # ------------------------------------------
        "*** YOUR CODE HERE ***"
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            # 检查目标格是否为墙 / Check if the target cell is a wall
            if not self.startingGameState.hasWall(nextx, nexty):
                nextState = (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))
        return successors
