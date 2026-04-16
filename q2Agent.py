import logging
import time
from collections import deque

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState

# ================================================================
# Q2: Adversarial Search Agent — Alpha-Beta Pruning + Transposition Table
# Q2: 对抗搜索智能体 —— Alpha-Beta 剪枝 + 转置表
#
# 功能 / Purpose:
#   在有限时间内，使用迭代加深 Alpha-Beta 搜索为 Pac-Man 选取最优动作。
#   Within a limited time budget, use iterative-deepening Alpha-Beta search
#   to select the optimal action for Pac-Man.
#
# 核心模块 / Core Modules:
#   1. registerInitialState  —— 地形预处理，构建拓扑地图与全局距离表
#      registerInitialState  —— terrain pre-processing, build topology map & global dist table
#   2. getAction             —— 迭代加深决策中枢，动态分配每步时间
#      getAction             —— iterative-deepening decision hub, dynamic time allocation
#   3. _initiate_alpha_beta  —— Alpha-Beta 搜索根节点，整合转置表优先排序
#      _initiate_alpha_beta  —— Alpha-Beta root, integrates transposition table for move ordering
#   4. _ab_max_node          —— Pac-Man（Max 玩家）节点
#      _ab_max_node          —— Pac-Man (Max player) node
#   5. _ab_min_node          —— 幽灵（Min 玩家）节点
#      _ab_min_node          —— Ghost (Min player) node
#   6. _strategic_assessment —— 启发式状态评估函数
#      _strategic_assessment —— heuristic state evaluation function
# ================================================================

def scoreEvaluationFunction(currentGameState):
    """默认评估函数：直接返回游戏得分 / Default eval fn: return raw game score"""
    return currentGameState.getScore()

class TimeExhaustedError(Exception):
    """自定义超时阻断器，用于控制每步的搜索时间
    Custom exception to abort search when the per-step time budget is exhausted"""
    pass

# 转置表状态标识 (TT Flags)
# Transposition table entry flags
TT_EXACT = 'E'  # 精确值 / Exact value
TT_ALPHA = 'A'  # Alpha 截断（上界）/ Alpha cutoff (upper bound)
TT_BETA  = 'B'  # Beta 截断（下界）/ Beta cutoff (lower bound)

class Q2_Agent(Agent):
    """
    Q2 对抗搜索智能体。
    Q2 adversarial search agent.

    使用迭代加深 Alpha-Beta 剪枝 + 转置表（记忆搜索）
    在每步固定时间内尽可能深入搜索，选取最优动作。
    Uses iterative-deepening Alpha-Beta pruning + transposition table (memoized search)
    to search as deep as possible within the per-step time budget.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        # 核心时间与进度管理 / Core time and progress tracking
        self.session_start = None       # 游戏开始时间戳 / Game start timestamp
        self.time_budget = 29.0         # 全局时间预算（秒）/ Global time budget (seconds)
        self.step_counter = 0           # 已执行步数 / Steps taken so far
        self.starting_food_count = 0    # 初始食物总数 / Initial total food count

        # 统一拓扑地形缓存 / Unified topological terrain cache
        self.grid_matrix = None         # 墙壁网格 / Wall grid
        self.cols = 0                   # 地图宽度 / Map width
        self.rows = 0                   # 地图高度 / Map height
        self.topo_map = {}              # 每个可行格的拓扑属性 / Topological attributes per walkable cell

        # 导航与记忆中枢 / Navigation and memory hub
        self.nav_memo = {}              # 按需 BFS 缓存 / On-demand BFS cache
        self.global_dist_table = {}     # 全局距离表（小地图预计算）/ Global distance table (pre-computed for small maps)
        self.is_fully_mapped = False    # 是否已完成全量预计算 / Whether full pre-computation is done

        self.memory_bank = {}           # 转置表 / Transposition table
        self.memory_limit = 350000      # 转置表最大条目数 / Max entries in transposition table

    # ================================================================
    # 地形测绘与预处理
    # Terrain Mapping & Pre-processing
    # ================================================================
    def registerInitialState(self, state):
        """
        游戏开始时调用一次：预处理地形信息，构建拓扑地图，
        对小地图全量预计算 BFS 距离表。
        Called once at game start: pre-process terrain, build topology map,
        and for small maps pre-compute the full BFS distance table.

        拓扑属性 / Topological attributes per cell:
            exits    —— 有效出口数（判断分叉口、走廊、死胡同）
                        Number of valid exits (identifies junctions, corridors, dead ends)
            hub_dist —— 到最近分叉口的距离（BFS 多源）
                        Distance to nearest junction (multi-source BFS)
        """
        self.session_start = time.time()
        self.grid_matrix = state.getWalls()
        self.cols, self.rows = self.grid_matrix.width, self.grid_matrix.height
        self.starting_food_count = state.getNumFood()

        # 遍历所有可行格，统计出口数并记录分叉口
        # Scan all walkable cells, count exits, record junctions (exits >= 3)
        hubs = []
        for x in range(self.cols):
            for y in range(self.rows):
                if not self.grid_matrix[x][y]:
                    valid_exits = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.cols and 0 <= ny < self.rows and not self.grid_matrix[nx][ny]:
                            valid_exits += 1

                    self.topo_map[(x, y)] = {'exits': valid_exits, 'hub_dist': 0, 'escapes': 0}
                    if valid_exits >= 3:
                        hubs.append((x, y))

        # 多源 BFS：计算每个可行格到最近分叉口的距离
        # Multi-source BFS: compute distance from each walkable cell to the nearest junction
        if hubs:
            visited = {h: 0 for h in hubs}
            queue = deque(hubs)
            while queue:
                curr_x, curr_y = queue.popleft()
                current_d = visited[(curr_x, curr_y)]
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows and not self.grid_matrix[nx][ny]:
                        if (nx, ny) not in visited:
                            visited[(nx, ny)] = current_d + 1
                            queue.append((nx, ny))

            for pos in self.topo_map:
                self.topo_map[pos]['hub_dist'] = visited.get(pos, 0)

        # 小地图（≤200 个可行格）预计算全量 BFS 距离表
        # For small maps (≤200 walkable cells), pre-compute the full BFS distance table
        walkable_tiles = len(self.topo_map)
        if walkable_tiles <= 200:
            self._compile_full_matrix()

    def _compile_full_matrix(self):
        """
        全量预计算：对每个可行格做 BFS，填充全局距离表。
        Full pre-computation: BFS from each walkable cell to fill the global distance table.
        结果以 frozenset 为键（无向图），节省一半存储。
        Uses frozenset as key (undirected graph) to halve storage.
        """
        for start_pos in self.topo_map.keys():
            distances = self._execute_bfs(start_pos)
            for target_pos, d in distances.items():
                self.global_dist_table[frozenset([start_pos, target_pos])] = d
        self.is_fully_mapped = True

    def _execute_bfs(self, origin):
        """
        从 origin 出发 BFS，返回所有可达位置的最短距离字典。
        BFS from origin; returns a dict of shortest distances to all reachable positions.
        """
        routes = {origin: 0}
        queue = deque([origin])
        while queue:
            cx, cy = queue.popleft()
            cost = routes[(cx, cy)]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and not self.grid_matrix[nx][ny]:
                    if (nx, ny) not in routes:
                        routes[(nx, ny)] = cost + 1
                        queue.append((nx, ny))
        return routes

    def _calculate_route(self, p_a, p_b):
        """
        查询两点之间的最短路径步数。
        Query the shortest path distance between two points.

        优先级 / Priority:
        1. 全量预计算表（小地图，O(1) 查询）
           Full pre-computed table (small maps, O(1) lookup)
        2. 按需 BFS 缓存（大地图，懒加载）
           On-demand BFS cache (large maps, lazy loading)
        3. 曼哈顿距离（fallback，仅在无 BFS 数据时使用）
           Manhattan distance (fallback, only when BFS data unavailable)
        """
        a = (int(p_a[0]), int(p_a[1]))
        b = (int(p_b[0]), int(p_b[1]))
        if a == b: return 0

        if self.is_fully_mapped:
            return self.global_dist_table.get(frozenset([a, b]), 999999)

        if a not in self.nav_memo:
            self.nav_memo[a] = self._execute_bfs(a)
        return self.nav_memo[a].get(b, abs(a[0]-b[0]) + abs(a[1]-b[1]))

    def _generate_fingerprint(self, state):
        """
        生成游戏状态的哈希指纹，用于转置表键值。
        Generate a hash fingerprint of the game state for use as transposition table key.

        包含：食物分布、胶囊位置、Pac-Man 位置、幽灵惊吓计时器、幽灵位置。
        Includes: food grid hash, capsule positions, Pac-Man position,
                  ghost scared timers, ghost positions.
        """
        return (
            hash(state.getFood()),
            tuple(state.getCapsules()),
            state.getPacmanPosition(),
            tuple(gs.scaredTimer for gs in state.getGhostStates()),
            tuple(state.getGhostPosition(i) for i in range(1, state.getNumAgents()))
        )

    # ================================================================
    # 决策中枢与迭代加深搜寻
    # Decision Hub & Iterative-Deepening Search
    # ================================================================
    @log_function
    def getAction(self, gameState: GameState):
        """
        每步决策入口：使用迭代加深 Alpha-Beta 搜索在时间预算内选取最优动作。
        Per-step decision entry: use iterative-deepening Alpha-Beta search
        to select the best action within the time budget.

        动态时间分配策略 / Dynamic time allocation:
            alloc_time = min(剩余时间 / max(食物数+10, 20), 2.5 秒)，但至少保留 0.04 秒
            alloc_time = min(remaining / max(food+10, 20), 2.5s), minimum 0.04s

        迭代加深：每层搜索若上一层用时×4超出剩余时间，则终止加深。
        Iterative deepening: stop deepening if last_layer_time×4 > remaining budget.
        """
        logger = logging.getLogger('root')
        logger.info('Executing Deep-Dive Tactical Agent')

        current_time = time.time()
        if not self.session_start:
            self.registerInitialState(gameState)

        self.step_counter += 1
        time_elapsed = current_time - self.session_start
        time_remaining = self.time_budget - time_elapsed

        valid_moves = gameState.getLegalActions(0)
        if not valid_moves: return Directions.STOP
        if len(valid_moves) == 1: return valid_moves[0]  # 唯一合法动作，直接返回 / Only one legal move

        # 剩余时间极少时，使用紧急后备策略
        # Fallback to emergency strategy when time is nearly exhausted
        if time_remaining <= 0.05:
            return self._emergency_fallback(gameState, valid_moves)

        food_left = gameState.getNumFood()
        # 微调：确保残局即使时间紧凑，也保留最基础的 0.04 秒思考时间避免送头
        # Ensure at least 0.04s thinking time in endgame to avoid suicidal moves
        alloc_time = max(min(time_remaining / max(food_left + 10, 20), 2.5), 0.04)
        hard_deadline = current_time + alloc_time

        agent_count = gameState.getNumAgents()
        chosen_move = self._emergency_fallback(gameState, valid_moves)  # 默认后备 / Default fallback

        # 转置表过大时清空，防止内存溢出 / Clear transposition table if too large
        if len(self.memory_bank) > self.memory_limit:
            self.memory_bank.clear()

        # 迭代加深主循环 / Iterative deepening main loop
        last_layer_time = 0.0
        for current_depth in range(1, 60):
            if hard_deadline - time.time() <= 0.005: break
            # 若上一层用时×4超出剩余时间，预计下一层无法完成，提前终止
            # If last_layer_time×4 > remaining, next layer likely won't finish — stop
            if current_depth > 1 and last_layer_time * 4.0 > (hard_deadline - time.time()): break

            layer_start = time.time()
            try:
                result_move = self._initiate_alpha_beta(gameState, current_depth, agent_count, hard_deadline)
                if result_move: chosen_move = result_move
            except TimeExhaustedError:
                break  # 超时中断，使用上一层结果 / Timed out, use result from previous depth
            last_layer_time = max(time.time() - layer_start, 0.0001)

        return chosen_move

    def _emergency_fallback(self, state, actions):
        """
        紧急后备策略：对每个合法动作执行一步评估，选取得分最高的动作。
        Emergency fallback: evaluate one step for each legal action, pick the best.
        跳过 STOP 动作（避免原地等死）/ Skip STOP (avoid staying put).
        """
        top_score = -float('inf')
        top_action = Directions.STOP
        for act in actions:
            if act == Directions.STOP: continue
            val = self._strategic_assessment(state.generateSuccessor(0, act))
            if val > top_score:
                top_score, top_action = val, act
        return top_action

    # ================================================================
    # Alpha-Beta 剪枝与转置表
    # Alpha-Beta Pruning & Transposition Table
    # ================================================================
    def _initiate_alpha_beta(self, state, target_depth, num_agents, deadline):
        """
        Alpha-Beta 搜索根节点（Pac-Man 的 Max 层）。
        Alpha-Beta search root node (Pac-Man's Max layer).

        转置表优先排序：将上一次搜索记录的最优动作排到队首，
        提升 Alpha-Beta 剪枝效率（最优情况下剪去近一半节点）。
        Transposition table move ordering: place the previously best action first
        to improve Alpha-Beta pruning efficiency (nearly halves nodes in best case).
        """
        alpha, best_val = -float('inf'), -float('inf')
        best_act = None

        # 从转置表中取出上一次搜索的最优动作，优先排序
        # Retrieve previous best action from TT for move ordering
        state_hash = self._generate_fingerprint(state)
        cached_data = self.memory_bank.get(state_hash)
        priority_act = cached_data[3] if cached_data else None

        # 构建候选动作列表，结合启发式评分排序
        # Build candidate move list, sorted by heuristic score (+ TT priority boost)
        move_pool = []
        for act in state.getLegalActions(0):
            nxt = state.generateSuccessor(0, act)
            heuristic = self._strategic_assessment(nxt)
            if act == Directions.STOP: heuristic -= 100   # 惩罚 STOP / Penalise STOP
            if act == priority_act: heuristic += 10000    # 优先排序 TT 最优动作 / Boost TT best action
            move_pool.append((heuristic, act, nxt))

        move_pool.sort(key=lambda x: x[0], reverse=True)  # 高分优先 / Highest score first

        for _, act, nxt in move_pool:
            if time.time() >= deadline - 0.002: raise TimeExhaustedError()
            val = self._ab_min_node(nxt, 1, 0, alpha, float('inf'), target_depth, num_agents, deadline)
            if val > best_val:
                best_val, best_act = val, act
            alpha = max(alpha, val)

        # 将本次搜索结果存入转置表 / Store this search result in transposition table
        self.memory_bank[state_hash] = (target_depth, TT_EXACT, best_val, best_act)
        return best_act

    def _ab_max_node(self, state, depth, alpha, beta, max_d, agents, dl):
        """
        Alpha-Beta Max 节点（Pac-Man 回合）。
        Alpha-Beta Max node (Pac-Man's turn).

        转置表查询 / Transposition table lookup:
            - TT_EXACT：直接返回精确值
              TT_EXACT: return exact value directly
            - TT_ALPHA：若缓存上界 ≤ alpha，剪枝
              TT_ALPHA: prune if cached upper bound ≤ alpha
            - TT_BETA：若缓存下界 ≥ beta，剪枝
              TT_BETA: prune if cached lower bound ≥ beta
        """
        if time.time() >= dl - 0.001: raise TimeExhaustedError()
        if state.isWin() or state.isLose() or depth >= max_d:
            return self._strategic_assessment(state)

        state_hash = self._generate_fingerprint(state)
        cached_data = self.memory_bank.get(state_hash)
        remaining_d = max_d - depth

        # 转置表命中且深度足够，尝试直接返回或剪枝
        # TT hit with sufficient depth: try direct return or pruning
        if cached_data and cached_data[0] >= remaining_d:
            c_flag, c_val = cached_data[1], cached_data[2]
            if c_flag == TT_EXACT: return c_val
            if c_flag == TT_ALPHA and c_val <= alpha: return c_val
            if c_flag == TT_BETA and c_val >= beta: return c_val

        moves = state.getLegalActions(0)
        if not moves: return self._strategic_assessment(state)

        # 将 TT 最优动作移到队首（动作排序提升剪枝效率）
        # Move TT best action to front (move ordering improves pruning)
        if cached_data and cached_data[3] in moves:
            moves.insert(0, moves.pop(moves.index(cached_data[3])))

        orig_a = alpha
        highest_v = -float('inf')
        opt_act = moves[0]

        for act in moves:
            nxt = state.generateSuccessor(0, act)
            val = self._ab_min_node(nxt, 1, depth, alpha, beta, max_d, agents, dl)
            if val > highest_v: highest_v, opt_act = val, act
            if highest_v > beta:
                # Beta 截断：最小化玩家不会允许此路径 / Beta cutoff: minimiser won't allow this
                self.memory_bank[state_hash] = (remaining_d, TT_BETA, highest_v, opt_act)
                return highest_v
            alpha = max(alpha, highest_v)

        # 存入转置表（精确值或 Alpha 上界）
        # Store in TT (exact value or alpha upper bound)
        flag = TT_ALPHA if highest_v <= orig_a else TT_EXACT
        self.memory_bank[state_hash] = (remaining_d, flag, highest_v, opt_act)
        return highest_v

    def _ab_min_node(self, state, agent_id, depth, alpha, beta, max_d, agents, dl):
        """
        Alpha-Beta Min 节点（幽灵回合）。
        Alpha-Beta Min node (Ghost's turn).

        支持多幽灵串行处理：同一深度层内依次处理每个幽灵（agent_id 递增），
        最后一个幽灵处理完后推进到下一深度的 Max 节点。
        Supports multiple ghosts in sequence: process each ghost (agent_id++) within
        the same depth level; after the last ghost, advance to the next Max node.
        """
        if time.time() >= dl - 0.001: raise TimeExhaustedError()
        if state.isWin() or state.isLose(): return self._strategic_assessment(state)

        moves = state.getLegalActions(agent_id)
        if not moves: return self._strategic_assessment(state)

        lowest_v = float('inf')
        for act in moves:
            nxt = state.generateSuccessor(agent_id, act)
            if agent_id == agents - 1:
                # 最后一个幽灵：推进到下一深度的 Max 节点
                # Last ghost: advance to next depth's Max node
                val = self._ab_max_node(nxt, depth + 1, alpha, beta, max_d, agents, dl)
            else:
                # 还有幽灵未行动：继续在同一深度处理下一幽灵
                # More ghosts to move: continue at same depth for next ghost
                val = self._ab_min_node(nxt, agent_id + 1, depth, alpha, beta, max_d, agents, dl)

            lowest_v = min(lowest_v, val)
            if lowest_v < alpha: return lowest_v  # Alpha 截断 / Alpha cutoff
            beta = min(beta, lowest_v)

        return lowest_v

    # ================================================================
    # 启发式状态评估中枢 (返璞归真版：平滑梯度 + 信任深搜)
    # Heuristic State Evaluation (smooth gradient + trust deep search)
    # ================================================================
    def _strategic_assessment(self, state):
        """
        启发式评估函数：综合游戏得分、食物距离、幽灵威胁、胶囊价值。
        Heuristic evaluation function: combines game score, food distances,
        ghost threats, and capsule value.

        设计原则 / Design principles:
            - 终局绝对判断优先（Win/Lose 直接返回极值）
              Terminal states take priority (Win/Lose → extreme values)
            - 平滑梯度引导搜索方向，避免梯度断崖导致的僵局
              Smooth gradients guide search direction, avoiding cliff-like discontinuities
            - 幽灵威胁使用陡峭但平滑的衰减，交由深搜处理包夹逻辑
              Ghost threat uses steep but smooth decay; encirclement logic delegated to deep search
            - 删除静态地形评估（_assess_terrain），避免误判限制走位
              Static terrain assessment removed to avoid false restrictions on movement
        """
        # 1. 终局绝对判断 / Terminal state check
        if state.isWin(): return 999999
        if state.isLose(): return -999999

        score = state.getScore()
        pac_pos = state.getPacmanPosition()

        # ----------------------------------------------------------------
        # 2. 食物评估 (Food)
        # 2. Food evaluation
        # ----------------------------------------------------------------
        food_grid = state.getFood().asList()
        food_count = len(food_grid)
        food_score = 0

        if food_count > 0:
            dists = [self._calculate_route(pac_pos, f) for f in food_grid]
            min_food_dist = min(dists)

            # 温和的残局乘数：确保吃豆人想赢，但不会为了 1 颗豆子送命
            # Mild endgame multiplier: ensure Pac-Man wants to win, but not at the cost of dying
            multiplier = 2.5 if food_count <= 4 else 1.5
            food_score -= multiplier * food_count      # 鼓励减少剩余食物 / Incentivise reducing food count
            food_score += 10.0 / (min_food_dist + 1)  # 奖励接近最近食物 / Reward proximity to nearest food

            # 保持移动连贯性，顺便考虑第二近的豆子
            # Maintain movement continuity, also consider second-nearest food
            if len(dists) > 1:
                dists.remove(min_food_dist)
                food_score += 3.0 / (min(dists) + 1)

        # ----------------------------------------------------------------
        # 3. 幽灵威胁评估 (Ghosts)
        # 3. Ghost threat evaluation
        # ----------------------------------------------------------------
        ghost_score = 0
        min_ghost_dist = 999

        for ghost in state.getGhostStates():
            d = self._calculate_route(pac_pos, ghost.getPosition())
            if ghost.scaredTimer > 0:
                # 狩猎模式：如果幽灵还在惊吓状态，且我们能追上，果断去吃！
                # Hunt mode: if ghost is scared and we can catch it, go for it!
                if ghost.scaredTimer > d + 1:
                    ghost_score += 300.0 / (d + 1)
            else:
                min_ghost_dist = min(min_ghost_dist, d)
                # 极其陡峭且平滑的死亡边界（取消原先复杂的包夹判定，交给 Alpha-Beta 预测）
                # Steep but smooth death boundary (encirclement logic delegated to Alpha-Beta)
                if d <= 1:
                    ghost_score -= 99999  # 必死无疑 / Certain death
                elif d == 2:
                    ghost_score -= 2000   # 极度危险 / Extreme danger
                elif d == 3:
                    ghost_score -= 400
                elif d <= 5:
                    ghost_score -= 100.0 / d

        # ----------------------------------------------------------------
        # 4. 胶囊/大力丸评估 (Capsules)
        # 4. Capsule evaluation
        # ----------------------------------------------------------------
        capsules = state.getCapsules()
        cap_score = 0
        if capsules:
            cap_score -= 15.0 * len(capsules)  # 鼓励吃掉大力丸 / Incentivise eating capsules
            c_dists = [self._calculate_route(pac_pos, c) for c in capsules]
            min_cap_dist = min(c_dists)

            # 当幽灵逼近时（距离 <= 5），大幅提高吃大力丸的优先级，实现"极限反杀"
            # When ghosts are close (distance ≤ 5), greatly boost capsule priority for counter-attack
            if min_ghost_dist <= 5:
                cap_score += 500.0 / (min_cap_dist + 1)
            else:
                cap_score += 10.0 / (min_cap_dist + 1)

        # 删除了原先的 _assess_terrain
        # 理由：静态检查地形会产生误判，反而限制了吃豆人的走位。
        # 让 Alpha-Beta 树通过预测未来 5 步的 ghost_score 来自然规避死胡同！
        # Removed _assess_terrain:
        # Reason: static terrain checks cause false positives that restrict Pac-Man's movement.
        # Let the Alpha-Beta tree naturally avoid dead ends by predicting ghost_score over 5 steps ahead.

        return score + food_score + ghost_score + cap_score
