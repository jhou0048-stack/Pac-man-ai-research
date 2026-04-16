#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

# ================================================================
# Q1c: Large-Map Food Collection Planner
# Q1c: 大地图全量食物收集规划器
#
# 功能 / Purpose:
#   在时间预算（8.6 秒）内，为大地图上的 Pac-Man 规划收集所有食物的最优/近优路径。
#   Within a time budget (8.6 s), plan an optimal/near-optimal path for Pac-Man
#   to collect all food on large maps.
#
# 整体策略 / Overall Strategy:
#   1. 复杂度预估 → 超大地图用快速树覆盖，跳过精确求解
#      Complexity estimate → very large maps use fast tree-cover, skip exact solver
#   2. 预处理 BFS 距离表，覆盖起始点 + 所有食物节点之间的最短路径
#      Pre-compute BFS distance/path table covering start + all food nodes
#   3. 小地图：精确 A* 求解
#      Small maps: exact A* with MST+nearest+farthest heuristic
#   4. 中大地图：多种贪心启发式（最近邻、前瞻、最远优先）生成候选游览序列
#      Medium/large maps: multiple greedy heuristics (nearest, lookahead, farthest-first)
#   5. 2-opt 局部搜索改善游览序列
#      2-opt local search to improve tour quality
#   6. 3-opt 扰动 + 模拟退火式接受策略进行迭代优化
#      3-opt perturbation + simulated annealing acceptance for iterative improvement
#   7. 得分感知束搜索：以实际游戏分数为目标，选取最高得分前缀
#      Score-aware beam search: maximise actual game score; trim to best-score prefix
# ================================================================

import random
import time
from collections import deque

from game import Directions


# ================================================================
# 主求解函数 / Main Solver Entry Point
# ================================================================
def q1c_solver(problem: q1c_problem):
    """
    Q1c 主求解器：在时间预算内返回最佳动作序列。
    Q1c main solver: returns the best action sequence within the time budget.
    """
    start_time = time.time()
    planning_deadline = start_time + 8.6  # 总时间预算 8.6 秒 / Total time budget: 8.6 seconds

    start_state = problem.getStartState()
    start_pos, initial_food = start_state
    if not initial_food:
        return []  # 无食物，直接返回 / No food, nothing to do

    walls = problem.walls
    width, height = problem.width, problem.height
    food_list = list(initial_food)

    # ----------------------------------------------------------------
    # 阶段 1：从起始位置 BFS，过滤不可达食物，估算复杂度
    # Phase 1: BFS from start, filter unreachable food, estimate complexity
    # ----------------------------------------------------------------
    # First BFS from start lets us:
    # 1) filter unreachable food early
    # 2) estimate whether full all-pairs preprocessing may timeout
    start_dist, start_parent = _bfs_dist_parent(start_pos, walls, width, height)
    reachable_food = [f for f in food_list if f in start_dist]
    if not reachable_food:
        return []

    # 复杂度过高时用快速树覆盖算法（无需构建全量距离表）
    # If complexity is too high, use fast tree-cover (no full distance table needed)
    complexity_est = len(reachable_food) * len(start_dist)
    if complexity_est > 950000 or len(reachable_food) > 900:
        actions = _fast_tree_cover_actions(
            start_pos=start_pos,
            start_parent=start_parent,
            reachable_food=reachable_food,
        )
        return _best_score_prefix(actions, start_pos, problem.initial_food)

    # ----------------------------------------------------------------
    # 阶段 2：预计算 BFS 最短路径距离表（start + 所有食物节点两两之间）
    # Phase 2: Pre-compute pairwise BFS shortest-path distance/path table
    # ----------------------------------------------------------------
    # Build shortest-path distance/path between start+food nodes once.
    key_nodes = [start_pos] + reachable_food
    dm = {}      # dm[(src, dst)] = 最短步数 / shortest step count
    paths = {}   # paths[(src, dst)] = 经过的位置序列 / sequence of positions on the path
    all_dists = {start_pos: start_dist}
    for src in key_nodes:
        if src == start_pos:
            dist, parent = start_dist, start_parent
        else:
            dist, parent = _bfs_dist_parent(src, walls, width, height)
        all_dists[src] = dist
        for dst in key_nodes:
            if src == dst or dst not in dist:
                continue
            dm[(src, dst)] = dist[dst]
            paths[(src, dst)] = _reconstruct_path(parent, src, dst)

    # ----------------------------------------------------------------
    # 阶段 3：精确 A* 求解（小实例）
    # Phase 3: Exact A* for small instances
    # ----------------------------------------------------------------
    # Small instances: use exact A* on reachable food with a stronger lower bound.
    exact_start_state = (
        start_pos,
        frozenset(f for f in reachable_food if f != start_pos),
    )
    # 根据食物数量动态调整精确求解的时间和节点预算
    # Dynamically adjust time/expansion budget based on food count
    if len(reachable_food) <= 60:
        exact_food_limit = 60
        exact_time_budget = 2.8
        exact_expansion_limit = 260000
    elif len(reachable_food) <= 90:
        exact_food_limit = 90
        exact_time_budget = 1.0
        exact_expansion_limit = 90000
    else:
        exact_food_limit = 0
        exact_time_budget = 0.0
        exact_expansion_limit = 0

    exact_actions = _solve_small_exact(
        start_state=exact_start_state,
        reachable_food=reachable_food,
        walls=walls,
        width=width,
        height=height,
        dm=dm,
        dist_cache=all_dists,
        deadline=min(planning_deadline, start_time + exact_time_budget),
        max_food_for_exact=exact_food_limit,
        max_expansions=exact_expansion_limit,
    )
    if exact_actions is not None:
        # 精确解找到，截取得分最高前缀返回 / Exact solution found, trim to best-score prefix
        return _best_score_prefix(exact_actions, start_pos, problem.initial_food)

    # ----------------------------------------------------------------
    # 阶段 4：贪心启发式 + 2-opt + 3-opt 迭代优化（中大地图）
    # Phase 4: Greedy heuristics + 2-opt + 3-opt iterative optimisation (medium/large maps)
    # ----------------------------------------------------------------
    food_set = set(reachable_food)
    start_food_eaten = {start_pos} if start_pos in food_set else set()

    # 预计算每段路径经过的食物集合（用于快速评估游览序列代价）
    # Pre-compute which food dots are on each path segment (for fast tour cost evaluation)
    seg_foods = {}
    for src in [start_pos] + reachable_food:
        for dst in reachable_food:
            if src == dst:
                continue
            path = paths.get((src, dst))
            if path is None:
                continue
            seg_foods[(src, dst)] = frozenset(p for p in path if p in food_set)

    def evaluate_tour(tour):
        """
        评估游览序列的总步数代价，同时返回精简后的有效节点序列。
        Evaluate total step cost of a tour; also returns the compacted effective node sequence.
        若路径不完整（有食物无法到达），返回无穷大。
        Returns infinity if the tour cannot reach all food.
        """
        eaten = set(start_food_eaten)
        cur = start_pos
        total = 0
        compact = []
        goal_food_count = len(food_set)

        for target in tour:
            if target in eaten:
                continue
            seg = paths.get((cur, target))
            if seg is None:
                return float('inf'), compact
            compact.append(target)
            total += len(seg)
            eaten.update(seg_foods[(cur, target)])
            cur = target
            if len(eaten) == goal_food_count:
                break

        if len(eaten) < goal_food_count:
            return float('inf'), compact
        return total, compact

    def nearest_tour(beam_width=1, rng=None, farthest_first=False):
        """
        最近邻贪心策略：每步选择距当前最近的未访问食物。
        Nearest-neighbour greedy: at each step pick the closest unvisited food.
        beam_width > 1 时随机从前 k 个候选中选取（增加多样性）。
        beam_width > 1: randomly pick from the top-k candidates (add diversity).
        farthest_first=True 时先跳到最远食物再贪心（避免早期绕路）。
        farthest_first=True: start at the farthest food, then greedy (avoids early detours).
        """
        rem = set(reachable_food)
        tour = []
        cur = start_pos
        if farthest_first and rem:
            first = max(rem, key=lambda f: dm[(cur, f)])
            tour.append(first)
            rem.remove(first)
            cur = first
        while rem:
            ordered = sorted(rem, key=lambda f: (dm[(cur, f)], f[0], f[1]))
            if rng is None or beam_width <= 1 or len(ordered) == 1:
                nxt = ordered[0]
            else:
                nxt = ordered[rng.randrange(min(beam_width, len(ordered)))]
            tour.append(nxt)
            rem.remove(nxt)
            cur = nxt
        return tour

    def lookahead_tour(k=6):
        """
        前瞻贪心策略：每步考虑距当前最近的 k 个候选，
        选择「到达代价 + 到下一最近食物代价」最小的节点。
        Lookahead greedy: at each step consider the k nearest candidates;
        pick the one minimising (reach cost + min cost to next food).
        """
        rem = set(reachable_food)
        tour = []
        cur = start_pos
        while rem:
            if len(rem) == 1:
                nxt = next(iter(rem))
            else:
                candidates = sorted(rem, key=lambda f: dm[(cur, f)])[:min(k, len(rem))]
                best_score = None
                nxt = None
                for c in candidates:
                    score = dm[(cur, c)]
                    if len(rem) > 1:
                        score += min(dm[(c, o)] for o in rem if o != c)
                    if best_score is None or score < best_score:
                        best_score = score
                        nxt = c
            tour.append(nxt)
            rem.remove(nxt)
            cur = nxt
        return tour

    def two_opt_metric(tour):
        """
        2-opt 局部搜索：不断尝试反转子序列，直到无法继续改善或超时。
        2-opt local search: repeatedly reverse sub-sequences until no improvement or time out.
        使用预计算的 dm 距离表，不依赖 BFS 运行时调用。
        Uses pre-computed dm table; no runtime BFS calls.
        """
        n = len(tour)
        if n < 4:
            return tour
        improved = True
        while improved and time.time() < planning_deadline:
            improved = False
            for i in range(n - 1):
                prev_node = start_pos if i == 0 else tour[i - 1]
                node_i = tour[i]
                for j in range(i + 1, n):
                    node_j = tour[j]
                    next_node = tour[j + 1] if j + 1 < n else None
                    old_cost = dm[(prev_node, node_i)] + (dm[(node_j, next_node)] if next_node else 0)
                    new_cost = dm[(prev_node, node_j)] + (dm[(node_i, next_node)] if next_node else 0)
                    if new_cost < old_cost:
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    seed_rng = random.Random(20260317)
    candidate_tours = []

    # ----------------------------------------------------------------
    # 生成初始候选游览序列
    # Generate initial candidate tours
    # ----------------------------------------------------------------
    if len(reachable_food) >= 150:
        # Legacy large-map routine remains as a strong candidate, but we also
        # compare with new seeds and keep the truly best one.
        # 大地图：先用遗留的大地图专用例程，再叠加其他启发式候选
        # Large map: use legacy large-map routine + additional heuristic candidates
        legacy_deadline = min(planning_deadline, start_time + 6.0)
        candidate_tours.append(
            _legacy_large_tour(
                start_pos=start_pos,
                reachable_food=reachable_food,
                dm=dm,
                paths=paths,
                food_set=food_set,
                deadline=legacy_deadline,
            )
        )
        candidate_tours.append(nearest_tour())
        candidate_tours.append(nearest_tour(farthest_first=True))
        candidate_tours.append(nearest_tour(beam_width=5, rng=seed_rng))
    else:
        # 中小地图：多种策略候选 + 随机束搜索
        # Medium/small map: multiple strategy candidates + random beam searches
        candidate_tours = [
            nearest_tour(),
            nearest_tour(farthest_first=True),
            lookahead_tour(),
        ]
        for _ in range(4):
            candidate_tours.append(nearest_tour(beam_width=4, rng=seed_rng))

    # ----------------------------------------------------------------
    # 对所有候选序列执行 2-opt，选出最优
    # Apply 2-opt to all candidates, select the best
    # ----------------------------------------------------------------
    best_tour = []
    best_cost = float('inf')
    for cand in candidate_tours:
        if time.time() >= planning_deadline:
            break
        cand = two_opt_metric(cand[:])
        cand_cost, compact = evaluate_tour(cand)
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_tour = compact

    if not best_tour:
        # Guaranteed fallback: deterministic nearest path.
        # 保底后备：确定性最近邻 / Guaranteed fallback: deterministic nearest
        best_tour = nearest_tour()

    current_tour = best_tour[:]
    current_cost, current_tour = evaluate_tour(current_tour)
    if current_cost < best_cost:
        best_cost = current_cost
        best_tour = current_tour[:]

    # ----------------------------------------------------------------
    # 3-opt 扰动 + 模拟退火式接受策略
    # 3-opt perturbation + simulated annealing acceptance
    # ----------------------------------------------------------------
    # 根据食物数量设置停滞上限（无改善多少轮后终止）
    # Set stagnation limit based on food count (terminate after N non-improving iterations)
    stagnation = 0
    if len(reachable_food) >= 150:
        stagnation_limit = len(reachable_food) * 12
    elif len(reachable_food) >= 90:
        stagnation_limit = len(reachable_food) * 5
    else:
        stagnation_limit = max(80, len(reachable_food) * 2)
    while time.time() < planning_deadline and stagnation < stagnation_limit and len(current_tour) >= 4:
        n = len(current_tour)
        # 随机选三个分割点，做 3-opt 片段重组（或-opt 扰动）
        # Randomly pick 3 split points for 3-opt segment reassembly (or-opt perturbation)
        a, b, c = sorted(seed_rng.sample(range(1, n), 3))
        perturbed = current_tour[:a] + current_tour[b:c] + current_tour[a:b] + current_tour[c:]
        perturbed = two_opt_metric(perturbed)
        perturbed_cost, perturbed_compact = evaluate_tour(perturbed)

        if perturbed_cost < best_cost:
            # 找到全局最优，更新 / New global best found
            best_cost = perturbed_cost
            best_tour = perturbed_compact[:]
            current_tour = perturbed_compact[:]
            stagnation = 0
            continue

        stagnation += 1
        # 以 8% 概率接受更差解（跳出局部最优）/ Accept worse solution with 8% probability (escape local optima)
        if perturbed_cost < current_cost or seed_rng.random() < 0.08:
            current_cost = perturbed_cost
            current_tour = perturbed_compact[:]

    # 将最优游览序列转换为实际动作列表，并截取得分最高前缀
    # Convert best tour to action list, then trim to best-score prefix
    actions = _tour_to_actions(start_pos, best_tour, paths, food_set)
    actions = _best_score_prefix(actions, start_pos, problem.initial_food)

    # ----------------------------------------------------------------
    # 阶段 5：得分感知束搜索（备选规划，若得分更高则替换）
    # Phase 5: Score-aware beam search backup (replace if score is higher)
    # ----------------------------------------------------------------
    # Score-aware backup planner: run multiple short beam restarts and keep
    # the best scoring result.
    if len(reachable_food) <= 260 and time.time() < planning_deadline - 0.05:
        beam_deadline = min(
            planning_deadline,
            time.time() + (2.0 if len(reachable_food) <= 120 else 1.0),
        )
        base_score = _simulate_score(actions, start_pos, problem.initial_food)
        alt_actions = _multi_restart_score_beam_actions(
            start_pos=start_pos,
            reachable_food=reachable_food,
            dm=dm,
            paths=paths,
            seg_foods=seg_foods,
            initial_food=problem.initial_food,
            deadline=beam_deadline,
        )
        alt_score = _simulate_score(alt_actions, start_pos, problem.initial_food)
        if alt_score > base_score:
            actions = alt_actions

    return actions


# ================================================================
# 工具函数：BFS 距离与路径重建
# Utilities: BFS Distance & Path Reconstruction
# ================================================================

def _bfs_dist_parent(start, walls, width, height):
    """
    从 start 出发 BFS，返回 (dist, parent) 两个字典。
    BFS from start; returns (dist, parent) dicts.

    dist[pos] = 从 start 到 pos 的最短步数
    dist[pos] = shortest step count from start to pos

    parent[pos] = BFS 树中 pos 的父节点（用于路径回溯）
    parent[pos] = parent of pos in the BFS tree (used for path reconstruction)
    """
    dist = {start: 0}
    parent = {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        nd = dist[(x, y)] + 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny] and nxt not in dist:
                dist[nxt] = nd
                parent[nxt] = (x, y)
                queue.append(nxt)
    return dist, parent


def _reconstruct_path(parent, start, goal):
    """
    通过 parent 字典从 goal 回溯到 start，返回路径（不含 start，含 goal）。
    Backtrack from goal to start via parent dict; returns path (excluding start, including goal).
    """
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# ================================================================
# 工具函数：MST 代价（用于精确 A* 启发式）
# Utility: MST Cost (used in exact A* heuristic)
# ================================================================

def _mst_cost(remaining, dm, cache):
    """
    用 Prim 算法计算剩余食物节点的最小生成树代价（带缓存）。
    Compute MST cost over remaining food nodes using Prim's algorithm (with cache).

    作为 A* 启发式的组成部分：MST 代价是收集所有剩余食物的可接受下界。
    Used as part of the A* heuristic: MST cost is an admissible lower bound for collecting all remaining food.
    """
    rem = frozenset(remaining)
    if rem in cache:
        return cache[rem]
    if len(rem) <= 1:
        cache[rem] = 0
        return 0

    nodes = list(rem)
    used = {nodes[0]}
    not_used = set(nodes[1:])
    best = {n: dm[(nodes[0], n)] for n in not_used}
    cost = 0
    while not_used:
        nxt = min(not_used, key=lambda n: best[n])
        cost += best[nxt]
        not_used.remove(nxt)
        used.add(nxt)
        for other in not_used:
            d = dm[(nxt, other)]
            if d < best[other]:
                best[other] = d
    cache[rem] = cost
    return cost


# ================================================================
# 精确 A* 求解器（小实例专用）
# Exact A* Solver (small instances only)
# ================================================================

def _solve_small_exact(start_state, reachable_food, walls, width, height, dm,
                       dist_cache, deadline, max_food_for_exact, max_expansions):
    """
    对小实例（food 数量 ≤ max_food_for_exact）运行精确 A*。
    Run exact A* for small instances (food count ≤ max_food_for_exact).

    启发式 / Heuristic:
        h(s) = max(最远食物距离, 最近食物距离 + MST代价)
        h(s) = max(farthest food dist, nearest food dist + MST cost)
    这是一个更紧的可接受且一致的下界。
    This is a tighter admissible and consistent lower bound.

    若超时或超出节点展开上限，返回 None，交由上层贪心策略处理。
    Returns None if deadline or expansion limit is exceeded (falls back to greedy).
    """
    if len(reachable_food) > max_food_for_exact:
        return None

    start_pos, _ = start_state
    # 预构建食物节点间距离表和 MST 缓存
    # Pre-build food-to-food distance table and MST cache
    food_dm = {(a, b): dm[(a, b)] for a in reachable_food for b in reachable_food if a != b}
    mst_cache = {frozenset(): 0}
    pos_bfs_cache = dict(dist_cache)

    def heuristic(state):
        """
        启发函数：max(最远食物BFS距离, 最近食物距离 + MST代价)
        Heuristic: max(farthest food BFS dist, nearest food dist + MST cost)
        """
        pos, rem = state
        if not rem:
            return 0
        pos_dist = pos_bfs_cache.get(pos)
        if pos_dist is None:
            pos_dist, _ = _bfs_dist_parent(pos, walls, width, height)
            pos_bfs_cache[pos] = pos_dist
        nearest = min(pos_dist[f] for f in rem)
        farthest = max(pos_dist[f] for f in rem)
        tree = _mst_cost(rem, food_dm, mst_cache)
        return max(farthest, nearest + tree)

    frontier = util.PriorityQueue()
    g_best = {start_state: 0}
    parent = {}
    parent_action = {}
    frontier.push((start_state, 0), heuristic(start_state))

    expansions = 0
    while (not frontier.isEmpty()
           and time.time() < deadline
           and expansions < max_expansions):
        state, g = frontier.pop()
        # 过滤过期节点 / Skip stale entries
        if g != g_best.get(state):
            continue
        expansions += 1

        # 目标状态：回溯动作序列 / Goal state: backtrack action sequence
        if len(state[1]) == 0:
            actions = []
            cur = state
            while cur != start_state:
                actions.append(parent_action[cur])
                cur = parent[cur]
            actions.reverse()
            return actions

        # 扩展四个方向 / Expand four directions
        pos, rem = state
        x, y = pos
        for action, (dx, dy) in (
            (Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST, (1, 0)),
            (Directions.WEST, (-1, 0)),
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                npos = (nx, ny)
                nrem = rem
                if npos in rem:
                    nrem = frozenset(f for f in rem if f != npos)
                succ = (npos, nrem)
                ng = g + 1
                if ng < g_best.get(succ, float('inf')):
                    g_best[succ] = ng
                    parent[succ] = state
                    parent_action[succ] = action
                    frontier.push((succ, ng), ng + heuristic(succ))

    return None  # 超时或超节点限制，返回 None / Timed out or over limit, return None


# ================================================================
# 游览序列转动作 / Tour to Action Sequence
# ================================================================

def _tour_to_actions(start_pos, tour, paths, food_set):
    """
    将食物节点访问序列（tour）转换为 Pac-Man 的动作列表。
    Convert a food-node visit sequence (tour) into Pac-Man action list.

    跳过已通过路径间接收集的食物节点（避免重复访问）。
    Skip food nodes already collected indirectly along the path (avoid re-visits).
    """
    actions = []
    cur = start_pos
    eaten = {start_pos} if start_pos in food_set else set()

    for target in tour:
        if target in eaten:
            continue
        seg = paths.get((cur, target))
        if seg is None:
            continue
        for step in seg:
            dx = step[0] - cur[0]
            dy = step[1] - cur[1]
            if dx == 1:
                actions.append(Directions.EAST)
            elif dx == -1:
                actions.append(Directions.WEST)
            elif dy == 1:
                actions.append(Directions.NORTH)
            else:
                actions.append(Directions.SOUTH)
            cur = step
            if cur in food_set:
                eaten.add(cur)
    return actions


def _append_step_action(actions, src, dst):
    """
    根据两个相邻位置的坐标差，推断并追加一步方向动作。
    Infer and append one directional action from the coordinate difference between two adjacent positions.
    """
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 1:
        actions.append(Directions.EAST)
    elif dx == -1:
        actions.append(Directions.WEST)
    elif dy == 1:
        actions.append(Directions.NORTH)
    else:
        actions.append(Directions.SOUTH)


# ================================================================
# 快速树覆盖算法（超大/复杂地图专用）
# Fast Tree-Cover Algorithm (very large/complex maps)
# ================================================================

def _fast_tree_cover_actions(start_pos, start_parent, reachable_food):
    """
    基于 BFS 生成树的深度优先遍历，生成覆盖所有食物的快速路径。
    DFS traversal over the BFS spanning tree to generate a quick path covering all food.

    适用场景：地图过大无法在时间内构建全量 BFS 距离表时的高效替代方案。
    Use case: efficient fallback when the map is too large to build a full BFS distance table.

    策略 / Strategy:
        - 标记树中通往食物的所有必要节点
          Mark all tree nodes necessary to reach food
        - DFS 遍历时，指向最深食物的路径作为"终止路径"，不原路返回
          Treat the path to the deepest food as the "end path" (no backtrack)
        - 其余子树正常 DFS + 回溯
          Other subtrees use normal DFS + backtrack
    """
    food_set = set(reachable_food)
    if not food_set:
        return []

    # 从 BFS parent 字典构建子树结构
    # Build child tree from BFS parent dict
    children = {node: [] for node in start_parent}
    for node, parent in start_parent.items():
        if parent is not None:
            children[parent].append(node)

    # 子节点按坐标排序，保证结果确定性
    # Sort children by coordinates for determinism
    for node in children:
        children[node].sort(key=lambda p: (p[0], p[1]))

    needed = {}

    def mark_needed(node):
        """递归标记：节点本身是食物或其子树中含食物，则标记为必要。
        Recursively mark: a node is 'needed' if it is food or has food in its subtree."""
        keep = node in food_set
        for child in children.get(node, []):
            if mark_needed(child):
                keep = True
        needed[node] = keep
        return keep

    if not mark_needed(start_pos):
        return []

    # Leave traversal at a deepest needed node to avoid one final backtrack.
    # 找到最深的必要节点作为终止节点（不回溯）
    # Find the deepest needed node as the termination node (no final backtrack)
    deepest = start_pos
    deepest_depth = 0
    stack = [(start_pos, 0)]
    while stack:
        node, depth = stack.pop()
        if needed.get(node) and depth > deepest_depth:
            deepest = node
            deepest_depth = depth
        for child in children.get(node, []):
            stack.append((child, depth + 1))

    # 标记从 start 到 deepest 的终止路径
    # Mark the "end path" from start to deepest
    next_on_end_path = {}
    cur = deepest
    while start_parent[cur] is not None:
        parent = start_parent[cur]
        next_on_end_path[parent] = cur
        cur = parent

    actions = []

    def dfs(node, on_end_path):
        """
        DFS 遍历：
        - 非终止子树：访问后原路返回
        - 终止子树方向的子节点：不返回（作为终止节点）
        DFS traversal:
        - Non-end-path subtrees: visit and backtrack
        - End-path child: no backtrack (it is the termination direction)
        """
        required_children = [c for c in children.get(node, []) if needed.get(c)]
        if not required_children:
            return

        end_child = next_on_end_path.get(node) if on_end_path else None
        normal_children = [c for c in required_children if c != end_child]
        order = normal_children + ([end_child] if end_child in required_children else [])

        for child in order:
            _append_step_action(actions, node, child)
            child_on_end = on_end_path and child == end_child
            dfs(child, child_on_end)
            if not child_on_end:
                # 非终止子树：回溯到当前节点 / Non-end subtree: backtrack to current node
                _append_step_action(actions, child, node)

    dfs(start_pos, True)
    return actions


# ================================================================
# 得分感知前缀截取 / Score-Aware Prefix Trimming
# ================================================================

def _best_score_prefix(actions, start_pos, initial_food):
    """
    模拟执行动作序列，截取游戏得分最高的前缀。
    Simulate the action sequence and trim to the prefix with the highest game score.

    Pac-Man 得分规则 / Pac-Man scoring rules:
        - 每步 -1 / Each step: -1
        - 吃到食物 +10 / Eat food: +10
        - 清空所有食物额外 +500 / Clear all food: +500 bonus

    目的：剔除后期只消耗步数却不增加净得分的冗余移动。
    Purpose: remove trailing moves that cost steps without increasing net score.
    """
    if not actions:
        return actions

    remaining_food = set(initial_food)
    score = 0
    best_score = 0
    best_len = 0

    x, y = start_pos
    # 起始位置若有食物，视为已收集 / Collect food at start position if present
    if (x, y) in remaining_food:
        remaining_food.remove((x, y))
        score += 10
        best_score = score

    for i, action in enumerate(actions, 1):
        if action == Directions.NORTH:
            y += 1
        elif action == Directions.SOUTH:
            y -= 1
        elif action == Directions.EAST:
            x += 1
        else:
            x -= 1

        score -= 1  # 每步代价 / Step cost
        pos = (x, y)
        if pos in remaining_food:
            remaining_food.remove(pos)
            score += 10
            if not remaining_food:
                score += 500  # 清空奖励 / All-clear bonus

        # 记录得分最高时的动作序列长度 / Track the action length at the highest score
        if score > best_score:
            best_score = score
            best_len = i

    return actions[:best_len]


# ================================================================
# 得分模拟（用于束搜索比较）
# Score Simulation (for beam search comparison)
# ================================================================

def _simulate_score(actions, start_pos, initial_food):
    """
    完整模拟动作序列，返回最终游戏得分。
    Fully simulate the action sequence and return the final game score.

    用于比较不同规划方案的得分优劣。
    Used to compare the score quality of different planning approaches.
    """
    remaining_food = set(initial_food)
    score = 0
    x, y = start_pos

    if (x, y) in remaining_food:
        remaining_food.remove((x, y))
        score += 10

    for action in actions:
        if action == Directions.NORTH:
            y += 1
        elif action == Directions.SOUTH:
            y -= 1
        elif action == Directions.EAST:
            x += 1
        else:
            x -= 1

        score -= 1
        pos = (x, y)
        if pos in remaining_food:
            remaining_food.remove(pos)
            score += 10
            if not remaining_food:
                score += 500
    return score


# ================================================================
# 得分感知束搜索（单次运行）
# Score-Aware Beam Search (single run)
# ================================================================

def _score_beam_actions(start_pos, reachable_food, dm, paths, seg_foods, initial_food,
                        deadline, rng=None, profile=0):
    """
    以实际游戏得分为目标的束搜索规划器（单次运行）。
    Score-aware beam search planner (single run).

    在每步维护宽度为 width 的候选集（beam），
    按预期得分增量贪心选择下一个目标食物。
    Maintains a beam of width `width` candidates at each step,
    greedily selecting the next food target by expected score gain.

    三种 profile：
        0 = 标准贪心 (standard greedy)
        1 = 更深搜索，较窄束宽 (deeper search, narrower beam)
        2 = 更浅搜索，加随机扰动 (shallower search, more jitter)
    Three profiles:
        0 = standard greedy
        1 = deeper search, narrower beam
        2 = shallower search, more jitter
    """
    if rng is None:
        rng = random.Random(0)

    rem = frozenset(f for f in reachable_food if f != start_pos)
    if not rem:
        return []

    tour = []
    cur = start_pos

    while rem and time.time() < deadline:
        # 根据剩余食物数量动态调整搜索参数
        # Dynamically adjust search parameters based on remaining food count
        n_rem = len(rem)
        if n_rem <= 20:
            depth = 5
            width = 48
            k_near = 12
        elif n_rem <= 80:
            depth = 4
            width = 28
            k_near = 10
        else:
            depth = 3
            width = 18
            k_near = 8

        # 根据 profile 微调参数 / Fine-tune parameters per profile
        if profile == 1:
            depth = min(6, depth + 1)
            width = max(12, width - 8)
            k_near = min(len(rem), k_near + 2)
            jitter = 1.2
        elif profile == 2:
            depth = max(2, depth - 1)
            width = max(10, width - 10)
            k_near = min(len(rem), max(6, k_near - 1))
            jitter = 2.2
        else:
            jitter = 0.0

        # 初始化束：(当前位置, 剩余食物, 累积得分增量, 首个选择的食物)
        # Init beam: (current pos, remaining food, accumulated gain, first food chosen)
        beam = [(cur, rem, 0, None)]
        best_gain = 0
        best_first = None

        for _ in range(depth):
            if time.time() >= deadline:
                break
            expanded = []
            for pos, rset, gain, first in beam:
                if not rset:
                    if gain > best_gain and first is not None:
                        best_gain = gain
                        best_first = first
                    continue

                # 对候选食物排序（含随机扰动 jitter）
                # Sort candidate food (with optional jitter for diversity)
                if jitter > 0:
                    ordered = sorted(
                        rset,
                        key=lambda f: (dm[(pos, f)] + jitter * rng.random(), f[0], f[1]),
                    )
                else:
                    ordered = sorted(
                        rset,
                        key=lambda f: (dm[(pos, f)], f[0], f[1]),
                    )
                candidates = ordered[:min(k_near, len(ordered))]
                for target in candidates:
                    # 计算本段路径顺带收集的食物得分
                    # Compute score gain from food collected along this path segment
                    seg = seg_foods[(pos, target)] & rset
                    if not seg:
                        seg = frozenset([target])
                    ngain = gain + 10 * len(seg) - dm[(pos, target)]
                    nr = rset - seg
                    if not nr:
                        ngain += 500  # 清空奖励 / All-clear bonus
                    nfirst = target if first is None else first
                    if ngain > best_gain and nfirst is not None:
                        best_gain = ngain
                        best_first = nfirst
                    expanded.append((target, nr, ngain, nfirst))

            if not expanded:
                break
            # 束排序：按得分降序，保留前 width 个
            # Sort beam by score descending, keep top width
            if jitter > 0:
                expanded.sort(key=lambda item: (item[2], rng.random() * 1e-6), reverse=True)
            else:
                expanded.sort(key=lambda item: item[2], reverse=True)
            beam = expanded[:width]

        if best_first is None:
            break

        tour.append(best_first)
        seg = seg_foods[(cur, best_first)] & rem
        if not seg:
            seg = frozenset([best_first]) if best_first in rem else frozenset()
        rem = rem - seg
        cur = best_first

        # Stop if further expansion cannot increase score from this prefix.
        # 若无法再提升得分，提前终止 / Early stop if no further score gain is possible
        if best_gain <= 0 and len(tour) >= 2:
            break

    actions = _tour_to_actions(start_pos, tour, paths, set(reachable_food))
    return _best_score_prefix(actions, start_pos, initial_food)


# ================================================================
# 多次重启束搜索（选取最高得分结果）
# Multi-Restart Beam Search (keep best-score result)
# ================================================================

def _multi_restart_score_beam_actions(start_pos, reachable_food, dm, paths, seg_foods,
                                      initial_food, deadline):
    """
    多次重启束搜索：使用不同 profile 和随机种子运行多次，取得分最高的结果。
    Multi-restart beam search: run multiple times with different profiles and seeds,
    return the result with the highest game score.

    时间预算按剩余次数均分，确保每次运行都有足够时间。
    Time budget is split evenly across remaining runs.
    """
    if time.time() >= deadline:
        return []

    n_food = len(reachable_food)
    # 根据食物数量决定运行次数和 profile 组合
    # Determine number of runs and profile combination based on food count
    if n_food <= 80:
        profiles = [0, 1, 2, 1]
    elif n_food <= 160:
        profiles = [0, 1, 2]
    else:
        profiles = [0, 1]

    best_actions = []
    best_score = float("-inf")

    for idx, profile in enumerate(profiles):
        now = time.time()
        if now >= deadline:
            break

        # 均分剩余时间 / Evenly split remaining time
        runs_left = len(profiles) - idx
        slot = max(0.08, (deadline - now) / runs_left)
        run_deadline = min(deadline, now + slot)

        rng = random.Random(20260317 + idx * 9973 + profile * 131)
        cand_actions = _score_beam_actions(
            start_pos=start_pos,
            reachable_food=reachable_food,
            dm=dm,
            paths=paths,
            seg_foods=seg_foods,
            initial_food=initial_food,
            deadline=run_deadline,
            rng=rng,
            profile=profile,
        )
        cand_score = _simulate_score(cand_actions, start_pos, initial_food)
        if cand_score > best_score:
            best_score = cand_score
            best_actions = cand_actions

    return best_actions


# ================================================================
# 大地图遗留例程（含 2-opt + 模拟退火）
# Legacy Large-Map Routine (with 2-opt + simulated annealing)
# ================================================================

def _legacy_large_tour(start_pos, reachable_food, dm, paths, food_set, deadline):
    """
    大地图专用游览序列生成器：最近邻初始化 + 2-opt + 模拟退火扰动。
    Large-map tour generator: nearest-neighbour init + 2-opt + simulated annealing perturbation.

    与主流程中的策略类似，但作为大地图时的主要候选之一保留。
    Similar to the main pipeline strategy, kept as a primary candidate for large maps.
    """
    def evaluate_cost(tour):
        """计算游览序列的总路径步数代价 / Compute total step cost of the tour"""
        eaten = {start_pos} if start_pos in food_set else set()
        cur = start_pos
        cost = 0
        for nxt in tour:
            if nxt in eaten:
                continue
            seg = paths[(cur, nxt)]
            cost += len(seg)
            for p in seg:
                if p in food_set:
                    eaten.add(p)
            cur = nxt
        return cost

    # 最近邻贪心初始化 / Nearest-neighbour greedy initialisation
    order = []
    rem = set(reachable_food)
    cur = start_pos
    while rem:
        nxt = min(rem, key=lambda f: dm[(cur, f)])
        order.append(nxt)
        rem.remove(nxt)
        cur = nxt

    def two_opt(tour):
        """2-opt 局部搜索改善游览序列 / 2-opt local search to improve tour"""
        n = len(tour)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                prev_node = start_pos if i == 0 else tour[i - 1]
                node_i = tour[i]
                for j in range(i + 1, n):
                    node_j = tour[j]
                    next_node = tour[j + 1] if j + 1 < n else None
                    d1 = dm[(prev_node, node_i)]
                    d2 = dm[(node_j, next_node)] if next_node else 0
                    d3 = dm[(prev_node, node_j)]
                    d4 = dm[(node_i, next_node)] if next_node else 0
                    if d3 + d4 < d1 + d2:
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        improved = True
        return tour

    order = two_opt(order)
    best_tour = order[:]
    best_cost = evaluate_cost(best_tour)
    curr_tour = best_tour[:]
    rng = random.Random(42)

    # 模拟退火扰动循环 / Simulated annealing perturbation loop
    while time.time() < deadline:
        n = len(curr_tour)
        if n < 4:
            break
        # 3-opt 片段重组 / 3-opt segment reassembly
        p1, p2, p3 = sorted(rng.sample(range(1, n), 3))
        new_tour = curr_tour[:p1] + curr_tour[p2:p3] + curr_tour[p1:p2] + curr_tour[p3:]
        new_tour = two_opt(new_tour)
        tc = evaluate_cost(new_tour)
        if tc < best_cost:
            best_cost = tc
            best_tour = new_tour[:]
            curr_tour = new_tour[:]
        elif rng.random() < 0.05:
            # 以 5% 概率接受更差解，跳出局部最优 / Accept worse with 5% probability
            curr_tour = new_tour[:]

    return best_tour
