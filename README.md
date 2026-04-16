# FIT5047 Pacman AI Project

This repository documents a Pacman AI project completed for FIT5047. The project focuses on two core tasks:

- Q1(c): collecting all reachable food dots efficiently in maze layouts
- Q2: building a stronger Pacman agent for ghost-filled environments using adversarial search

The work combines classical AI search, heuristic design, caching, and maze-aware preprocessing to improve both solution quality and runtime performance.

## Project Summary

The project explores how search-based methods can be adapted to different Pacman problem settings.

For food collection, the solver uses a hybrid strategy: it attempts exact A* search when the maze is small enough, and switches to approximate route construction when exact planning becomes too expensive. This makes the agent more practical under strict time limits.

For the adversarial Pacman task, the agent uses iterative deepening alpha-beta search supported by BFS-based topology preprocessing, cached route information, repeated-state reuse, and an adaptive heuristic evaluation function.

## Key Ideas

### Q1(c): Hybrid Food Search

- Represents each state as Pacman's position plus the set of remaining food dots
- Uses BFS preprocessing to identify reachable food and reuse shortest-path information
- Runs exact A* on smaller instances
- Uses MST-based lower bounds to strengthen the A* heuristic
- Falls back to approximate planning on larger maps
- Refines candidate tours with local search and score-aware improvement

### Q2: Alpha-Beta Pacman Agent

- Uses iterative deepening alpha-beta search for time-aware decision making
- Preprocesses the maze with BFS to capture topology and route distances
- Reuses information across repeated states with a transposition-table-like cache
- Applies adaptive heuristic weighting based on food, ghosts, capsules, and game context
- Includes emergency fallback behaviour when decision time is nearly exhausted

## Reported Results

### Q1(c)

The hybrid food-search solver was reported to:

- win on small and large reachable layouts
- stay effective when exact A* becomes too slow
- handle unwinnable layouts more sensibly by filtering unreachable food

Representative reported outcomes include:

- `q1c_tinySearch`: win, score 573
- `q1c_boxSearch`: win, score 740
- `q1c_openSearch`: win, score 1301
- `q1c_mediumSearch`: win, score 1426
- `q1c_bigSearch`: win, score 2434

### Q2

The final Q2 agent was reported to achieve:

- average score: `2616.6`
- wins: `34/35`

Family-level results from the report include strong performance on `contestClassic`, `mediumClassic2`, `originalClassic`, and `trickyClassic`.

## Repository Contents

This repository snapshot currently contains report-related materials, including LaTeX sources under:

- [`output/latex/FIT5047-36547719-Ass2-natural-version.tex`](/Users/houjiakuan/Documents/New%20project/output/latex/FIT5047-36547719-Ass2-natural-version.tex)
- [`output/latex/FIT5047-36763624-2026S1-Ass2.tex`](/Users/houjiakuan/Documents/New%20project/output/latex/FIT5047-36763624-2026S1-Ass2.tex)

If Pacman source code, agents, or evaluation scripts are added later, this README can be extended with setup and usage instructions.

## Technical Focus

- A* search
- breadth-first search (BFS)
- minimum spanning tree (MST) heuristic design
- alpha-beta pruning
- iterative deepening
- state caching
- adaptive evaluation functions

## Author

Jia Kuan Hou  
Monash University  
FIT5047
