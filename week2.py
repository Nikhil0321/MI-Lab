"""
You can create any other helper funtions.
Do not modify the given functions
"""


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    path= []
    front = [start_point]
    frontier = [[0+heuristic[start_point], front]]
    while len(frontier) > 0:
        curr_cost, curr_path = frontier.pop(0)
        n = curr_path[-1]
        curr_cost -= heuristic[n]
        if n in goals:
            return curr_path
        path.append(n)
        children = [i for i in range(len(cost[0]))
                    if cost[n][i] not in [0, -1]]
        for i in children:
            new_curr_path = curr_path + [i]
            new_path_cost = curr_cost + cost[n][i] + heuristic[i]
            if i not in path and new_curr_path not in [i[1] for i in frontier]:
                frontier.append((new_path_cost, new_curr_path))
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in frontier]:
                index = search_q(frontier, new_curr_path)
                frontier[index][0] = min(frontier[index][0], new_path_cost)
                frontier = sorted(frontier, key=lambda x: (x[0], x[1]))
    return path
   


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    frontier = []
    n = len(cost)
    frontier.append(start_point)
    while len(frontier) != 0:
        curr = frontier.pop()  
        path.append(curr)
        if curr in goals:
                return path
        for i in range(n - 1, 0, -1):
                if cost[curr][i] != -1 and cost[curr][i] != 0 and (i not in path):
                        frontier.append(i)
    return path
