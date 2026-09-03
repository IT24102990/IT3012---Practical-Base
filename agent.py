from collections import deque
import heapq
import math

class SearchAgent:
    """Goal-based agent using BFS, DFS and UCS search."""

    def __init__(self):
        # Stores the actions that the agent has already planned
        self.plan = []

        # Select which search algorithm to use
        self.active_algo = 'BFS'

        # Starting position from Lab 02
        self.position = (0, 0)

        # Starting direction from Lab 02
        self.facing = 'Right'

    # Get valid neighboring cells
    def get_neighbors(self, state, grid_size, walls):
        x, y = state

        possible_moves = [
            ((x, y - 1), 'Up'),
            ((x, y + 1), 'Down'),
            ((x - 1, y), 'Left'),
            ((x + 1, y), 'Right')
        ]

        neighbors = []

        width, height = grid_size

        for next_state, action in possible_moves:

            nx, ny = next_state

            # Check grid boundaries
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Check walls
            if next_state in walls:
                continue

            neighbors.append((next_state, action))

        return neighbors

    # BFS - Breadth First Search
    def bfs_search(self, start, goals, grid_size, walls):

        queue = deque()

        # Store state and path
        queue.append((start, []))

        # Reached set prevents revisiting states
        reached = {start}

        while queue:

            state, path = queue.popleft()

            # Goal test
            if state in goals:
                return path

            for next_state, action in self.get_neighbors(
                    state, grid_size, walls):

                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    queue.append((next_state, new_path))

        return []

    # DFS - Depth First Search
    def dfs_search(self, start, goals, grid_size, walls):

        stack = []

        # Store state and path
        stack.append((start, []))

        # Reached set prevents infinite loops
        reached = {start}

        while stack:

            state, path = stack.pop()

            # Goal test
            if state in goals:
                return path

            for next_state, action in self.get_neighbors(
                    state, grid_size, walls):

                if next_state not in reached:

                    reached.add(next_state)

                    new_path = path + [action]

                    stack.append((next_state, new_path))

        return []

    # UCS - Uniform Cost Search
    def ucs_search(self, start, goals, grid_size, walls):

        priority_queue = []

        # Each movement has cost 1
        counter = 0

        # (total_cost, counter, state, path)
        heapq.heappush(
            priority_queue,
            (0, counter, start, [])
        )

        reached = {}

        # Cost of reaching the start state
        reached[start] = 0

        while priority_queue:

            cost, _, state, path = heapq.heappop(priority_queue)

            # Goal test
            if state in goals:
                return path

            for next_state, action in self.get_neighbors(
                    state, grid_size, walls):

                new_cost = cost + 1

                # Add state if it has not been reached
                # or if a cheaper path has been found
                if (next_state not in reached or
                        new_cost < reached[next_state]):

                    reached[next_state] = new_cost

                    counter += 1

                    new_path = path + [action]

                    heapq.heappush(
                        priority_queue,
                        (
                            new_cost,
                            counter,
                            next_state,
                            new_path
                        )
                    )

        return []

    def manhattan_distance(self, pos, goal):
        """
        Calculate Manhattan Distance between two grid positions.
        h(n) = |x1 - x2| + |y1 - y2|
        """

        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    
    def euclidean_distance(self, pos, goal):
        """
        Calculate Euclidean Distance between two grid positions.
        h(n) = sqrt((x1-x2)^2 + (y1-y2)^2)
        """

        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    def astar_search(self,start_pos,goal_pos,walls,grid_size,heuristic_type='manhattan'):

        priority_queue = []

        reached_states = set()

        # g(n) = 0 at the starting position
        g_cost = 0

        # Calculate h(n)
        if heuristic_type == 'euclidean':
            h_cost = self.euclidean_distance(
                start_pos,
                goal_pos
            )
        else:
            h_cost = self.manhattan_distance(
                start_pos,
                goal_pos
            )

        # f(n) = g(n) + h(n)
        f_cost = g_cost + h_cost

        # Tuple:
        # (f_cost, g_cost, current_pos, path_taken)
        heapq.heappush(
            priority_queue,
            (
                f_cost,
                g_cost,
                start_pos,
                []
            )
        )

        while priority_queue:

            f_cost, g_cost, current_pos, path_taken = \
                heapq.heappop(priority_queue)

            # Goal test
            if current_pos == goal_pos:

                return self.direction_path_to_actions(
                    path_taken
                )

            # Skip already explored states
            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            # Expand neighboring cells
            for next_pos, direction in self.get_neighbors(
                    current_pos,
                    grid_size,
                    walls):

                if next_pos in reached_states:
                    continue

                # Every movement has cost 1
                new_g_cost = g_cost + 1

                # Calculate heuristic
                if heuristic_type == 'euclidean':

                    new_h_cost = self.euclidean_distance(
                        next_pos,
                        goal_pos
                    )

                else:

                    new_h_cost = self.manhattan_distance(
                        next_pos,
                        goal_pos
                    )

                # f(n) = g(n) + h(n)
                new_f_cost = new_g_cost + new_h_cost

                new_path = path_taken + [direction]

                heapq.heappush(
                    priority_queue,
                    (
                        new_f_cost,
                        new_g_cost,
                        next_pos,
                        new_path
                    )
                )

        # No path found
        return []

    def direction_path_to_actions(self, direction_path):
        """
        Convert grid directions such as Up/Down/Left/Right
        into the environment actions:
        TurnLeft, TurnRight, MoveForward.
        """

        actions = []

        current_facing = self.facing

        directions = ['Up', 'Right', 'Down', 'Left']

        for direction in direction_path:

            current_index = directions.index(current_facing)
            target_index = directions.index(direction)

            # Calculate clockwise difference
            difference = (target_index - current_index) % 4

            if difference == 1:
                actions.append('TurnRight')

            elif difference == 2:
                actions.append('TurnRight')
                actions.append('TurnRight')

            elif difference == 3:
                actions.append('TurnLeft')

            # Move forward after facing the correct direction
            actions.append('MoveForward')

            # Update our local facing direction
            current_facing = direction

        return actions

    

    # Execute the search plan one action at a time
    def sense_and_act(self, percept: dict) -> str:

        # 1. If a plan already exists, execute it
        if self.plan:
            action = self.plan.pop(0)

            self.update_internal_state(action)

            return action

        # 2. Get information from the percept
        grid_size = percept['grid_size']

        walls = set(percept['walls'])

        all_food = set(percept['all_food'])

        # 3. If there is no food, there is no goal
        if not all_food:
            return 'MoveForward'

        # 4. Search for a path to food
        start = self.position

        if self.active_algo == 'BFS':

            direction_path = self.bfs_search(
                start,
                all_food,
                grid_size,
                walls
            )

        elif self.active_algo == 'DFS':

            direction_path = self.dfs_search(
                start,
                all_food,
                grid_size,
                walls
            )

        elif self.active_algo == 'UCS':

            direction_path = self.ucs_search(
                start,
                all_food,
                grid_size,
                walls
            )

        elif self.active_algo == 'AStar':

            remaining_food = percept['all_food']

            if not remaining_food:
                return 'MoveForward'

            # Find closest food using Manhattan distance
            goal_pos = min(
                remaining_food,
                key=lambda food:
                    self.manhattan_distance(
                        self.position,
                        food
                    )
            )

            self.plan = self.astar_search(
                self.position,
                goal_pos,
                set(percept['walls']),
                percept['grid_size'],
                heuristic_type='manhattan'
            )

        else:

            # Default to BFS
            direction_path = self.bfs_search(
                start,
                all_food,
                grid_size,
                walls
            )

        # 5. Convert search directions into environment
        #    actions
        self.plan = self.direction_path_to_actions(direction_path)

        # 6. If no path exists
        if not self.plan:
            return 'TurnLeft'

        # 7. Execute first action from the plan
        action = self.plan.pop(0)

        self.update_internal_state(action)

        return action

        
    # Update internal position and facing direction
    def update_internal_state(self, action):

        directions = ['Up', 'Right', 'Down', 'Left']

        if action == 'TurnLeft':

            index = directions.index(self.facing)

            self.facing = directions[(index - 1) % 4]

        elif action == 'TurnRight':

            index = directions.index(self.facing)

            self.facing = directions[(index + 1) % 4]

        elif action == 'MoveForward':

            x, y = self.position

            if self.facing == 'Up':
                self.position = (x, y - 1)

            elif self.facing == 'Down':
                self.position = (x, y + 1)

            elif self.facing == 'Left':
                self.position = (x - 1, y)

            elif self.facing == 'Right':
                self.position = (x + 1, y)

if __name__ == "__main__":

    agent = SearchAgent()

    start = (0, 0)
    goal = (3, 4)

    print("Manhattan Distance:",
        agent.manhattan_distance(start, goal))

    print("Euclidean Distance:",
        agent.euclidean_distance(start, goal))