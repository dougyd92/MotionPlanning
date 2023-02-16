from ast import literal_eval
from configparser import ConfigParser
from enum import IntEnum
from queue import PriorityQueue
from typing import Any, Callable

from matplotlib import color_sequences
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import TypeAlias

Coordinate: TypeAlias = tuple[int, int]
State: TypeAlias = tuple[Coordinate, ...]
PrioritizedState: TypeAlias = tuple[int, State]


class NodeState(IntEnum):
    EMPTY = 0
    START = 1
    GOAL = 2
    REACHED = 3
    FRONTIER = 4
    OBSTACLE = 5
    IMPASSABLE = 6


class MultiAgentAStar:
    """
    Used to perform A* path search for multiple agents and
    multipe objectives, in an obstacle-filled grid environment.
    """

    def __init__(
        self,
        height: int,
        width: int,
        starts: State,
        goals: State,
        obstacles: list[Coordinate],
        obstacle_cost: int,
        barriers: list[Coordinate],
        heuristic: Callable[[State, State], int],
    ) -> None:
        self.height = height
        self.width = width
        self.starts = starts
        self.goals = goals
        self.obstacles = obstacles
        self.obstacle_cost = obstacle_cost
        self.barriers = barriers
        self.heuristic = heuristic

        self.frontier: PriorityQueue[PrioritizedState] = PriorityQueue()
        self.reached_from: dict[State, State] = {}
        self.cost_to_reach: dict[State, int] = {}
        self.history: list[np.ndarray] = []
        self.update_history()
        self.iterations = 0
        self.max_queue_size = 0

    def get_next_possible_states(self, current_state: State) -> list[State]:
        next_possible_states: list[State] = []
        for i, agent in enumerate(current_state):
            (x, y) = agent
            neighbors = []
            if x > 0:
                neighbors.append((x - 1, y))
            if y > 0:
                neighbors.append((x, y - 1))
            if x < self.width - 1:
                neighbors.append((x + 1, y))
            if y < self.height - 1:
                neighbors.append((x, y + 1))
            # remove moves that would run into a barrier or would collide with another agent
            possible_moves = [
                node
                for node in neighbors
                if (node not in self.barriers and node not in current_state)
            ]
            next_possible_states += [
                tuple(list(current_state[0:i]) + [move] + list(current_state[i + 1 :]))
                for move in possible_moves
            ]
        return next_possible_states

    def get_cost(self, current_state: State, next_state: State) -> int:
        # Find the new location being moved into
        for coord in next_state:
            if coord not in current_state:
                if coord in self.obstacles:
                    return self.obstacle_cost
                else:
                    return 1
        raise Exception("Couldn't find the cost for the next state.")

    def update_history(self) -> None:
        grid = np.zeros((self.height, self.width))
        for goal in self.goals:
            grid[goal[1]][goal[0]] = NodeState.GOAL
        for start in self.starts:
            grid[start[1]][start[0]] = NodeState.START
        for x, y in self.obstacles:
            grid[y][x] = NodeState.OBSTACLE
        for x, y in self.barriers:
            grid[y][x] = NodeState.IMPASSABLE
        for state in self.reached_from:
            for x, y in state:
                grid[y][x] = NodeState.REACHED
        for _priority, state in self.frontier.queue:
            for x, y in state:
                grid[y][x] = NodeState.FRONTIER
        self.history.append(grid)

    def perform_search(self) -> bool:
        all_goals_reached = False
        self.frontier.put((0, self.starts))
        self.reached_from[self.starts] = self.starts
        self.cost_to_reach[self.starts] = 0

        while not self.frontier.empty():
            self.iterations += 1
            self.max_queue_size = max(self.max_queue_size, self.frontier.qsize())
            (_priority, current_state) = self.frontier.get()
            if current_state == self.goals:
                all_goals_reached = True
                break
            for next_state in self.get_next_possible_states(current_state):
                updated_cost = self.cost_to_reach[current_state] + self.get_cost(
                    current_state, next_state
                )
                if (
                    next_state not in self.reached_from
                    or updated_cost < self.cost_to_reach[next_state]
                ):
                    self.cost_to_reach[next_state] = updated_cost
                    priority = updated_cost + self.heuristic(next_state, self.goals)
                    self.frontier.put((priority, next_state))
                    self.reached_from[next_state] = current_state
                    self.update_history()
        self.update_history()
        return all_goals_reached

    def reconstruct_best_path(self) -> list[State]:
        current_state = self.goals
        path = []
        while current_state != self.starts:
            path.append(current_state)
            current_state = self.reached_from[current_state]
        path.append(self.starts)
        path.reverse()
        return path


def l1_norm(starts: State, ends: State) -> int:
    """Sum of the Manhattan distance between the start points and respective end points"""
    total = 0
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        total += abs(start[0] - end[0]) + abs(start[1] - end[1])
    return total


def animation_step(
    i: int, im: Any, history: list[np.ndarray], path: list[State], arrows: Any
) -> list[Any]:
    if i == 0:
        # Reset arrows so they won't appear when the animation repeats
        for a in arrows:
            a.remove()
        arrows.clear()

    if i % 1000 == 0:
        print(f"Animating frame {i} of {len(history)}")

    if i < len(history):
        # Draw each iteration of the search algorithm
        im.set_array(history[i])
    else:
        # Animate the path that was found
        j = i - len(history)
        prev_state = path[j]
        next_state = path[j + 1]
        for k in range(len(prev_state)):
            (from_x, from_y) = prev_state[k]
            (to_x, to_y) = next_state[k]
            arrow = im.axes.arrow(
                from_x + k * 0.1,
                from_y + k * 0.1,
                (to_x - from_x) * 0.9,
                (to_y - from_y) * 0.9,
                head_width=0.25,
                length_includes_head=True,
                facecolor=color_sequences["tab10"][k],
                zorder=2 + k,
            )
            arrows.append(arrow)
    return [im]


def animate_search(
    grid_height: int,
    grid_width: int,
    starts: list[Coordinate],
    goals: list[Coordinate],
    obstacles: list[Coordinate],
    history: list[np.ndarray],
    path: list[State],
) -> None:
    fig, ax = plt.subplots()
    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax.set_xticks(np.arange(-0.5, grid_width))
    ax.set_xticklabels(np.arange(0, grid_width + 1))
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels()[-1], visible=False)
    plt.xticks(ha="left")
    ax.set_yticks(np.arange(-0.5, grid_height))
    ax.set_yticklabels(np.arange(0, grid_height + 1))
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    plt.yticks(va="top")

    cmap = ListedColormap(
        ["white", "darkblue", "green", "blue", "skyblue", "gray", "black"]
    )
    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=max(NodeState))

    for i, start in enumerate(starts):
        ax.annotate(f"START {i}", start, ha="center", va="center")
    for i, goal in enumerate(goals):
        ax.annotate(f"GOAL {i}", goal, ha="center", va="center")
    for obstacle in obstacles:
        ax.annotate(chr(174), obstacle, fontsize="xx-large", ha="center", va="center")
    arrows: list[Any] = []

    _anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(im, history, path, arrows),
        frames=len(history) + len(path) - 1,
        interval=20,
        repeat_delay=10000,
    )

    plt.show()


def main() -> None:
    config = ConfigParser()
    config.read("multi_agent.ini")
    grid_width = config["Grid"].getint("width")
    grid_height = config["Grid"].getint("height")
    starts = literal_eval(config["Agents"]["starts"])
    goals = literal_eval(config["Goal"]["goals"])
    obstacle_cost = config["Obstacles"].getint("movement_cost")
    obstacles = literal_eval(config["Obstacles"]["obstacles"])
    barriers = literal_eval(config["Obstacles"]["barriers"])

    heuristic = l1_norm

    search = MultiAgentAStar(
        grid_height,
        grid_width,
        starts,
        goals,
        obstacles,
        obstacle_cost,
        barriers,
        heuristic,
    )
    goal_found = search.perform_search()
    if goal_found:
        path = search.reconstruct_best_path()
        print(f"Total cost to reach goal: {search.cost_to_reach[goals]}")
        print(f"Iterations taken: {search.iterations}")
        print(f"Peak queue size: {search.max_queue_size}")
    else:
        path = []
        print("No path found. Goal was unreachable.")
    print("search.history", len(search.history))
    print("path", len(path))
    animate_search(
        grid_height,
        grid_width,
        starts,
        goals,
        obstacles,
        search.history,
        path,
    )


if __name__ == "__main__":
    main()
