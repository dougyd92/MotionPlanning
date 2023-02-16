from ast import literal_eval
from configparser import ConfigParser
from enum import IntEnum
from queue import PriorityQueue
from typing import Any

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import TypeAlias


class NodeState(IntEnum):
    EMPTY = 0
    START = 1
    GOAL = 2
    REACHED = 3
    FRONTIER = 4
    OBSTACLE = 5
    IMPASSABLE = 6


Coordinate: TypeAlias = tuple[int, int]
PrioritizedCoord: TypeAlias = tuple[int, Coordinate]


class SingleAgentAStar:
    """
    Not actually A-Star yet.
    Started with Dijkstra's to get the main program structure and animation.
    """

    def __init__(
        self,
        height: int,
        width: int,
        start: Coordinate,
        goal: Coordinate,
        obstacles: list[Coordinate],
        obstacle_cost: int,
        barriers: list[Coordinate],
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.obstacle_cost = obstacle_cost
        self.barriers = barriers

        self.frontier: PriorityQueue[PrioritizedCoord] = PriorityQueue()
        self.reached_from: dict[Coordinate, Coordinate] = {}
        self.cost_to_reach: dict[Coordinate, int] = {}
        self.history: list[np.ndarray] = []
        self.update_history()

    def get_neighbors(self, coordinate: Coordinate) -> list[Coordinate]:
        (x, y) = coordinate
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if x < self.width - 1:
            neighbors.append((x + 1, y))
        if y < self.height - 1:
            neighbors.append((x, y + 1))
        return [node for node in neighbors if node not in self.barriers]

    def get_cost(self, coordinate: Coordinate) -> int:
        if coordinate in self.obstacles:
            return self.obstacle_cost
        else:
            return 1

    def update_history(self) -> None:
        grid = np.zeros((self.height, self.width))
        grid[self.goal[1]][self.goal[0]] = NodeState.GOAL
        grid[self.start[1]][self.start[0]] = NodeState.START
        for x, y in self.obstacles:
            grid[y][x] = NodeState.OBSTACLE
        for x, y in self.barriers:
            grid[y][x] = NodeState.IMPASSABLE
        for x, y in self.reached_from:
            grid[y][x] = NodeState.REACHED
        for _, (x, y) in self.frontier.queue:
            grid[y][x] = NodeState.FRONTIER
        self.history.append(grid)

    def perform_search(self) -> None:
        self.frontier.put((0, self.start))
        self.reached_from[self.start] = self.start
        self.cost_to_reach[self.start] = 0

        while not self.frontier.empty():
            (_, current_node) = self.frontier.get()
            if current_node == self.goal:
                break
            for next_node in self.get_neighbors(current_node):
                updated_cost = self.cost_to_reach[current_node] + self.get_cost(
                    next_node
                )
                if (
                    next_node not in self.reached_from
                    or updated_cost < self.cost_to_reach[next_node]
                ):
                    self.cost_to_reach[next_node] = updated_cost
                    self.frontier.put((updated_cost, next_node))
                    self.reached_from[next_node] = current_node
                    self.update_history()
        self.update_history()

    def reconstruct_best_path(self) -> list[Coordinate]:
        current_node = self.goal
        path = []
        while current_node != self.start:
            path.append(current_node)
            current_node = self.reached_from[current_node]
        path.append(self.start)
        path.reverse()
        return path


def animation_step(
    i: int, im: Any, history: list[np.ndarray], path: list[Coordinate]
) -> list[Any]:
    if i < len(history):
        # Draw each iteration of the search algorithm
        im.set_array(history[i])
    else:
        # Animate the path that was found
        j = i - len(history)
        (from_x, from_y) = path[j]
        (to_x, to_y) = path[j + 1]
        im.axes.arrow(
            from_x,
            from_y,
            (to_x - from_x) * 0.8,
            (to_y - from_y) * 0.8,
            head_width=0.25,
            length_includes_head=True,
            zorder=2,
        )
    return [im]


def animate_search(
    grid_height: int,
    grid_width: int,
    start: Coordinate,
    goal: Coordinate,
    obstacles: list[Coordinate],
    history: list[np.ndarray],
    path: list[Coordinate],
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

    ax.annotate("START", start, ha="center", va="center")
    ax.annotate("GOAL", goal, ha="center", va="center")
    for obstacle in obstacles:
        ax.annotate(chr(174), obstacle, fontsize="xx-large", ha="center", va="center")

    _anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(im, history, path),
        frames=len(history) + len(path) - 1,
        interval=500,
        repeat=False,
    )

    plt.show()


def main() -> None:
    config = ConfigParser()
    config.read("single_agent.ini")
    grid_width = config["Grid"].getint("width")
    grid_height = config["Grid"].getint("height")
    goal_x = config["Goal"].getint("pos_x")
    goal_y = config["Goal"].getint("pos_y")
    robot_x = config["Agent"].getint("init_x")
    robot_y = config["Agent"].getint("init_y")
    obstacle_cost = config["Obstacles"].getint("movement_cost")
    obstacles = literal_eval(config["Obstacles"]["obstacles"])
    barriers = literal_eval(config["Obstacles"]["barriers"])

    search = SingleAgentAStar(
        grid_height,
        grid_width,
        (robot_x, robot_y),
        (goal_x, goal_y),
        obstacles,
        obstacle_cost,
        barriers,
    )
    search.perform_search()
    path = search.reconstruct_best_path()

    animate_search(
        grid_height,
        grid_width,
        (robot_x, robot_y),
        (goal_x, goal_y),
        obstacles,
        search.history,
        path,
    )


if __name__ == "__main__":
    main()
