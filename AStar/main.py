# import ast
from collections import deque
from configparser import ConfigParser
from enum import IntEnum

from matplotlib.animation import FuncAnimation  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from typing_extensions import TypeAlias


class NodeState(IntEnum):
    EMPTY = 0
    START = 1
    REACHED = 2
    FRONTIER = 3
    GOAL = 4


Coordinate: TypeAlias = tuple[int, int]


class SingleAgentAStar:
    """
    Not actually A-Star yet.
    Started with BFS to get the main program structure and animation.
    """

    def __init__(
        self, height: int, width: int, start: Coordinate, goal: Coordinate
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.frontier: deque[Coordinate] = deque()
        self.reached_from: dict[Coordinate, Coordinate] = {}
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
        return neighbors

    def update_history(self) -> None:
        grid = np.zeros((self.height, self.width))
        grid[self.goal[1]][self.goal[0]] = NodeState.GOAL
        for x, y in self.reached_from:
            grid[y][x] = NodeState.REACHED
        for x, y in self.frontier:
            grid[y][x] = NodeState.FRONTIER
        grid[self.start[1]][self.start[0]] = NodeState.START
        self.history.append(grid)

    def find_best_path(self) -> None:
        self.frontier.append(self.start)
        self.reached_from[self.start] = self.start

        while len(self.frontier) > 0:
            current_node = self.frontier.popleft()
            for next_node in self.get_neighbors(current_node):
                if next_node not in self.reached_from:
                    self.frontier.append(next_node)
                    self.reached_from[next_node] = current_node
                    self.update_history()
        self.update_history()

    def reconstruct_path(self) -> list[Coordinate]:
        current_node = self.goal
        path = []
        while current_node != self.start:
            path.append(current_node)
            current_node = self.reached_from[current_node]
        path.append(self.start)
        path.reverse()
        return path


def animation_step(i, im, history: list[np.ndarray], path: list[Coordinate]):
    if i < len(history):
        im.set_array(history[i])
    else:
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

    cmap = ListedColormap(["white", "red", "blue", "grey", "green"])
    im = ax.imshow(history[0], cmap=cmap)
    ax.annotate("START", start, ha="center")
    ax.annotate("GOAL", goal, ha="center")
    _anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(im, history, path),
        frames=len(history) + len(path) - 1,
        interval=2,
        repeat=False,
    )

    plt.show()


def main():
    config = ConfigParser()
    config.read("single_agent.ini")
    grid_width = config["Grid"].getint("width")
    grid_height = config["Grid"].getint("height")
    goal_x = config["Goal"].getint("pos_x")
    goal_y = config["Goal"].getint("pos_y")
    robot_x = config["Agent"].getint("init_x")
    robot_y = config["Agent"].getint("init_y")
    # obstacles_rocks = ast.literal_eval(config['Obstacles']['rocks'])
    # obstacles_trees = ast.literal_eval(config['Obstacles']['trees'])

    search = SingleAgentAStar(
        grid_height, grid_width, (robot_x, robot_y), (goal_x, goal_y)
    )
    search.find_best_path()
    path = search.reconstruct_path()

    animate_search(
        grid_height,
        grid_width,
        (robot_x, robot_y),
        (goal_x, goal_y),
        search.history,
        path,
    )


if __name__ == "__main__":
    main()
