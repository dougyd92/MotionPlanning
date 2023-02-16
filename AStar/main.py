# import ast
from collections import deque
from enum import IntEnum

from configparser import ConfigParser
from matplotlib.animation import FuncAnimation  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


class NodeState(IntEnum):
    EMPTY = 0
    START = 1
    REACHED = 2
    FRONTIER = 3
    GOAL = 4


class SingleAgentAStar:
    """
    Not actually A-Star yet.
    Started with BFS to get the main program structure and animation.
    """

    def __init__(
        self, height: int, width: int, start: tuple[int, int], goal: tuple[int, int]
    ) -> None:
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.frontier: deque[tuple[int, int]] = deque()
        self.reached: set[tuple[int, int]] = set()
        self.history: list[np.ndarray] = []
        self.update_history()

    def get_neighbors(self, coordinate: tuple[int, int]) -> list[tuple[int, int]]:
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
        for x, y in self.reached:
            grid[y][x] = NodeState.REACHED
        for x, y in self.frontier:
            grid[y][x] = NodeState.FRONTIER
        grid[self.start[1]][self.start[0]] = NodeState.START
        self.history.append(grid)

    def find_best_path(self) -> None:
        self.frontier.append(self.start)
        self.reached.add(self.start)

        while len(self.frontier) > 0:
            current_node = self.frontier.popleft()
            for next_node in self.get_neighbors(current_node):
                if next_node not in self.reached:
                    self.frontier.append(next_node)
                    self.reached.add(next_node)
                    self.update_history()
        self.update_history()


def animation_step(i, im, history: list[np.ndarray]):
    im.set_array(history[i])
    return [im]


def animate_search(
    grid_height: int, grid_width: int, history: list[np.ndarray]
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
    _anim = FuncAnimation(
        fig,
        animation_step,
        fargs=(im, history),
        frames=len(history),
        interval=250,
        repeat_delay=3000,
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
    animate_search(grid_height, grid_width, search.history)


if __name__ == "__main__":
    main()
