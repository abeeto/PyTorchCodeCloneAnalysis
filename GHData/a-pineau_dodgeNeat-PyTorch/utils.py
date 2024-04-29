import math
import pygame as pg
import matplotlib.pyplot as plt


def plot(mean_scores, sum_rewards, file_name):
    # see: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html

    # 1
    _, ax1 = plt.subplots()
    plt.title("Training...")

    color = "tab:red"
    ax1.set_xlabel("Games")
    ax1.set_ylabel("Mean scores", color=color)
    ax1.plot(mean_scores, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    value = round(mean_scores[-1], 1)
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(value), color=color)

    # 2
    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel(
        "Sum rewards",
        color=color,
    )
    ax2.plot(sum_rewards, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    value = sum_rewards[-1]
    plt.text(len(sum_rewards) - 1, sum_rewards[-1], str(value), color=color)

    # plot/saving
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.025)
    # plt.show()


def message(screen, msg, font_size, color, position) -> None:
    """
    Displays a message on screen.

    Parameters
    ----------
    msg: string (required)
        Actual text to be displayed
    font_size: int (required)
        Font size
    color: tuple of int (required)
        Text RGB color code
    position: tuple of int (required)
        Position on screen
    """
    font = pg.font.SysFont("Calibri", font_size)
    text = font.render(msg, True, color)
    text_rect = text.get_rect(topleft=(position))
    screen.blit(text, text_rect)


def distance(p1, p2) -> float:
    """Returns the distance between two points p1, p2."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
