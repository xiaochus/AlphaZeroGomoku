# -*- coding: utf-8 -*-
"""
Version : 0.1.0
Date : 5th Mar 2018

Author : xiaochus
Email : xiaochus@live.cn
Affiliation : School of Computer Science and Communication Engineering
                - Jiangsu University - China

License : MIT
Status : Under Active Development

Description :
Pygame and Keras implementation of Gomoku game with simple AlphaZero.
"""
import pygame

from alpha import config as c
from alpha.game.game import Game
from alpha.model.player import AlphaZeroPlayer, HumanPlayer


def draw_background(screen, edge, grid):
    """Draw the check board background.

    # Arguments
        screen: game screen.
        edge: Integer, edge size of screen.
        grid: Integer, grid size in screen.
    """
    background = pygame.image.load('images/background.png').convert()
    back_rect = background.get_rect()
    screen.blit(background, back_rect)

    rect = [((grid, grid), (grid, edge - grid)),
            ((grid, grid), (edge - grid, grid)),
            ((grid, edge - grid), (edge - grid, edge - grid)),
            ((edge - grid, grid), (edge - grid, edge - grid)), ]

    for line in rect:
        pygame.draw.line(screen, (0, 0, 0), line[0], line[1], 2)

    for i in range(c.SIZE[0]):
        pygame.draw.line(screen, (0, 0, 0),
                         (grid * (2 + i), grid),
                         (grid * (2 + i), edge - grid))
        pygame.draw.line(screen, (0, 0, 0),
                         (grid, grid * (2 + i)),
                         (edge - grid, grid * (2 + i)))


def draw_movements(screen, movements, grid):
    """Draw the all piece position.

    # Arguments
        screen: game screen.
        movements: Dict, piece of different player.
        grid: Integer, grid size in screen.
    """
    for m in movements["first"]:
        pos = (((m[1] + 1) * grid, (m[0] + 1) * grid))
        pygame.draw.circle(screen, (0, 0, 0), pos, 16)
    for m in movements["second"]:
        pos = (((m[1] + 1) * grid, (m[0] + 1) * grid))
        pygame.draw.circle(screen, (255, 255, 255), pos, 16)


def draw_text(screen, text, size, x, y, color):
    """Draw the text on screen.

    # Arguments
        screen: game screen.
        text: String, string on the board.
        size: Integer, size of text.
        x: Integer, x coordinate screen.
        y: Integer, y coordinate screen.
        color: tuple, color of text.
    """
    font = pygame.font.Font(pygame.font.get_default_font(), size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    screen.blit(text_surface, text_rect)


def show_game_result(screen, edge, win, winner):
    """Check winner and draw result.

    # Arguments
        screen: game screen.
        edge: Integer, edge size of screen.
        win: Integer, if win.
        winner: winner.
    """
    text = ''

    if win == 0:
        text = "Draw!"
    if win == 1:
        if (c.FIRST and winner == 1) or (not c.FIRST and winner == 2):
            text = "You lose!"
        else:
            text = "You win!"

    size = 64
    x, y = edge // 2, 10
    draw_text(screen, text, size, x, y, (255, 0, 0))

    size = 22
    x, y = edge // 2, edge // 2
    draw_text(screen, 'Press any key to exit.', size, x, y, (0, 0, 255))
    pygame.display.flip()
    waiting = True

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYUP:
                waiting = False


def main():
    FPS = 30
    edge = 720
    grid = edge // (c.SIZE[0] + 1)

    pygame.init()

    screen = pygame.display.set_mode((edge, edge))
    pygame.display.set_caption("Gomoku")

    clock = pygame.time.Clock()

    running = True

    game = Game(c.SIZE, c.PIECE, 1)
    AIPlayer = AlphaZeroPlayer()
    ManPlayer = HumanPlayer(grid)

    if c.FIRST:
        players = [AIPlayer, ManPlayer]
    else:
        players = [ManPlayer, AIPlayer]

    turn = 0

    while running:
        click = None
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click = event

        cp = game.board.get_current_player()
        if not click and players[cp - 1].id == 'human':
            if turn == 0:
                draw_background(screen, edge, grid)
                pygame.display.flip()
            continue

        win, winner, movements = game.play(players, click)

        turn += 1

        draw_background(screen, edge, grid)
        draw_movements(screen, movements, grid)
        pygame.display.flip()

        if win in [0, 1]:
            show_game_result(screen, edge, win, winner)
            running = False

    pygame.quit()


if __name__ == '__main__':
    main()
