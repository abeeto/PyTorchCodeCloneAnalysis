#!/usr/bin/python
import curses
import curses.wrapper
from adventure import Adventure
from lightsource import Adventurer
from lightsource import LightSource


def print_main_window(window):
    #Print the command list
    window.border(0)
    maxyx = window.getmaxyx()
    maxy = maxyx[0]
    command_list = []
    command_list.append("1. Add Adventurer")
    command_list.append("2. Modify Adventurer")
    command_list.append("3. Use Light Source")
    command_list.append("4. Modify Light Source")
    command_list.append("5. Advance Turn Counter")
    command_list.append("6. Quit")
    x = len(command_list)
    for command in command_list:
        window.addstr(maxy - (x + 1), 1, command)
        x -= 1
    #Print the list of active light sources
    index = 0
    initial_line = (window.getmaxyx()[0] - len(current_adventure.lightsources) + 1) / 2
    window.addstr(initial_line, 1, "Active Light Sources: ")
    initial_line += 1
    for lightsource in current_adventure.lightsources:
        window.addstr(initial_line + index, 1, lightsource.get_type())
        index += 1
    #Print the list of adventurers and their available sources
    adventurer_list = current_adventure.get_adventurer_list()
    line = 1
    lighting_position = 21
    torches_position = 30
    candles_position = 41
    oil_position = 52
    for adventurer in adventurer_list:
        window.addstr(line, 1, str(line) + " " + adventurer.get_name())
        window.addstr(line, lighting_position, adventurer.calculate_lighting())
        window.addstr(line, torches_position, str(adventurer.torches_available) + " torches")
        window.addstr(line, candles_position, str(adventurer.candles_available) + " candles")
        if adventurer.lantern_available:
            window.addstr(line, oil_position, str(adventurer.lantern_oil_available) + " oil flasks")
        line += 1
def main_command_loop(window, adventure):
    while(True):
        print_main_window(window)
        command = window.getch()
        if(command == ord('6')):
            break
        if(command == ord('1')):
            add_adventurer_command(window, adventure)
        if(command == ord('2')):
            select_adventurer_for_edit(window, adventure)
        window.move(0, 0)
        window.clear()
        window.refresh()

def add_adventurer_command(window, adventure):
#    curses.nocbreak()
    curses.echo()
    main_maxyx = window.getmaxyx()
    add_adventurer_window_width = main_maxyx[1] / 2
    add_adventurer_window_height = main_maxyx[0] / 2
    add_adventurer_window = curses.newwin(add_adventurer_window_height,
                                          add_adventurer_window_width,
                                          main_maxyx[0] / 4,
                                          main_maxyx[1] / 4)
    add_adventurer_window.border(0)
    add_adventurer_window.keypad(True)
    prompt_string = "Enter new adventurer's name:"
    add_adventurer_window.addstr(add_adventurer_window_height / 2,
                                1,
                                prompt_string)
    add_adventurer_window.addstr((add_adventurer_window_height / 2) + 2, 1, "(F1 To Cancel)")
    add_adventurer_window.move(add_adventurer_window_height / 2, len(prompt_string) + 1)
    key = add_adventurer_window.getch()
    if not key == curses.KEY_F1:
        new_adventurer = Adventurer(chr(key) + add_adventurer_window.getstr())
        adventure.add_adventurer(new_adventurer)
        curses.cbreak()
        curses.noecho()
        #Now call the function to allow adding sources
        edit_adventurer(add_adventurer_window, new_adventurer)
    curses.cbreak()
    curses.noecho()
    #Now call the function to allow adding sources

def select_adventurer_for_edit(window, adventure):
    curses.echo()
    main_maxyx = window.getmaxyx()
    select_adventurer_window_width = main_maxyx[1] / 2
    select_adventurer_window_height = main_maxyx[0] /2
    select_adventurer_window = curses.newwin(select_adventurer_window_height,
                                             select_adventurer_window_width,
                                             main_maxyx[0] / 4,
                                             main_maxyx[1] / 4)
    select_adventurer_window.border(0)
    select_adventurer_window.keypad(True)
    prompt_string = "Which adventurer do you wish to edit?"
    select_adventurer_window.addstr(select_adventurer_window_height / 2,
                                    1,
                                    prompt_string)
    select_adventurer_window.addstr((select_adventurer_window_height / 2) + 2,
                                    1,
                                    "(F1 To Cancel)")
    select_adventurer_window.move(select_adventurer_window_height / 2, len(prompt_string) + 1)
    key = select_adventurer_window.getch()
    index = -1
    try:
        index = int(chr(key)) - 1
    except ValueError:
        index = -1
    if not key == curses.KEY_F1 and index >= 0 and len(adventure.get_adventurer_list()) > index:
        current_adventurer = adventure.get_adventurer_list()[index]
        curses.cbreak()
        curses.noecho()
        edit_adventurer(select_adventurer_window, current_adventurer)
    curses.cbreak()
    curses.noecho()

def edit_adventurer(window, adventurer):
    lantern_added = False
    torches_added = 0
    candles_added = 0
    lantern_oil_added = 0
    while(True):
        window.clear()
        if lantern_added:
            prompt_string = "Lantern Added!"
            window.addstr((window.getmaxyx()[0] / 2)+1,
                          (window.getmaxyx()[1] - len(prompt_string)) / 2,
                          prompt_string)
        if torches_added > 0:
            prompt_string = " torches added"
            window.addstr((window.getmaxyx()[0] / 2)+2,
                        (window.getmaxyx()[1] - len(prompt_string)) / 2,
                         str(torches_added) + prompt_string)
        if candles_added > 0:
            prompt_string = " candles added"
            window.addstr((window.getmaxyx()[0] / 2)+3,
                          (window.getmaxyx()[1] - len(prompt_string)) / 2,
                          str(candles_added) + prompt_string)
        if lantern_oil_added > 0:
            prompt_string = " flasks of oil added"
            window.addstr((window.getmaxyx()[0] / 2)+4,
                          (window.getmaxyx()[1] - len(prompt_string)) / 2,
                          str(lantern_oil_added) + prompt_string)

        window.border(0)
        window.addstr(1, 1, "1. Add Lantern")
        window.addstr(2, 1, "2. Add Torches")
        window.addstr(3, 1, "3. Add Candles")
        window.addstr(4, 1, "4. Add Lantern Oil")
        window.addstr(5, 1, "5. Finished Editing")
        command = window.getch()
        if command == ord('5'):
            break
        if command == ord('1'):
            adventurer.add_lantern()
            lantern_added = True
        if command == ord('2'):
            window.clear()
            window.border(0)
            prompt_string = "How many torches?"
            window.addstr(1, 1, prompt_string)
            curses.echo()
            torches_added = int(window.getstr())
            adventurer.add_torches(torches_added)
            curses.noecho()
        if command == ord('3'):
            window.clear()
            window.border(0)
            prompt_string = "How many candles?"
            window.addstr(1, 1, prompt_string)
            curses.echo()
            candles_added = int(window.getstr())
            adventurer.add_candles(candles_added)
            curses.noecho()
        if command == ord('4'):
            window.clear()
            window.border(0)
            prompt_string = "How many flasks?"
            window.addstr(1, 1, prompt_string)
            curses.echo()
            lantern_oil_added = int(window.getstr())
            adventurer.add_lantern_oil(lantern_oil_added)
            curses.noecho()

current_adventure = Adventure()
torch1 = LightSource("Torch", 2, 2, 2)
torch2 = LightSource("Torch", 2, 2, 2)
current_adventure.add_lightsource(torch1)
current_adventure.add_lightsource(torch2)
myscreen = curses.initscr()
curses.cbreak()
curses.noecho()
myscreen.border(0)
main_command_loop(myscreen, current_adventure)
myscreen.refresh()
curses.endwin()
