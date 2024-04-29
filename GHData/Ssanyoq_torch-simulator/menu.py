import math
import os
import pygame
import sys

import files_manager
import main

pygame.init()
SIZE = 1200, 720


def draw_text(screen, text, pos_x, pos_y, color="White", font_size=50):
    """
    Рисует текст по заданным параметрам,
    если None, то ничего не рисует
    :param screen: полотно для рисования
    :param text: текст для рисования
    :param pos_x: x левого верхнего угла текста
    :param pos_y: y левого верхнего угла текста
    :param color: необязательный параметр, текст будет цвета color
    :param font_size: необязательный параметр, текст будет размера font_size
    :return: None
    """
    font = pygame.font.Font(None, font_size)
    string_rendered = font.render(str(text), True, pygame.Color(color))
    intro_rect = string_rendered.get_rect()
    intro_rect.y = pos_y
    intro_rect.x = pos_x
    screen.blit(string_rendered, intro_rect)


def draw_texts(screen, text, pos_x, pos_y, text_delta=0, color='White', font_size=50):
    """
    Для цикличного отрисовывания как в draw_text, не особо универсально
    :param screen: полотно для рисования
    :param text: список из текстов для рисования
    :param pos_y: y первого текста
    :param pos_x: x всех текстов
    :param text_delta: какое расстояние должно быть между текстами
                        (по оси Y)
    :param color: необязательный параметр, текст будет цвета color
    :param font_size: необязательный параметр, текст будет размера font_size
    :return: None
    """
    for line in text:  # Отрисовка текста

        draw_text(screen, line, pos_x, pos_y, color=color, font_size=font_size)
        pos_y += text_delta


def get_levels_names(folder='misc/levels'):
    """
    Возвращает список с именами всех .txt файлов из
    папки folder, например, ['level1','example2','level_1']
    folder - относительный путь папки от данного файла
    """
    level_names = []
    for (dir_path, dir_names, filenames) in os.walk(folder):
        for filename in filenames:
            if "." not in filename:
                # Бывают файлы без расширения
                # и они вызвали бы ошибку ниже
                continue
            elif filename.split('.')[1] == 'txt':
                # Проверка на .txt
                level_names.append(filename.split('.')[0])
    return level_names


def save_data(coins, torches, levels, is_changed):
    """
    Небольшая надстройка над функцией сохранения из
    files_manager, сделана в том числе для возможности
    улучшения игры
    """
    if not is_changed:
        return None
    files_manager.save_player_data(coins, torches, levels)


def check_os():
    """
    :return: True, если ОС продходит для проигрывания
            музыки и False, если нет
    """
    # Возникла проблема в том, что почему-то
    # звуки в pygame вызывают ошибку, если
    # программа запущена на Linux, поэтому
    # я сделал проверку на ОС. На всякий случай
    # звук будет работать только на Windows (фича)

    # Проблемы, походу, не с башкой, а с самим pygame

    return sys.platform.startswith("win")  # Windows - win32


def get_current_levels(all_levels, page):
    """
    Возвращает названия кнопок и названия уровней

    :param all_levels: список с названиями всех уровней
    :param page: номер страницы, нумерация с 0

    :return: картеж с 2-мя списками:
    первый список - названия самих файлов с уровнями,
    второй - названия кнопок
    Названия кнопок - это просто названия файлов,
    начинающиеся с заглавной буквы и
    все '_' заменены на ' '
    Также название урезано до 20 символов
    Если page больше, чем
    math.ceil(len(all_levels) / 5), то
    просто вернет ([],[])
    (прикольный смайлик кстати ([],[]))
    """
    current_levels = all_levels[page * 5:(page + 1) * 5]
    current_namings = []
    for name in current_levels:
        new_name = name.replace("_", " ").capitalize()
        if len(new_name) > 20:
            # Название длиннее 20 символов,
            # а значит его надо обрезать
            new_name = new_name[:20] + '...'
        current_namings.append(new_name)
    return current_levels, current_namings
    # ([],[])


def shop_screen(screen):
    """
    Меню с магазином, где можно купить факелы
    за монеты
    """
    screen.fill((0, 0, 0))

    button_pos_y = 250
    # y первой кнопки

    buttons = []
    coins, torches, levels = files_manager.load_player_data()
    # Список вида [[<rect кнопки>, <надпись на кнопке>]]
    # Вернется в меню, если было нажатие на кнопку с индексом 2

    buttons_names = ['Buy torch for 4 coins', 'Back to menu']
    for i in range(len(buttons_names)):
        button = pygame.draw.rect(screen, (200, 200, 200), (350, button_pos_y, 530, 60))
        buttons.append([button, buttons_names[i]])
        button_pos_y += 100
    text_pos_y = 265
    text_delta = 100
    draw_text(screen, "Shop", 550, 65, font_size=60)
    draw_texts(screen, buttons_names, 355, text_pos_y, text_delta)

    data_changed = False
    # Переменная для того, чтобы понять, были ли какие-либо
    # изменения в количестве монет или факелов

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_data(coins, torches, levels, data_changed)
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for i, button in enumerate(buttons):
                        # Проверка на совпадение координаты мыши с одной из кнопок
                        if pygame.Rect.collidepoint(button[0], pygame.mouse.get_pos()):
                            if i == 1:
                                save_data(coins, torches, levels, data_changed)
                                start_screen()
                                return None
                            elif button[1] is None:
                                continue
                            elif button[1].startswith('Buy torch for'):
                                price = int(button[1].split()[3])
                                if coins < price:
                                    continue
                                else:
                                    coins -= price
                                    torches += 1
                                    data_changed = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    save_data(coins, torches, levels, data_changed)
                    start_screen()
                    return None
            info_gui(screen, coins, torches)
        pygame.display.flip()


def level_screen(menu_screen):
    """
    Меню для выбора уровня
    """
    menu_screen.fill((0, 0, 0))

    # Создание кнопок
    button_pos_y = 50
    buttons = []
    # Список вида [[<rect кнопки>, <файл или функция, которой соответствует эта кнопка>,
    # <пройдено или нет>]]
    # Пройдено или нет - True/False, для всех кнопок не под
    # уровни этот флаг будет равен False
    # Вернется в меню, если было нажатие на кнопку с индексом 5
    # Тут нету кнопок вправо и влево (они другого размера)

    # Подготовка надписей на кнопках с уровнями
    all_levels = get_levels_names('misc/levels')
    current_levels, buttons_names = get_current_levels(all_levels, 0)
    current_page = 0

    text_pos_y = 65
    # Начальный button_pos_y + 1/4 размера кнопки, чтобы ровно посередине
    text_delta = 100

    coins, torches, levels_data = files_manager.load_player_data()

    # Рисование кнопок
    for i in range(5):
        button_color = (0, 0, 0) if len(buttons_names) - 1 < i else (200, 200, 200)
        button = pygame.draw.rect(menu_screen, button_color, (350, button_pos_y, 530, 60))
        button_pos_y += 100
        if len(current_levels) - 1 < i:
            # Значит на странице должно быть меньше 5 кнопок
            buttons.append([button, None, False])
        else:
            if current_levels[i] in levels_data.keys() and files_manager.check_if_completed(
                    levels_data[current_levels[i]], current_levels[i]):
                # Значит уровень пройден и надо это написать
                buttons.append([button, current_levels[i], True])
            else:
                buttons.append([button, current_levels[i], False])
    button = pygame.draw.rect(menu_screen, (200, 10, 10), (350, button_pos_y, 530, 60))
    buttons.append([button, None, False])
    button_pos_y += 100
    left_button = pygame.draw.rect(menu_screen, (200, 200, 200), (350, button_pos_y, 245, 40))
    right_button = pygame.draw.rect(menu_screen, (200, 200, 200),
                                    (350 + 265 + 20, button_pos_y, 245, 40))

    # Отрисовка текста
    draw_texts(menu_screen, buttons_names, 355, text_pos_y, text_delta)
    draw_text(menu_screen, "Back to menu", 500, text_pos_y + text_delta * 5)  # x на глазок
    draw_text(menu_screen, "<", 472, text_pos_y + text_delta * 6 - 15)
    draw_text(menu_screen, ">", 757, text_pos_y + text_delta * 6 - 15)

    change_buttons = False
    # Переменная чтобы красивее было
    for button in buttons:
        if button[2]:
            # Значит этот уровень пройден
            draw_text(menu_screen, "Completed", 790, button[0].y + 20, color="Green", font_size=25)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for i, button in enumerate(buttons):
                        # Проверка на совпадение координаты мыши с одной из кнопок
                        if pygame.Rect.collidepoint(button[0], pygame.mouse.get_pos()):
                            if i == 5:
                                start_screen()
                                return None
                            elif button[1] is None:
                                continue
                            else:
                                main.main(button[1])
                                return None

                    if pygame.Rect.collidepoint(left_button, pygame.mouse.get_pos()):
                        current_page -= 1 if current_page != 0 else 0
                        change_buttons = True
                    if pygame.Rect.collidepoint(right_button, pygame.mouse.get_pos()):
                        if current_page < math.ceil(len(all_levels) / 5) - 1:
                            current_page += 1
                            change_buttons = True
            if change_buttons:
                change_buttons = False
                current_levels, buttons_names = get_current_levels(all_levels, current_page)
                for i in range(5):
                    if len(buttons_names) - 1 < i:
                        buttons[i][1] = None
                        buttons[i][2] = False
                    else:
                        buttons[i][1] = current_levels[i]
                        if current_levels[i] in levels_data.keys() and \
                                files_manager.check_if_completed(
                                    levels_data[current_levels[i]], current_levels[i]):
                            buttons[i][2] = True
                        else:
                            buttons[i][2] = False
                button_pos_y = 50

                # Отрисовка новых кнопок и названий
                for i in range(5):
                    if len(buttons_names) - 1 < i:
                        color = (0, 0, 0)
                    else:
                        color = (200, 200, 200)
                    pygame.draw.rect(menu_screen, color,
                                     (350, button_pos_y, 530, 60))
                    button_pos_y += 100

                draw_texts(menu_screen, buttons_names, 355, text_pos_y, text_delta)
                for button in buttons:
                    if button[2]:
                        draw_text(menu_screen, "Completed", 790, button[0].y + 20, color="Green",
                                  font_size=25)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    start_screen()
                    return None
        info_gui(menu_screen, coins, torches)
        pygame.display.flip()


def ending_screen(screen, won=True, cur_level=None, reward=5):
    """
    Менюшка, надписи на которой зависят от того,
    победил игрок или нет
    :param screen: экран
    :param won: победил или нет (bool)
    :param cur_level: название нынешнего уровня, нужно только
                        для экрана поражения
    :param reward: сколько монет получил игрок
    """
    retry_button = None
    pygame.draw.rect(screen, (152, 130, 199), (300, 180, 600, 360))
    if won:
        bold_text = 'You won!'
        small_text = " " * 11 + f'+ {reward} coins'
    else:
        bold_text = 'You lost'
        small_text = 'Better luck next time'
        retry_button = pygame.draw.rect(screen, (200, 20, 20), (450, 460, 300, 45))
        draw_text(screen, 'Retry', 570, 470, font_size=35)

    draw_text(screen, bold_text, 490, 250, font_size=75)
    draw_text(screen, small_text, 430, 410)
    quit_button = pygame.draw.rect(screen, (200, 0, 0), (825, 180, 75, 75))
    draw_text(screen, 'X', 845, 195, font_size=75)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                main.clear_stuff()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Нажат кнопка с крестиком
                if quit_button.collidepoint(pygame.mouse.get_pos()):
                    main.clear_stuff()
                    start_screen()
                    return None
                if not won:
                    # Только если пользователь проиграл
                    if retry_button.collidepoint(pygame.mouse.get_pos()):
                        # Нажата кнопка retry
                        main.clear_stuff()
                        main.main(cur_level)
                        return None
        pygame.display.flip()
    pygame.quit()


def info_gui(screen, coins, torches):
    """
    Рисует маленький gui, где показано количество монет
    и факелов
    :param screen: экран для рисования
    :param coins: сколько монет
    :param torches: сколько торчей
    :return: None
    """
    pygame.draw.rect(screen, (184, 184, 184), (1100, 620, 98, 98))
    # Фон

    pygame.draw.rect(screen, (255, 156, 50), (1120, 640, 3, 3))
    pygame.draw.rect(screen, (50, 50, 50), (1120, 643, 3, 10))
    # Факел

    pygame.draw.circle(screen, (211, 183, 58), (1125, 687), 10)
    # Монетка

    draw_text(screen, torches, 1150, 640, font_size=25)
    draw_text(screen, coins, 1150, 680, font_size=25)
    # Тексты (кол-во монеток и факелов)


def start_screen():
    """
    Основное меню, где можно выйти из игры,
    войти в магазин или в меню выбора уровней
    """
    pygame.init()
    pygame.display.set_caption("Murky Gloom")
    screen = pygame.display.set_mode(SIZE)

    # Возникла проблема в том, что почему-то
    # звуки в pygame вызывают ошибку, если
    # программа запущена на Linux, поэтому
    # я сделал проверку на ОС. На всякий случай
    # звук будет работать только на Windows (фича)
    radio_available = check_os()

    buttons = [
        [pygame.rect.Rect(350, 230, 530, 60), (210, 200, 200), 'Start'],
        [pygame.rect.Rect(350, 320, 530, 60), (200, 200, 200), 'Shop'],
        [pygame.rect.Rect(350, 410, 530, 60), (200, 170, 170), 'Exit'],
        [pygame.rect.Rect(1110, 690, 75, 25), (150, 170, 170), 'Music'],
    ]
    # Список формата [[rect кнопки, RGB цвет, надпись на кнопке]]
    # Надписи будут сделаны белым цветом

    for i, button in enumerate(buttons):
        pygame.draw.rect(screen, button[1], button[0])
        if i == len(buttons) - 1:
            # Значит это кнопка радио
            draw_text(screen, button[2], 1125, 695, font_size=25)
        else:
            draw_text(screen, button[2], 550, 245 + 90 * i)
            # 245 - это button[0][0].y + button[0][0].height // 4

    radio = None
    check_radio = None

    if radio_available:
        radio = pygame.mixer.Sound('misc/sounds/music_in_menu.mp3')
        radio.play()
        radio.set_volume(0.1)
        check_radio = True

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if pygame.Rect.collidepoint(button[0], pygame.mouse.get_pos()):
                        if button[2] == "Start":
                            if radio_available:
                                radio.stop()
                            level_screen(screen)
                            # Отрисовываем меню с уровнями
                            return None

                        if button[2] == "Shop":
                            if radio_available:
                                radio.stop()
                            shop_screen(screen)
                            # Перекидывает на меню магазина
                            return None

                        if button[2] == "Exit":
                            # Просто выкидывает из игры
                            running = False
                            break

                        if button[2] == "Music":
                            # Выключает музыку
                            if radio_available:
                                if check_radio:
                                    radio.stop()
                                    check_radio = False
                                else:
                                    radio = pygame.mixer.Sound('misc/sounds/music_in_menu.mp3')
                                    radio.play()
                                    radio.set_volume(0.1)
                                    check_radio = True

        pygame.display.flip()
        clock.tick(100)
    if radio_available:
        radio.stop()


if __name__ == '__main__':
    start_screen()
