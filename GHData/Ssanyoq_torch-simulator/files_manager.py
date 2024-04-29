import pathlib
import time  # Для тестирования


def last_change(file_path):
    """
    Возвращает дату последнего изменения
    файла с относительным путем file_path
    """
    file_name = pathlib.Path(file_path)
    assert file_name.exists(), f"File {file_name} does not exist"
    last_change_time = file_name.stat().st_mtime
    return last_change_time


def encrypt_str(text):
    """
    Ну это типа шифровка, просто чтобы немного защищеннее было
    text - текст для шифровки
    Вызывает ошибку, если его нету
    Ничего не возвращает, изменяет файл
    Возвращает зашифрованный str
    """
    new_data = ''
    count = 0
    for symbol in text:
        count += 1
        count = count % 5
        new_data += chr(ord(symbol) + count)
    return new_data


def decode_file(file):
    """
    Ну это типа декодировка, просто чтобы немного защищеннее было
    file - относительный путь файла от программы
    Вызывает ошибку, если его нету
    Возвращает расшифрованный файл в формате str
    """
    file = pathlib.Path(file)
    assert file.exists(), f"File {file} does not exist"
    with open(file, mode='r', encoding='utf-8') as f:
        file_data = f.read()
    new_data = ''
    count = 0
    for symbol in file_data:
        count += 1
        count = count % 5
        if symbol == '\n':
            new_data += symbol
        else:
            new_data += chr(ord(symbol) - count)
    return new_data


def load_player_data(file='misc/playerdata.txt'):
    """
    1-ая строка - количество монет
    2-ая - количество факелов
    Последующие строки сделаны в формате
    <название файла с уровнем>/<дата прохождения>
    Датой прохождения считается тот момент, когда
    пользователь запустил уровень и хранится в
    количестве секунд с начала эпохи
    :param file:
    :return: (<кол-во монет>, <кол-во факелов>, {<название уровня>: <дата прохождения>})

    Если файл пустой или его не существует,
    то все будет работать тоже
    """
    try:
        data = decode_file(file).split('\n')
    except AssertionError:
        data = [0, 0]
    try:
        coins = int(data[0])
    except Exception:
        coins = 0
    try:
        torches = int(data[1])
    except Exception:
        torches = 0
    levels_data = {}
    for i in range(1, len(data)):
        level_data = data[i].split('/')
        if len(level_data) != 2:
            continue
        if level_data[0] in levels_data.keys():
            # Произошло что-то странное
            continue
        try:
            levels_data[level_data[0]] = float(level_data[1])
        except ValueError:
            continue
    return coins, torches, levels_data


def save_player_data(coins=0, torches=0, levels_data=None, file='misc/playerdata.txt'):
    """
    Полностью противоположная load_player_data
    функция
    :param coins: кол-во монет, int или str, который можно переделать в int
    :param torches: кол-во факелов, int или str, который можно переделать в int
    :param levels_data: словарь формата {<название уровня>:<дата прохождения>}
    :param file: относительный путь до файла, куда надо загрузить
    :return: None
    """
    if levels_data is None:
        levels_data = {}
    file_data = f"{str(coins)}\n{str(torches)}"
    for key in levels_data.keys():
        file_data += f"\n{key}/{str(levels_data[key])}"
    file_data = encrypt_str(file_data)
    file = pathlib.Path(file)

    assert file.exists(), f"File {file} does not exist"

    with open(file, mode='w', encoding='utf-8') as f:
        f.write(file_data)


def check_if_completed(completing_time, file, path='misc/levels/', extension='.txt'):
    """
    Проверяет, пройден уровень или нет
    Если последнее
    """
    change = last_change(path + file + extension)
    if change > completing_time:
        return False
    else:
        return True
