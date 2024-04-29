import os
import shutil
import datetime
import sys



path_save_folder = r'C:\Users\TOSHIBA\Documents\my games\runic games\torchlight 2\save'

try:
    folder_name = os.listdir(path_save_folder)[0]
    print('folder name: ', folder_name)
except IndexError:
    print('Ошибка! Папка с сохранением не найдена!')
    sys.exit(1)

cmd = input('>>>>')


def copy_save(path_original_dir):

    timestamp = datetime.datetime.today().strftime("%Y%m%d_%H_%M_%S")
    path_finish_folder = os.path.join(os.getcwd(), 'temp', timestamp)
    # print(os.listdir(os.path.join(os.getcwd(), 'temp')))
    shutil.copytree(path_original_dir, path_finish_folder)
    return print('Done!')

def back_save(path_original_dir):
    last_dir = os.listdir(os.path.join(os.getcwd(), 'temp', ))[-1]

    # удаление исходной папки с файлами сохранения игры
    try:
        shutil.rmtree(os.path.join(path_original_dir, folder_name))
    except:
        print('Ошибка удаления каталога!')

    # копируем папку со старым сохраннием обратно в папку save
    try:
        shutil.copytree(os.path.join(os.getcwd(), 'temp', last_dir, folder_name), os.path.join(path_original_dir, folder_name))
    except:
        print('Ошибка копирования каталога!')

    return



if cmd == 'save' :
    copy_save(path_save_folder)

elif cmd == 'back' :
    back_save(path_save_folder)

else:
    print('error')


