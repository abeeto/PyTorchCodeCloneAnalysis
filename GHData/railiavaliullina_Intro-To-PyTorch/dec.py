"""
В Python всё является объектом, а не только объекты, которые вы создаёте из классов.
В этом смысле Python полностью соответствует идеям объектно-ориентированного программирования.
Это значит, что в Python всё это — объекты:
- числа;
- строки;
- классы (да, даже классы!);
- функции (то, что нас интересует).
Тот факт, что всё является объектами, открывает перед нами множество возможностей.
Мы можем сохранять функции в переменные, передавать их в качестве аргументов и возвращать из других функций.
Можно даже определить одну функцию внутри другой. Иными словами, функции — это объекты первого класса.
"""
#
# def hello_world():
#     print('Hello world!')
#
# # хранение функции в переменной
# hello = hello_world
# # hello()
#
# # создание функции внутри функции
# def wrapper_function():
#     def hello_world():
#         print('Hello world!')
#     hello_world()
#
# # wrapper_function()
#
# # передавать функции в качестве аргумента и возвращать их
# # функция высшего порядка, так как принимает функцию в качестве аргумента
# def higher_order(func):
#     print('Получена функция {} в качестве аргумента'.format(func))
#     func()
#     return func
#
# # higher_order(hello_world)
# ######################################################################
# """
# Декоратор — это функция, которая позволяет обернуть другую функцию для расширения
# её функциональности без непосредственного изменения её кода.
# """
# def decorator_function(func):
#     # Внутри decorator_function() определили другую функцию-обёртку,
#     # которая обёртывает функцию-аргумент и затем изменяет её поведение.
#     def wrapper():
#         print('Функция-обёртка!')
#         print('Оборачиваемая функция: {}'.format(func))
#         print('Выполняем обёрнутую функцию...')
#         func()
#         print('Выходим из обёртки')
#     return wrapper
#
# @decorator_function
# def hello_world1():
#     print('Hello world!')
#
# # hello_world1()
#
# # hello_world1 = decorator_function(hello_world1)
#
# """Иными словами, выражение @decorator_function вызывает decorator_function() с hello_world1
# в качестве аргумента и присваивает имени hello_world1 возвращаемую функцию."""
#
# # декоратор, замеряющий время выполнения функции
# def benchmark(func):
#     import time
#
#     def wrapper():
#         start = time.time()
#         func()
#         end = time.time()
#         print('[*] Время выполнения: {} секунд.'.format(end - start))
#
#     return wrapper
#
#
# @benchmark
# def fetch_webpage():
#     import requests
#     webpage = requests.get('https://google.com')
#
#
# # fetch_webpage()
#
# # decorator ы аргументами
#
#
# input_str = "My func!"
# def decorator_func(func):
#     def wrapper(*args):
#         print("Something is happening before the function is called.")
#         func(*args)
#         print("Something is happening after the function is called.")
#
#     return wrapper
#
# @decorator_func
# def my_func(arg1):
#     print(arg1)
#
# # my_func(input_str)
# # Something is happening before the function is called.
# # My func!
# # Something is happening after the function is called.


def dec_func(f):
    def wrapper():
        print(1)
        f()
        print(2)
    return wrapper


@dec_func
def func():
    print('hello world')

func()
