import pyautogui as pag
import time
import numpy as np
import cv2
import imutils
from selenium import webdriver
import pip
import pytesseract
from tkinter import Tk
try:
    from PIL import Image
except ImportError:
    import Image
base = {}
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
time.sleep(2)

def viber_photo(number):
    pos_click_num_x, pos_click_num_y = 257, 101
    pos_write_num_x, pos_write_num_y = 303, 259
    pos_click_to_send_message_x, pos_click_to_send_message_y = 407, 457
    pos_click_icon_inf_x, pos_click_icon_inf_y = 1284, 107
    pos_click_sharecontact_x, pos_click_sharecontact_y = 1241, 298
    pos_click_izbrannoe_x, pos_click_izbrannoe_y = 965, 262
    pos_click_to_confirm_x, pos_click_to_confirm_y = 759, 645

    pag.click(pos_click_num_x, pos_click_num_y, 1)  # набрать номер
    time.sleep(0.25)
    pag.click(pos_write_num_x, pos_write_num_y, 1)  # набирать номер
    time.sleep(1)
    pag.typewrite(number, interval=0.25)  # вбиваем номер в строку  number
    pag.click(pos_click_to_send_message_x, pos_click_to_send_message_y, 1)  # нажимаем на иконку
    time.sleep(1)
    search = pag.locateOnScreen(r'D:\screenshot\screen.png')
    if str(search) == "Box(left=791, top=408, width=260, height=120)": #всегда проверять координаты для оптимизации(могут не совпадать с тем, что в if)
        print("Not Exist")
        return "not exist"
    else:
        print(str(search))
        exit()
    pag.click(pos_click_icon_inf_x, pos_click_icon_inf_y, 1)  # нажимаем на иконку для получения информации
    time.sleep(1)
    pag.click(pos_click_sharecontact_x, pos_click_sharecontact_y, 1)  # нажимаем на иконку поделиться контактом
    pag.doubleClick()
    time.sleep(1)
    pag.click(pos_click_izbrannoe_x, pos_click_izbrannoe_y, 1)  # выбираем контакт избранное
    time.sleep(1)
    pag.click(pos_click_to_confirm_x, pos_click_to_confirm_y, 1)  # подтверждаем выбор контакта для сохраниния в избраном
    time.sleep(1)
    imagee = pag.screenshot(region=(696, 598, 280, 30))
    imagee.save(r"D:\\text\\text.png")
    text = pytesseract.image_to_string(Image.open('D:\\text\\text.png'), lang='rus+eng')
    pag.click(803, 647, 1) # нажимаем на отправленное фото чтобы открыть его в чате в большем формате
    time.sleep(1.3)
    image = pag.screenshot(region=(992, 66, 300, 260))
    image.save(r"D:\\photo\\1.png")

    return text
#viber_photo()

for i in range(9999999 + 1):
    number = 375296371921
    number += i
    number = str("+" + str(number))
    text = viber_photo(number)
    text = str(text).replace(".", '').replace(":", "").replace(",", '').replace('_', '').replace("/", '')
    if text == "not exist":
        base[number] = "not_exist"
    else:
        base[number] = str(text).replace("\n", '').replace("‘", '')
    print(number[6:], text)
    #break

def web():
    driver = webdriver.Chrome(r"C:\Users\user\chromedriver.exe")
    driver.get("https://search4faces.com/vkok/index.html")
    driver.find_element_by_xpath('//*[@id="upload-button"]').click()
    driver.find_element_by_xpath('/html/body/div[2]/div[1]/section/div/div/div[1]/button').click()
    time.sleep(2)
    pag.click(352, 426)
    a = 'D:\photo\1.png'
    copy(a)
    pag.hotkey('ctrl', 'v')
    driver.find_element_by_xpath('').click()
    time.sleep(100)
#web()

