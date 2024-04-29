import pytesseract
from PIL import Image
import os
import sys
# image = Image.open('paper-png/P19-1001/6.png')
# string = pytesseract.image_to_string('png0_data.txt')  # print ocr text from image
file = open("result.txt", mode='w')
file.write('hello')
# or

# import pdfplumber
# import cv2
#
# pdf = pdfplumber.open("paper-pdf/P19-1001.pdf")
# page = pdf.pages[6]
# # print(len(pdf.pages))
# tables = page.find_tables(
#     table_settings={"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 1,
#                     "min_words_horizontal": 3, "text_y_tolerance": 0}
# for table in tables:
#     print("hello_________________")
#     print(len(table.rows))
#     im = page.crop(table.bbox).to_image()
#     im.save("cooooool.png", format("PNG"))
