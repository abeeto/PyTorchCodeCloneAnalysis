#   -*- coding: utf-8   -*-

import time
import telebot

token='578427356:AAEkzG_1LUiXbD99SRlkwttql8_L5LFNiV4'

bot = telebot.TeleBot(token)

@bot.message_handler(content_types=['text'])
def	echo_msg(message):
				bot.send_message(message.chat.id,	message.text)
if  __name__    ==  '__main__':
                    bot.polling(none_stop=True)
