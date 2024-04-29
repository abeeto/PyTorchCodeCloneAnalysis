import telegram
import json


class ExamAlarmBot:
    def __init__(self, key_path='data/telegram_key.json'):
        with open(key_path, 'r') as f:
            self.keys = json.load(f)
        self.bot = telegram.Bot(token=self.keys['token'])

    def send_msg(self, message):
        self.bot.sendMessage(chat_id=self.keys['chat_id'], text=message)
