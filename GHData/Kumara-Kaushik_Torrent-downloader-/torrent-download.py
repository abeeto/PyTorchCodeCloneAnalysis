#! python3
import sys
import pyperclip
import pyautogui
import time

pyautogui.PAUSE = 1
pyautogui.FAILSAFE = True
torch = (562, 1060)
newwindow = (545, 905)
search = (1350, 45)
click1 = (1700, 600)
cross = (381, 14)
click2 = (200, 500)
download = (500, 875)
green = (1200, 815)
gdown = (1747, 46)
print("Press Ctrl-C to quit.")
# writing code to copy search term to clipboard
try:
    if len(sys.argv) > 1:
        show = '-'.join(sys.argv[1:])
        searchterm = 'torrentcounter.eu/' + show + '-torrent'

        # gui automation: opening torch browser
        pyautogui.rightClick(torch[0], torch[1])
        pyautogui.click(newwindow[0], newwindow[1], duration=0.25)
        time.sleep(3)
        pyautogui.click(search[0], search[1])
        pyautogui.typewrite(['backspace'])
        pyautogui.typewrite(searchterm)
        pyautogui.press('enter')
        time.sleep(3)
        pyautogui.click(click1[0], click1[1])
        pyautogui.click(cross[0], cross[1], duration=0.25)
        pyautogui.click(click2[0], click2[1])
        pyautogui.scroll(-250)
        pyautogui.click(download[0], download[1])
        time.sleep(17)
        pyautogui.click(green[0], green[1])
        pyautogui.click(gdown[0], gdown[1], duration=0.5)


except KeyboardInterrupt:
    print('\nDone.')
