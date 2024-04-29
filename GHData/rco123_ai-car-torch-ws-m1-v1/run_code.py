kevent = None
key = None
angle = None
stop_cnt = None
rtn = None
speed = None
gostop = None


from robot import ps_controller
from robot import ai_controller
from robot import robot
ps = ps_controller()
robo = robot()
ai = ai_controller()
ai.cam_open()
stop_cnt = 0
speed = 200
gostop = 'stop'

while True:
    kevent = ps.check_event()
    if kevent == True:
        key = ps.key_read()
        if key == 7:
            key = ps.key_read()
            robo.delay(1)
        if key == 8:
            gostop = 'go'
            speed = 200
        if key == 9:
            gostop = 'stop'
            speed = 0
            robo.move(angle, speed)
        if key == 10:
            stop_cnt = 0
            print('stop_cnt = 0')
    ai.cam_img_get()
    angle = ai.cam_img_to_angle()

    rtn = ai.xml_detector('./classifier/cascade.xml')
    if rtn:
        if stop_cnt == 0:
            gostop = 'stop'
            robo.move(angle, 0)
        stop_cnt = stop_cnt + 1
    ai.img_display()
    if gostop == 'go':
        robo.move(angle, speed)
        print(f'{angle}, {speed}')
    robo.delay(0.01)
