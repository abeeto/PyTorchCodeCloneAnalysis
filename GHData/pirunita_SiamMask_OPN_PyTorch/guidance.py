import cv2

class Guide(object):
    def __init__(self, name, img, padding_factor):
        
        self.name = name
        
        self.img = img
        self.img_Y, self.img_X, _ = img.shape
        self.linewidth = int(min(self.img_Y, self.img_X) // 100) # for rectangle linewidth
        self.padding_X = int(self.img_X * padding_factor)
        self.padding_Y = int(self.img_Y * padding_factor)
        
        self.rect_img = self.img.copy()
        self.temp_img = None
        self.point_toggle = False
        self.drawing_toggle = False
        self.roi = None
        
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.yellow = (0, 255, 230)
        
        self.minX, self.minY, self.maxX, self.maxY, self.aX, self.aY, self.bX, self.bY = -1, -1, -1, -1, -1, -1, -1, -1
        
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback(self.name, self.draw_box, param=self.rect_img)
        while True:
            cv2.imshow(self.name, self.rect_img)
            
            input_key = cv2.waitKey(20) & 0xFF
            if input_key == 27:
                break
            
                
                
            elif input_key == ord('a') and self.point_toggle == False:
                print('X')
                self.padding_control()
                self.roi = cv2.rectangle(self.rect_img, (self.minX, self.minY), (self.maxX, self.maxY), (0, 255, 0), self.linewidth)
                print(self.minX, self.minY, self.maxX, self.maxY)
            
            elif input_key == ord('b'):
                self.padding_control()
                break
    cv2.destroyAllWindows()
        
    def init_rect(self):
        return self.minX, self.minY, self.maxX, self.maxY
    
    def draw_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_toggle = True
            
            cv2.imshow(self.name, self.rect_img)
            if self.point_toggle is False:
                self.rect_img = self.img.copy()
                self.aX, self.aY = x, y
            else:
                self.bX, self.bY = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_toggle:           
            if self.drawing_toggle:
                if self.point_toggle is False:
                    self.rect_img = self.img.copy()
                    cv2.circle(self.rect_img, (x, y), self.linewidth, self.red, self.linewidth)
                    cv2.line(self.rect_img, (x, y), (self.img_X, y), self.red, self.linewidth)
                    cv2.line(self.rect_img, (x, y), (x, self.img_Y), self.red, self.linewidth)
                    
                    
                    # Padding Visualization
                    if x <= self.padding_X:
                        k = cv2.rectangle(self.rect_img, (0, y), (x, self.img_Y), color=self.yellow, thickness=-1)
                        cv2.addWeighted(k, 0.2, self.rect_img, 0.8, 0, self.rect_img)
                    if y <= self.padding_X:
                        k = cv2.rectangle(self.rect_img, (x, 0), (self.img_X, y), color=self.yellow, thickness=-1)
                        cv2.addWeighted(k, 0.2, self.rect_img, 0.8, 0, self.rect_img)
                    
                    self.temp_img = self.rect_img.copy()
                
                else:
                    self.rect_img = self.temp_img.copy()
                    cv2.circle(self.rect_img, (x, y), self.linewidth, self.blue, self.linewidth)
                    cv2.line(self.rect_img, (0, y), (x, y), self.blue, self.linewidth)
                    cv2.line(self.rect_img, (x, 0), (x, y), self.blue, self.linewidth)
                    
                    # Padding Visualization
                    if x >= self.img_X - self.padding_X:
                        k = cv2.rectangle(self.rect_img, (x, 0), (self.img_X, y), color=self.yellow, thickness=-1)
                        cv2.addWeighted(k, 0.2, self.rect_img, 0.8, 0, self.rect_img)
                    if y >= self.img_Y - self.padding_Y:
                        k = cv2.rectangle(self.rect_img, (0, y), (x, self.img_Y), color=self.yellow, thickness=-1)
                        cv2.addWeighted(k, 0.2, self.rect_img, 0.8, 0, self.rect_img)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_toggle = False
            if self.point_toggle is False:
                cv2.circle(self.rect_img, (x, y), self.linewidth, self.red, self.linewidth)
                cv2.line(self.rect_img, (x, y), (self.img_X, y), self.red, self.linewidth)
                cv2.line(self.rect_img, (x, y), (x, self.img_Y), self.red, self.linewidth)
                self.point_toggle = True
                
                self.aX, self.aY = x, y
            
            else:
                cv2.circle(self.rect_img, (x, y), self.linewidth, self.blue, self.linewidth)
                cv2.line(self.rect_img, (0, y), (x, y), self.blue, self.linewidth)
                cv2.line(self.rect_img, (x, 0), (x, y), self.blue, self.linewidth)
                self.point_toggle = False
                
                self.bX, self.bY = x, y
                self.minX, self.maxX = min(self.aX, self.bX), max(self.aX, self.bX)
                self.minY, self.maxY = min(self.aY, self.bY), max(self.aY, self.bY)
    
    def padding_visualization(self, guide_type):
        if guide_type == 1:
            self.roi = cv2.rectangle(self.rect_img, (self.minX, self.minY), (self.maxX, self.maxY), (0, 255, 0), self.linewidth)
            
                

    def padding_control(self):
        # maxX
        if self.maxX >= self.img_X - self.padding_X:
            self.maxX = self.img_X
        # minX
        if self.minX <= self.padding_X:
            self.minX = 0
        # maxY
        if self.maxY >= self.img_Y - self.padding_Y:
            self.maxY = self.img_Y
        #minY
        if self.minY <= self.padding_Y:
            self.minY = 0
    