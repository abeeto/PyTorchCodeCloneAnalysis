from PIL import Image
from flask import Flask, make_response, send_from_directory
from flask_restful import Resource, Api
from utils.utils import get_location_parsing,get_param_parsing, get_image, encode_img, FileControl, get_box_num, get_time_list, load_encoded_img
from tracking_utils.tracking_module import main as tracking

path = FileControl()

UPLOAD_FOLDER = path.image_dir
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


class Index(Resource):
    description = '<h1>API Description</h1>'\
                  '<table border="1" style="border-collapse:collapse">'\
                  '<tr><td>/index</td><td>GET</td><td>API 설명 페이지(this)</td></tr>'\
                  '<tr><td>/tracking</td><td>POST</td><td>image를 전송해 tracking 처리하여 저장하는 API</td></tr>'\
                  '<tr><td>/get_image</td><td>GET</td><td>Image를 요청하여 받는 API</td></tr>'\
                  '<tr><td>/room_info</td><td>GET</td><td>카메라 번호(Room 번호) 리스트를 요청하여 받는 API</td></tr>'\
                  '</table>'

    def get(self):
        res = make_response(self.description)
        res.headers['Content-type'] = 'text/html; charset=utf-8'
        return res

class TrackingImage(Resource):
    """Image를 HTTP post 방식으로 받아서 tracking해주는 API
        location(방 번호)와 frame(프레임 번호)는 GET 방식으로 입력받음
    
    Args:
        Resource ([Resource object]): /tracking?location=a&time=b
            a ([int]): 트레킹 요청된 location, default = 0
            b ([str]): 트레킹 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
        Request file ([base64 object]): 트레킹 요청된 이미지
    
    Returns:
        ([Response object])
            file_path ([str]): 이미지 저장된 경로,
            image ([base64 object]): Tracking 결과 이미지
            box_num ([int]):검출된 box 개수
    """
    # TODO: 매 프레임마다 해야하는 동작에 return이 불필요하고 많음

    def get(self):
        return self.post()

    @staticmethod
    def post():
        location, time, _ = get_param_parsing()
        img = get_image()
        file_path, img, box_num = tracking(location, time, img)
        encoded_img = encode_img(img)
    
        res = make_response({'file_path':file_path,'image':encoded_img,'box_num':box_num})
        res.headers['Content-type'] = 'application/json'

        return res


class SendImage(Resource):
    """특정 location, time에 해당하는 결과 이미지, 검출된 박스 개수 반환
    
    Args:
        Resource ([Resource object]): /get_image?location=a&time=b
            a ([int]): 요청된 location, default = 0
            b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
    Returns:
        Json
        ([Response object])
            image ([base64 object]): Tracking 결과 이미지
            box_num ([int]):검출된 box 개수
    """
    
    @staticmethod
    def get():
        location, time, show_image = get_param_parsing()
        
        # return time, saved_image, box num
        if show_image:
            file_path, file_name = path.get_tracked_image_path(location, time, return_join=False)

            return send_from_directory(file_path, file_name)
        else:
            img_path = path.get_tracked_image_path(location, time)
            encoded_img = load_encoded_img(img_path)
            
            box_num = get_box_num(location,time)
            res = make_response({'image': encoded_img, 'box_num': box_num})
            res.headers['Content-type'] = 'application/json'

            return res

class SendInfo(Resource):
    """특정 location에 대한 time_list 반환
    
    Args:
        Resource ([Resource object]): /location_info?location=a
            a = 요청된 location, default = 0
    Returns:
        ([str list]): 특정 location에 대한 time_list
    """
    @staticmethod
    def get():
        location = get_location_parsing()
        res = make_response({'time_list': get_time_list(location)})
        res.headers['Content-type'] = 'application/json'
        return res

class SendRooms(Resource):
    """특정 location에 대한 time_list 반환

    Args:
        Resource ([Resource object]): /location_info?location=a
            a = 요청된 location, default = 0
    Returns:
        ([str list]): 특정 location에 대한 time_list
    """

    @staticmethod
    def get():
        import glob
        room_list = glob.glob('./media/tracking/*')
        room_list = [room.split('/')[-1] for room in room_list]
        if len(room_list[0].split('\\')) > 1:
            room_list = [room.split('\\')[-1] for room in room_list]
        room_list = [{'room_num': room_num} for room_num in room_list]
        res = make_response({'room_list': room_list})
        res.headers['Content-type'] = 'application/json'
        return res


api.add_resource(Index, '/', '/index')
# API를 간단히 설명해주는 페이지

api.add_resource(TrackingImage, '/tracking')
# /tracking?location=a&time=b
# a ([int]): 요청된 location, default = 0
# b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
# 

api.add_resource(SendImage, '/get_image')
# /get_image?location=a&time=b
# a ([int]): 요청된 location, default = 0
# b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'

api.add_resource(SendInfo, '/location_info')
# /location_info?location=a
# a = 방의 위치(카메라의 번호), default=0

api.add_resource(SendRooms, '/room_info')
# /room_info

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run()
