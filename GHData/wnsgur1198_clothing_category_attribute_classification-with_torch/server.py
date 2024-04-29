# -*- coding: UTF-8 -*-
import asyncio
import websockets
from get_img import GetImage

img = GetImage()

async def accept(websocket, path):
    while True:
        data = await websocket.recv();
        #print(data)
        url = data[0:4]
        #print(url)
        
        if url == 'http':
            data2 = img.getImage(data);  
            #print("receive : " + data2); 
            await websocket.send(data2);            
        else:
            print("receive : " + data);
            await websocket.send(data);
            #await websocket.send("echo : " + data);
		
# 웹 소켓 서버 생성.호스트는 localhost에 port는 8003로 생성한다.
start_server = websockets.serve(accept, "0.0.0.0", 8003);

# 비동기로 서버를 대기한다.
asyncio.get_event_loop().run_until_complete(start_server);
asyncio.get_event_loop().run_forever();
