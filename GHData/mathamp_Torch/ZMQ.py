import time
import zmq


def run():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    while True: # Wait for next request from client
        message = socket.recv()
        print(f"Received request: {message}")
        start = time.time()
        while True:
            if time.time() - start >= 1:
                break
        socket.send(b"World")
        print(time.time())
