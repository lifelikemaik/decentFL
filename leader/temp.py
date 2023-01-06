import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 500)
socket.connect("tcp://localhost:5549")
socket.send(b"Hi")
message = socket.recv()
# print(f"[ {message} ]")
message.decode()
