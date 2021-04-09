from pynng import Pair1
s1 = Pair1()

td_addr = "tcp://172.25.111.30:5001"
proj_addr = "tcp://172.25.157.35:5001"
# s1.dial(td_addr)
while True:
   s1.dial(td_addr)
   msg = s1.recv_msg()
   print(msg)
   content = msg.bytes.decode()
   print(content)


