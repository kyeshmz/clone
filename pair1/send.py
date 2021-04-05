from pynng import Pair1

td_addr = "tcp://172.25.111.30:5001"
proj_addr = "tcp://172.25.157.35:5001"

s1 = Pair1()
s1.listen(td_addr)
s1.send(b'Well hello there')
s1.close()
