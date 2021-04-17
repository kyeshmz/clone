from pythonosc import udp_client
import random
client = udp_client.SimpleUDPClient('127.0.0.1', 2000)
while True:
    client.send_message("/filter", random.random())
    print('test')


