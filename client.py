from pythonosc.udp_client import SimpleUDPClient

ip = "172.25.157.35"
port = 5003

client = SimpleUDPClient(ip, port)  # Create client

client.send_message("/create", 123)   # Send float message
client.send_message("/some/address", [1, 2., "hello"])  # Send message with int, float and string
