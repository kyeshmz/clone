# load library
import spout as py_spout

def main() :
    # create spout object
    spout = py_spout.Spout(silent = False)
    # create receiver
    spout.createReceiver('input')
    # create sender
    spout.createSender('output')

    while True :

        # check on close window
        spout.check()
        # receive data
        data = spout.receive()
        # send data
        data = 255 - data
        print(data)
        spout.send(data)

if __name__ == "__main__":
    main()