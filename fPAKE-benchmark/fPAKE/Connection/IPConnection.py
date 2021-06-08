import socket
from Connection.ConnectionInterface import ConnectionInterface
from Connection.ConnectionInterface import CoudNotReadException
import pickle
import time


class IPConnection(ConnectionInterface):
    def __init__(self, IP="localhost", PORT = 10000):
        self.address = (IP, PORT)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
        self.connection = None
        self.info = None

    def connect(self):
        while True:
            try:
                self.socket.connect(self.address)
                #self.socket.setblocking(0)
                break
            except:
                time.sleep(0.1)
        return True

    def wait_for_connection(self):
        self.socket.bind(self.address)
        self.socket.listen(1)
        self.socket, self.info = self.socket.accept()  # should be blocking till connection is available
        #self.socket.setblocking(0)
        return True

    def send(self, args):
        data = pickle.dumps(args)
        if args == None:
            print("Data to send was none :D")
        #self.logger.log(("Send:",args))
        self.socket.sendall(data)


    def receive(self):
        data = self.socket.recv(8192)
        try :
            tmp = pickle.loads(data)
            return tmp
        except:
            return self._reread(data,1)

    def log(self,args):
        self.logger.log(args)

    def close(self):
        self.socket.shutdown(socket.SHUT_WR)
        self.socket.close()

    def _reread(self, data, depth):
        if (depth == 15):
            raise CoudNotReadException("Coud not read data")
        try:
            data += self.socket.recv(8192)
            return pickle.loads(data)
        except:
            self._reread(data,depth+1)
