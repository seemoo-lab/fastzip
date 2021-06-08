import time
import pickle
from Connection.ConnectionInterface import ConnectionInterface



class Connection (ConnectionInterface):
    def __init__(self, connection = None):
        self.received = False
        self.otherParty = connection
        self.data = None

    def connect(self):
        return True

    def wait_for_connection(self):
        return True

    def set_connection(self,connection):
        self.otherParty = connection

    def send(self, args):
        data = pickle.dumps(args)
        self.otherParty.getData(data)

    def getData(self, args):
        self.data = pickle.loads(args)
        self.received = True
    def receive(self):
        while True:
            if self.received:
                self.received = False
                return self.data
            time.sleep(0.1)

    def log(self, args):
        return

    def close(self):
        return



