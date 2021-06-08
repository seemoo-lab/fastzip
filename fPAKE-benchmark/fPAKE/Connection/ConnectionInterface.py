
class ConnectionInterface:
    def connect(self):
        raise NotImplementedError()
    def wait_for_connection(self):
        raise NotImplementedError()
    def send(self, args):
        raise NotImplementedError()
    def receive(self):
        raise NotImplementedError()
    def log(self,args):
        raise NotImplementedError()
    def close (self):
        raise NotImplementedError()

class NotAConnectionInterfaceException(Exception):
    pass
class CouldNotConnectException(Exception):
    pass
class CoudNotReadException(Exception):
    pass
