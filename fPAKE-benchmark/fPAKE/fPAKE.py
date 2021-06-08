from LiPAKE.liPAKE import LiPake
from cryptography.hazmat.primitives.asymmetric import ed25519, ed448
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import algorithms, modes
from cryptography.hazmat.primitives.asymmetric import x448, x25519, ec
from cryptography import exceptions
import RSS.RSSCodes
from Connection.ConnectionInterface import ConnectionInterface, NotAConnectionInterfaceException, \
    CouldNotConnectException
import os
from Connection.IPConnection import IPConnection
from Connection.Connection import Connection
from threading import Thread
from helper.timer import timer
import numpy as np


class fPAKE:
    """
    Fuzzy Password authenticated Key exchange Protocol
    """

    def __init__(self, weakPW: str = "10101010101", securityParam: int = 1, connection=None):
        """
        Initiates the fPAKE protocol with a given Password;
        n Will be the length of the given Password
        :param weakPW: The Password as binary String like "010101" to iterate over it. The passwords need to be of same length
        :param n: security parameter n in bytes default is 1 which is 256 bit |0:128 bit securuty,  1 : 256 bit security
        :param connection: a Connection to send and receive data (needs to implement the @ConnectionInterface) or use the @IPConnection
        """
        self.pw = weakPW
        if (connection != None and isinstance(connection, ConnectionInterface)):
            self.connection = connection
        else:
            raise NotAConnectionInterfaceException
        self.VK = None
        if securityParam == 1:
            self.hash = hashes.SHA256()
            self.symmAlgo = algorithms.AES
            self.mode = modes.CBC
            self.curve = x448
            self.ecpub = x448.X448PublicKey
            self.ed = ed448.Ed448PrivateKey
            self.edpub = ed448.Ed448PublicKey
            self.keySize = 32  # Keysize of the Curve448 public Key
            self.n = 32  # securityBits 256
        if securityParam == 0:
            self.hash = hashes.SHAKE128(16)
            self.symmAlgo = algorithms.AES
            self.mode = modes.CBC
            self.curve = x25519
            self.ecpub = x25519.X25519PublicKey
            self.ed = ed25519.Ed25519PrivateKey
            self.edpub = ed25519.Ed25519PublicKey
            self.keySize = 16  # Keysize of the Curve25519 public Key needed for symmetirc encryption to set the correct Block size so no padding is needed
            self.n = 16  # Security bits 128

    def init_Protocol(self, benchmark=None):
        """
        The initiator runs this protocol to exchange a symmetric key with the other party which has a similar password
        with upto k different bytes
        :return: the negotiated key with the other party if at least k bits of the password were the same
        """
        time = None
        c_time = 0
        c_time_total = 0
        n_time = 0
        n_time_total = 0
        if not self.connection.connect():
            raise CouldNotConnectException()
        if benchmark is not None:
            time = timer()

        # Generate signingKey and get Verification key bytes to send
        signingKey = self.ed.generate()
        vk = signingKey.public_key()
        vkBytes = vk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        if time != None: # Only for benchmarking
            c_time_total = time.stop_time()
            benchmark["VK_generation"] = c_time_total
            benchmark["LiPAKE"] = {}

        #Prepare for @LiPake exchange
        Ki = []
        key_array = b""
        # Execute @LiPake over each bit of the password
        for i in self.pw:

            if time is not None:  # This is only for benchmarking
                c_time = 0
                n_time = 0
                time.start_time()

            iv = os.urandom(16)  # We use fixed size of AES256 cause we can't get 128 Bit Keys with KDF
            lp = LiPake(pw=i, label=vkBytes, iv=iv, Hash=self.hash, mode=self.mode, curve=self.curve,
                        symmAlgo=self.symmAlgo, symmetricKeySize=self.keySize, n=self.n)
            Xs, l = lp.getX()

            if time is not None:  # This is only for benchmarking
                c_time += time.stop_time()
                time.start_time()

            # Send generated X receive Y
            self.connection.send((Xs, l, iv))
            Y, l = self.connection.receive()

            if time is not None:  # This is only for benchmarking
                n_time = time.stop_time()
                time.start_time()

            k = lp.getKey(Y, l, self.ecpub, False)
            Ki.append(k)
            key_array += k

            if time is not None:  # This is only for benchmarking
                c_time += time.stop_time()
                benchmark["LiPAKE"]["run{:02d}".format(Ki.__len__())] = {"bit": i,
                                                                         "crypto_time": c_time,
                                                                         "network_time": n_time}
                c_time_total += c_time
                n_time_total += n_time
                time.start_time()

        # We use robust shamir secret sharing with reed solomon error correcting codes.
        # each key from lipake is 32 bit and we have 32 keys -> 32 * 32 will be the size of C
        secretkey = os.urandom(self.n)
        rss = RSS.RSSCodes.robustShamir(self.pw.__len__(), 1, size=self.n)
        secretkey, C = rss.shamir_share(secretkey)
        E = []
        # E = C-K
        for i in range(self.pw.__len__()):
            E.append(RSS.RSSCodes.XORBytes(C[i], Ki[i]))
        #Sign our E with the secret key
        sig = signingKey.sign(RSS.RSSCodes.list_to_byte(E))

        if time is not None:  # Only for benchmarking
            c_time = time.stop_time()
            benchmark["rss_calc_time"] = c_time
            c_time_total += c_time
            time.start_time()
        #Send E + Signature + verification key + selected prime number to reconstruct
        self.connection.send((E, sig, vkBytes, rss.get_prime()))
        response = self.connection.receive()
        while response != "accepted":
            #print((E, sig, vkBytes, rss.get_prime()))
            self.connection.send((E, sig, vkBytes, rss.get_prime()))
            response = self.connection.receive()

        if time is not None:  # only for benchmarking purpose
            n_time = time.stop_time()
            benchmark["rss_network_time"] = n_time
            n_time_total += n_time
            benchmark["total_calculation_time"] = c_time_total
            benchmark["total_network_time"] = n_time_total
        # Finish the protocol to get both sides to end the protocol and close connections
        if self.connection.receive() == "finalize":
            self.connection.send("finalize")
        self.connection.close()
        return (secretkey)

    def receive_protocol(self, benchmark=None):
        """
        Protocol the receiving end is running to exchange a symmetric key if the password of each party is similar
        :return: the negotiated key if password was similar to a certain degree
        """
        time = None
        n_time = 0
        n_time_total = 0
        c_time = 0
        c_time_total = 0
        if benchmark is not None:  # For Benchmarking only
            benchmark["LiPAKE"] = {}
            time = timer()
        if not self.connection.wait_for_connection():
            raise CouldNotConnectException()
        labelList = []
        Ki = []
        key_array = b""
        # LiPake iteration for each bit in password
        for i in self.pw:
            if time is not None:  # For Benchmarking only
                n_time = 0
                c_time = 0
                time.start_time()
            # get Init Vectors as well as labels and X_s for LiPake
            Xs, l1, iv = self.connection.receive()

            if time is not None:  # For Benchmarking only
                n_time += time.stop_time()
                time.start_time()

            if (self.VK == None):
                try:
                    self.VK = self.edpub.from_public_bytes(l1)
                except:
                    print("VK was not correct ! Abort the protocol")
                    raise NotAVerificationKeyException
            labelList.append(l1)
            lp = LiPake(pw=i, label=b"", iv=iv, Hash=self.hash, mode=self.mode, curve=self.curve,
                        symmAlgo=self.symmAlgo, symmetricKeySize=self.keySize, n=self.n)
            # Generate receivers Y_s
            Ys, l2 = lp.getX()

            if time is not None:  # For Benchmarking only
                c_time += time.stop_time()
                time.start_time()
            # Send Y_s with its label
            self.connection.send((Ys, l2))
            if time is not None:  # For Benchmarking only
                n_time += time.stop_time()
                time.start_time()

            k = lp.getKey(Xs, l1, self.ecpub, True)
            Ki.append(k)
            key_array += k

            if time is not None: # For Benchmarking only
                c_time += time.stop_time()
                c_time_total += c_time
                n_time_total += n_time
                benchmark["LiPAKE"]["run{:02d}".format(Ki.__len__())] = {"bit": i,
                                                                         "crypto_time": c_time,
                                                                         "network_time": n_time}
                time.start_time()

        accepted = False
        while not accepted:
            try:
                E, sig, vk, prime = self.connection.receive()
                accepted = True
                self.connection.send("accepted")
            except:
                self.connection.send("Failed")
                print("Failed retrying")
                # E, sig, vk, prime = self.connection.receive()

        if time is not None: # For Benchmarking only
            n_time = time.stop_time()
            benchmark["rss_network_time"] = n_time
            n_time_total += n_time
            time.start_time()
        # Create RSS scheme with prime and password
        rss = RSS.RSSCodes.robustShamir(self.pw.__len__(), 1, size=self.n, PRIME=prime)
        # Reconstruct Verification key of bytes and verify the send E
        if (vk != self.VK.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)):
            raise exceptions.InvalidKey()
            # print("VK is worng")
        try:
            self.VK.verify(sig, RSS.RSSCodes.list_to_byte(E))
        except:  # Cancel if signature if wrong
            self.connection.send("finalize")
            if self.connection.receive() == "finalize":
                self.connection.close()
            return os.urandom(self.n)
        C = []
        # Calculate C by trying to revers the XOR. If enough Kis are correct we can reconstruct the shared secret with RSS below
        for i in range(self.pw.__len__()):
            C.append(RSS.RSSCodes.XORBytes(E[i], Ki[i]))
        try:
            # use RSS to reconstruct secret key if enough Kis were correct
            U = rss.shamir_robust_reconstruct(C)
        except:
            # If RSS was not successful the key is random
            U = (os.urandom(self.n))
        if time is not None:  # For Benchmarking only
            c_time = time.stop_time()
            benchmark["rss_calc_time"] = c_time
            c_time_total += c_time
            benchmark["total_calculation_time"] = c_time_total
            benchmark["total_network_time"] = n_time_total

        # Close the connection and tell the other party to close the connection
        self.connection.send("finalize")
        if self.connection.receive() == "finalize":
            self.connection.close()

        return U


class NotAVerificationKeyException(Exception):
    pass


def sender(connection=IPConnection()):
    print(fPAKE(connection=connection, securityParam=1, weakPW="10101001011").init_Protocol())
    print("Sender Done")


def receiver(connection=IPConnection()):
    print(fPAKE(connection=connection, securityParam=1, weakPW="10101010101").receive_protocol())
    print("receiver Done")

if __name__ == '__main__':
    import argparse
    # This is the command line application
    parser = argparse.ArgumentParser("Execute fuzzy password authenticated key exchange ")
    parser.add_argument("-ip",type=str,nargs=1,help="IP address of the target devices. If role is receiver the IP can be empty. If sender IP is required",default=["localhost"])
    parser.add_argument("-pw",type=str,required=True,nargs=1,help="binary password in the form of 101010101110 (need to be same length for both parties)")
    parser.add_argument("-role",choices=["initiator","receiver"],required=True,help="Sets the protocol to be either the initiating side or the receiving side")
    parser.add_argument("-sec_param",default=1,choices=["0","1"],help="1 : 256 bit security | 0 : 128 bit security")
    args = parser.parse_args()
    
    pake=  fPAKE(args.pw[0],securityParam=int(args.sec_param),connection=IPConnection(IP=args.ip[0]))
    if args.role == "receiver":
        print("Waiting for connection ...")
        print(pake.receive_protocol())
    else :
        print(pake.init_Protocol())
    
    """
    t2 = Thread(target=receiver)
    t1 = Thread(target=sender)
    t2.start()
    t1.start()

    t1.join()
    t2.join()
    print("Done")"""


