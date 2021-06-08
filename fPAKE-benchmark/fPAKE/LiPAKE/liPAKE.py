from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives.asymmetric import ec, x25519, x448
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_der_public_key
import os


def getPrefix():
    return b'0Z0\x14\x06\x07*\x86H\xce=\x02\x01\x06\t+$\x03\x03\x02\x08\x01\x01\x07\x03B\x00\x04'


class LiPake:
    def __init__(self, curve=None, symmAlgo=algorithms.AES, mode=modes.CBC, iv=os.urandom(16),
                 Hash=hashes.SHA256(), pw="1", label="label1".encode(), symmetricKeySize=256, n = 32):
        if (curve == x25519):
            self.parameter = x25519
            self.secret = x25519.X25519PrivateKey.generate()
        if curve == x448:
            self.parameter = x448
            self.secret = x448.X448PrivateKey.generate()  # Generates x
        self.pw = pw
        self.label = label
        self.symmetricKeySize = symmetricKeySize
        self.kdfSize = symmetricKeySize
        self.HashType = Hash
        hkdf = HKDFExpand(hashes.SHA3_256(), 32, None, default_backend())
        k = pw.encode() + label
        self.key = hkdf.derive(k)
        self.mode = mode(iv)
        self.symmAlgo = symmAlgo
        self.n = n
        self.Hash = hashes.Hash(Hash, default_backend())

    def getX(self):
        # Encrypt public key g^x with symm enc key
        encryptor = Cipher(self.symmAlgo(self.key), self.mode, default_backend()).encryptor()
        if (self.parameter == x25519):
            gx = self.secret.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        else:
            gx = self.secret.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)+os.urandom(8) # Must be a multiple from 32 byte with curve 448 we add random 8 byteso to get 64 byte to encrypt without padding
        # return both encrypted key and label
        return encryptor.update(gx) + encryptor.finalize(), self.label

    def getKey(self, Y, l, e_curve, receiver=False):
        # Returns the Hash(X,Y,Z), if reciever = True -> Hash(Y,X,Z)
        hkdf = HKDFExpand(hashes.SHA3_256(), 32, None, default_backend())
        key = hkdf.derive(self.pw.encode() + l)
        decryptor = Cipher(self.symmAlgo(key), self.mode, default_backend()).decryptor()
        ## decrypt then pow wiht x
        gy = decryptor.update(Y) + decryptor.finalize()
        if e_curve == x25519.X25519PublicKey:
            Y = e_curve.from_public_bytes(gy)
        else :
            Y = e_curve.from_public_bytes(gy[0:56])
        Ybytes = Y.public_bytes(Encoding.Raw, PublicFormat.Raw)
        Xbytes = self.secret.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        Z = self.secret.exchange(Y)  # Calc Y^x
        # Calculate Hash corresponding to being sender or receiver to put in values in correct order for has h function
        if receiver:
            self.Hash.update(Ybytes)
            self.Hash.update(Xbytes)
            self.Hash.update(Z)
        else:
            self.Hash.update(Xbytes)
            self.Hash.update(Ybytes)
            self.Hash.update(Z)
        k = self.Hash.finalize()
        return k
