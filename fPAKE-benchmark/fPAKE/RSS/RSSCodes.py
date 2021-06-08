import random
import Crypto.Util.number as nb
import os
from copy import copy
"""
Thanks to https://mortendahl.github.io/2017/08/13/secret-sharing-part3/

"""

def list_to_byte(list):
    acc = b""
    for b in list:
        acc+=b
    return acc

def bytes_to_int(bytes):
    return int.from_bytes(bytes,"little")

def int_to_bytes(integer,n):
    return integer.to_bytes(n,"little")

def XORBytes(b1,b2):
    return int_to_bytes(bytes_to_int(b1)^bytes_to_int(b2),b1.__len__())


class robustShamir:
    """
    Robust Shamir Secret Sharing implementation from https://mortendahl.github.io/2017/08/13/secret-sharing-part3/
    """
    def __init__(self, N, T, K=1, size=32, PRIME=None):
        """
        :param N: Number of shares to be created
        :param T: privacy Threshold = Max number of shares that may be seen without learning anything about the secret
        :param K: Number of secrets | This should be set to 1
        :param size: Security size / key Size / Size for the Prime Number
        :param PRIME: Takes the given prime to create the prime field. If prime = None pime is chosen by
            Crypto.Util.number.getPrime((size*8), os.urandom)
        """
        self.size = size
        if PRIME is None: #If no prime is given we generate one with the given size
            self.PRIME = nb.getPrime((size*8), os.urandom) #prime Size is in bits so bytes*8
        else:
            self.PRIME = PRIME
        self.K = K  # Number of secrets
        self.N = N  # Number of shares
        self.R = T+self.K  # Min number of shares required to reconstruct
        self.T = T  # Threshold = the maximum number of shares that may be seen without learning nothing about the secret, also known as the privacy threshold
        assert(self.R <= self.N)
        self.MAX_MANIPULATED = int((self.N-self.R)/2)
        #print("correction capability: ", self.MAX_MANIPULATED)

        #assert(self.R + self.MAX_MISSING + 2*self.MAX_MANIPULATED <= self.N)

        self.POINTS = [p for p in range(1, self.N+1)]

    def get_prime(self):
        return self.PRIME

    def shamir_share(self, secret):
        secret = bytes_to_int(secret)
        if secret > self.PRIME: #make sure secret is in group
            secret = secret % self.PRIME
        polynomial = [secret] + [random.randrange(self.PRIME) for _ in range(self.T)]
        shares = [int_to_bytes(self.poly_eval(polynomial, p), self.size) for p in self.POINTS]
        return int_to_bytes(secret,self.size), shares


    def shamir_robust_reconstruct(self,shares):
        # filter missing shares // Not needed for our case since we have all keys but not all are correct
        points_values = [(p, bytes_to_int(v)) for p, v in zip(self.POINTS, shares) if v is not None]

        # decode remaining faulty
        points, values = zip(*points_values)
        polynomial, error_locator = self.gao_decoding(points, values, self.R, self.MAX_MANIPULATED)

        # check if recovery was possible
        if polynomial is None: raise Exception("Too many errors, cannot reconstruct")

        # recover secret
        secret = self.poly_eval(polynomial, 0)

        # possible to find faulty indicies but we dont want that
        #error_indices = [i for i, v in enumerate(self.poly_eval(error_locator, p) for p in self.POINTS) if v == 0]
        return int_to_bytes(secret,self.size) #, error_indices

    def gao_decoding(self,points, values, max_degree, max_error_count):
        """
        Gao's Reed Solomon
        """
        #assert (len(values) == len(points))
        #assert (len(points) >= 2 * max_error_count + max_degree)

        # interpolate faulty polynomial
        H = self.lagrange_interpolation(points, values)

        # compute f
        F = [1]
        for xi in points:
            Fi = [self.base_sub(0, xi), 1]
            F = self.poly_mul(F, Fi)

        # run EEA-like algorithm on (F,H) to find EEA triple
        R0, R1 = F, H
        S0, S1 = [1], []
        T0, T1 = [], [1]
        while True:
            Q, R2 =self.poly_divmod(R0, R1)

            if self.deg(R0) < max_degree + max_error_count:
                G, leftover = self.poly_divmod(R0, T0)
                if leftover == []:
                    decoded_polynomial = G
                    error_locator = T0
                    return decoded_polynomial, error_locator
                else:
                    return G, T0

            R0, S0, T0, R1, S1, T1 = \
                R1, S1, T1, \
                R2, self.poly_sub(S0,self.poly_mul(S1, Q)), self.poly_sub(T0, self.poly_mul(T1, Q))

    def lagrange_interpolation(self,xs, ys):
        ls = self.lagrange_polynomials(xs)
        poly = []
        for i in range(len(ys)):
            term = self.poly_scalarmul(ls[i], ys[i])
            poly = self.poly_add(poly, term)
        return poly

    def lagrange_polynomials(self,xs):
        polys = []
        for i, xi in enumerate(xs):
            numerator = [1]
            denominator = 1
            for j, xj in enumerate(xs):
                if i == j: continue
                numerator   = self.poly_mul(numerator, [self.base_sub(0, xj), 1])
                denominator = self.base_mul(denominator, self.base_sub(xi, xj))
            poly = self.poly_scalardiv(numerator, denominator)
            polys.append(poly)
        return polys

    def poly_scalarmul(self,A, b):
        return self.canonical([self.base_mul(a, b) for a in A ])

    def poly_scalardiv(self,A, b):
        return self.canonical([self.base_div(a, b) for a in A ])

    def canonical(self,A):
        for i in reversed(range(len(A))):
            if A[i] != 0:
                return A[:i+1]
        return []

    def deg(self,A):
        return len(self.canonical(A)) - 1


    def lc(self,A):
        B = self.canonical(A)
        return B[-1]

    def expand_to_match(self,A, B):
        diff = len(A) - len(B)
        if diff > 0:
            return A, B + [0] * diff
        elif diff < 0:
            diff = abs(diff)
            return A + [0] * diff, B
        else:
            return A, B

    def poly_divmod(self,A, B):
        t = self.base_inverse(self.lc(B))
        Q = [0] * len(A)
        R = copy(A)
        for i in reversed(range(0, len(A) - len(B) + 1)):
            Q[i] = self.base_mul(t, R[i + len(B) - 1])
            for j in range(len(B)):
                R[i+j] = self.base_sub(R[i+j], self.base_mul(Q[i], B[j]))
        return self.canonical(Q), self.canonical(R)

    def poly_add(self,A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_add(f, g) for f, g in zip(F, G) ])

    def poly_sub(self,A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_sub(f, g) for f, g in zip(F, G) ])

    def poly_mul(self,A, B):
        C = [0] * (len(A) + len(B) - 1)
        for i in range(len(A)):
            for j in range(len(B)):
                C[i+j] = self.base_add(C[i+j], self.base_mul(A[i], B[j]))
        return self.canonical(C)

    def poly_eval(self,A, x):
        result = 0
        for coef in reversed(A):
            result = self.base_add(coef, self.base_mul(x, result))
        return result

    def base_add(self,a, b):
        return (a + b) % self.PRIME

    def base_sub(self,a, b):
        return (a - b) % self.PRIME

    def base_inverse(self,a):
        _, b, _ = self.base_egcd(a, self.PRIME)
        return b if b >= 0 else b+self.PRIME

    def base_mul(self,a, b):
        return (a * b) % self.PRIME


    def base_div(self,a, b):
        return self.base_mul(a, self.base_inverse(b))

    def base_egcd(self,a, b):
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1

        while r1 != 0:
            q, r2 = divmod(r0, r1)
            r0, s0, t0, r1, s1, t1 = \
                r1, s1, t1, \
                r2, s0 - s1 * q, t0 - t1 * q

        d = r0
        s = s0
        t = t0
        return d, s, t

"""
rss = robustShamir(15, 5)

secret = os.urandom(32)
secret, original_shares = rss.shamir_share(secret)
recovered_secret = rss.shamir_robust_reconstruct(original_shares)

received_shares = list()#copy(original_shares)
received_shares.append(original_shares[0])
received_shares.append(original_shares[1])
received_shares.append(original_shares[2])
received_shares.append(original_shares[3])
received_shares.append(original_shares[4])
received_shares.append(original_shares[5])
received_shares.append(original_shares[6])
received_shares.append(original_shares[7])
received_shares.append()
received_shares.append(b"0")
received_shares.append(b"0")
received_shares.append(b"0")
received_shares.append(b"0")
received_shares.append(b"0")
received_shares.append(b"0")
#print(original_shares)
#received_shares [0] = XORBytes(received_shares[0],received_shares[1])
#received_shares [1] = XORBytes(received_shares[1],received_shares[2])
#received_shares [2] = XORBytes(received_shares[2],received_shares[3])
#received_shares [3] = XORBytes(received_shares[3],received_shares[4])
#received_shares [4] = XORBytes(received_shares[4],received_shares[5])
#received_shares [5] = XORBytes(received_shares[5],received_shares[6])
#print(received_shares)

# robust reconstruction
recovered_secret = rss.shamir_robust_reconstruct(received_shares)
print(recovered_secret)
print(secret)
print(recovered_secret == secret)
"""