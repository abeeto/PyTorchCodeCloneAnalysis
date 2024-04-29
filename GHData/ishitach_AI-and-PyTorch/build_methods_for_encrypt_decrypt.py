import random

Q = 2345678909876543456789

def encrypt(x, n_shares=3):
  shares = list()
  for i in range(n_shares-1):
    shares.append(random.randint(0,Q))
    
  final_share = Q- (sum(shares)%Q) + x
  shares.append(final_share)
  return tuple(shares)
  
def decrypt(shares):
  return sum(shares)%Q

def add(a,b):
  c = list()
  assert (len(a) == len(b))
  
  for i in range(len(a)):
    c.append((a[i] + b[i])%Q)
    
  return tuple(c) 
  
decrypt(add(encrypt(5), encrypt(10)))

x= encrypt(5)
y = encrypt(10)
z = add(x,y)
z
#(2237305801742825267720, 117696406553744357984, 2336355611456517287889)

decrypt(z)
#15

#even after random values of z
