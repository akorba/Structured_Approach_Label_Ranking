# -*- coding: utf-8 -*-
from itertools import *
import numpy as np
from math import *
from pandas import Series


#################
#   Embeddings  #
#################

# Borda 
def BordaEmbed(sigma):
    n=len(sigma)
    phi=np.zeros(n)
    for i in range(1,n+1):
        phi[i-1]=sigma.index(i)+1
    return phi
    
# Kemeny    
    
def KemenyEmbed(sigma):
    n=len(sigma)
    phi=np.zeros((n*(n-1)/2))
    count=0
    for (i,j) in combinations(range(1,n+1),2):
        phi[count]=np.sign(sigma.index(j)-sigma.index(i))
        count+=1
    return phi

def KemenyInvert_unisign(phi):
    n = int(np.ceil(np.sqrt(2*len(phi))))
    sigma_count = np.zeros([n])
    count = 0
    phi_signed = 2*phi - 1
    for (i,j) in combinations(range(n),2):
        sigma_count[j] += phi_signed[count]
        sigma_count[i] -= phi_signed[count]
        count +=1
    sigma_inv = np.argsort(np.argsort(sigma_count))
    return sigma_inv

def KemenyEmbed_no_offset_unisign(sigma):
    n=len(sigma)
    phi=np.zeros((n*(n-1)/2))
    count=0
    for (i,j) in combinations(range(n),2):
        phi[count]=int(np.sign(sigma[j]-sigma[i])>0)
            #if sigma.index(j) == sigma.index(i):
            #phi[count] = .5
            #else:
            #phi[count] = int((sigma.index(j) - sigma.index(i))>0)
        count+=1
    return phi

def KemenyEmbed_no_offset_unisign_partial(sigma):
    # a completer ---
    n=len(sigma)
    phi=np.zeros((n*(n-1)/2))
    count=0
    for (i,j) in combinations(range(n),2):
        #phi[count]=np.sign(sigma.index(j)-sigma.index(i))
        if sigma[j] == sigma[i]:
            phi[count] = .5
        else:
            phi[count] = int((sigma[j] - sigma[i])>0)
        count+=1
    return phi


def e(k,n):
    if k>n:
        raise Exception
    ek=np.zeros((n*(n-1)/2))
    count=0
    for (i,j) in combinations(range(1,n+1),2):
        if i==k:
            ek[count]=1
        if j==k:
            ek[count]=-1
        count+=1
    return ek
    
    
def PK1(x,n):
    proj=np.zeros((n*(n-1)/2))
    for k in range(1,n+1):
        proj+=(1/(n*1.0))*(np.dot(x,e(k,n)))*e(k,n)
    return proj

# Hamming

def HammingEmbed(sigma):    
    phi=np.zeros((len(sigma),len(sigma)))
    for i in sigma:
        j=sigma.index(i)
        phi[i-1,j]+=1
    return phi        

def E(k,n):
    Ek=np.zeros((n,n))
    if k>n:
        raise Exception
    for j in range(1,n+1):
        Ek[k-1][j-1]=j
    return Ek
    
def PH1(x,n):
    proj=np.zeros((n,n))
    for k in range(1,n+1):
        proj+=(6/(n*(n+1)*(2*n+1)*1.0))*(np.trace(np.dot(np.transpose(x),E(k,n))))*E(k,n)
    return proj

def vectorize_hamming_labels(list_of_lists):
    return Series(np.asarray(list_of_lists).ravel())

def Hamming_to_sigma(hamming_mat):
    return 1

# Lehmer

def encode_lehmer(sigma):
    n = len(sigma)
    c = []
    c.append(0)
    for x in range(1, n):
        sigma_x = sigma[x]
        c_x = 0
        for y in range(0, x):
            sigma_y = sigma[y]
            if sigma_y >= sigma_x:
                c_x += 1
        c.append(c_x)
    c = tuple(c)
    return c

def inverse_lehmer(sigma_1):
    n=len(sigma_1)
    sigma=[0]*n
    for x in range(0,n):
        s=sigma_1[x]
        sigma[s-1]=x+1
    return tuple(sigma)

def decode_lehmer(code):
    n = len(code)
    code = [i + 1 - code[i] for i in range(0, n)]  # wiki
    # sigma=np.zeros(n)
    sigma = [0] * n
    free_positions = range(0, n)
    for x in range(n - 1, -1, -1):
        c_x = code[x]
        position = free_positions[c_x - 1]
        sigma[position] = x + 1
        free_positions.remove(position)
    sigma = inverse_lehmer(sigma)
    return tuple(sigma)