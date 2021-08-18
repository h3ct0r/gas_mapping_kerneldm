import sys
import os
import ctypes
import numpy

keyBits = ctypes.sizeof(ctypes.c_uint64) * 8
chrBits = ctypes.sizeof(ctypes.c_ubyte) * 8
indexBits = keyBits - chrBits
chrMask = numpy.uint64(numpy.int64(~numpy.ubyte(0)) << indexBits)
indexMask = ~chrMask

def split_pg_key(pg_key):
    c_ = chr(numpy.ubyte(numpy.int64(numpy.uint64(pg_key) & chrMask) >> indexBits))
    j_ = numpy.uint64(pg_key) & indexMask
    return c_, j_
# In [79]: split_pg_key(7061644215716937728)
# Out[79]: ('b', 0)

def join_pg_key(c_, j_):
    return numpy.uint64(ord(c_) << indexBits) + numpy.uint64(j_)
# In [8]: join_pg_key('b',0)
# Out[8]: 7061644215716937728

prefixToRobot = {'a': "husky1",
                 'b': "husky2",
                 'c': "husky3",
                 'f': "husky4",
                 'g': "spot1",
                 'h': "spot2",
                 'z': "fixed"}
    
def nodeKeyToRobot(key):
    #if key[0].isalpha():
    #print(key)
    # prefix = str(join_pg_key(key[0], int(key[1:])))
    # prefix = str(split_pg_key(str(key)))
    prefix = split_pg_key(str(key))[0]
    #*key_handling.split_pg_key(key), sep=''
    if prefix[0] in prefixToRobot.keys():
        return prefixToRobot[prefix[0]]
    else:
        #print("Did not recognize prefix", prefix, "given key", key)
        return None
