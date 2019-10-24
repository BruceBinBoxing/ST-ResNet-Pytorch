#!/usr/bin/env python
# encoding: utf-8
import sys
import cPickle as pickle
def view(fname):
    pkl = pickle.load(open(fname, 'rb'))
    for ke in pkl.keys():
         print '=' * 10
         print ke
         print pkl[ke]
view(sys.argv[1])
