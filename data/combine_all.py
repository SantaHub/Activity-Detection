# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import shutil

#filenames= ['','','','','','','']
filenames='walking'




activities=['b1','b2','cycl','ly','sit','stand','walk']

outfilename='data.csv'
with open(outfilename, 'w') as outfile:
    for i in range(1,8):
        fname='data'+str(i)+'.csv'
        with open(fname, 'r') as readfile:
            for line in readfile.read().split('\n') :
                outfile.write(line.strip('\n')+","+activities[i-1]+"\n")

import pandas
data = pandas.read_csv('data.csv')

