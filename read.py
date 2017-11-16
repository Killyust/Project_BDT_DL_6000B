import numpy
import os

file  = open('project2.txt','r+')
ids = open("project2_20459312.txt", "w+")

for item in file:
    item = item.strip('\n')
    item = item.replace(' ','')
    ids.write(item + "\n")

file.close()
ids.close()
