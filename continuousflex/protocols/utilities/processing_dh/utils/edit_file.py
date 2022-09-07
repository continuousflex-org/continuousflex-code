import os
import glob
import numpy as np

def edit(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i < 13:
                pass
            else:
                lines[i] = lines[i].replace(lines[i][0:72], 'synthimages_set/img'+lines[i][0:6]+'.spi')
        with open('../images.xmd', 'w') as output:
            for i in range(len(lines)):
                output.write(lines[i])
            output.close()



edit('../images.xmd')
