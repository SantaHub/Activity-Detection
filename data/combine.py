
outfilename='data.csv'
with open(outfilename, 'w') as outfile:
    for i in range(1,8):
        fname='data'+str(i)+'.csv'
        with open(fname, 'r') as readfile:
            outfile.write(readfile.read() + "")

