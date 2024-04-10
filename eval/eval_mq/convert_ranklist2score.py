
import os
import sys
import random


if __name__ == '__main__':
    infile1 = sys.argv[1]
    rankfile = sys.argv[2]
    outfile = sys.argv[3]
    fout = open(outfile,'w')

    scoreinfo = {}
    for line in open(rankfile,'r'):
        r = line.split()
        qid = r[0]
        did = r[2]
        score = float(r[4])
        if qid not in scoreinfo:
            scoreinfo[qid] = {}
        if did not in scoreinfo[qid]:
            scoreinfo[qid][did] = score
    nopos = set()
    for line in open(infile1,'r') :
        r = line.split()
        label = int(r[0])
        qid = r[1].split(':')[-1]
        did = r[50]
        #assert qid in scoreinfo
        #assert did in scoreinfo[qid]
        if qid not in scoreinfo or did not in scoreinfo[qid]:
            #assert( label <= 0)
            fout.write('%f'%random.random())
            #fout.write('0')
        else:
            fout.write('%f'%scoreinfo[qid][did])
        fout.write('\n')
    fout.close()


