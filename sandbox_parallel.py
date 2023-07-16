from multiprocessing import Pool
import textdistance

def f(d):
    d[1] += '1'
    d['2'] += 2

nproc = 4
chunk_sz = 1

def dist(s1 : str, s2 : str) -> tuple[str, str, float]:
    return (s1, s2, textdistance.hamming(s1, s2))

seqs = ["aaa", "aab", "abb", "bbb", "aba", "bba", "bab"]

if __name__ == '__main__':
    d = ()
    with Pool(nproc) as pool:
        d = pool.starmap(dist, ((s1, s2) for s1 in seqs for s2 in seqs if s1 > s2), chunk_sz)
    print(d)
    print(len(d))