from align import receptor_aligner

aln = receptor_aligner.AlignerCDR3()

print(aln.pad("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))
print(aln.pad("CELFF", "CFF"))
print(aln.score("CASS", "CAARR"))