from align import receptor_aligner

aln = receptor_aligner.AlignCDR()

print(aln.pad("CASSLAPGATNEKLFF", "CASSLATNEKLFF")[0] == ("CASSLAPGATNEKLFF", "CAS---SLATNEKLFF"))
print(aln.pad("CELFF", "CFF"))
print(aln.score("CASS", "CASS"))
print(aln.score("CASSS", "CASS-"))
print(aln.score("CASA", "CASS"))
print(aln.score("CASR", "CASR"))
print(aln.alns("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))
print(aln.score("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))
print(aln.score_norm("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))

print(receptor_aligner.AlignGermline.from_seqs({'v1': "AGGGA", 'v2': 'AEFGHHW'}).dist)