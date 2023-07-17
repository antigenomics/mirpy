from mir.distances import aligner
from mir.common import segments
import mir

aln = aligner.AlignCDR()

print(aln.pad("CASSLAPGATNEKLFF", "CASSLATNEKLFF")[0] == ("CASSLAPGATNEKLFF", "CAS---SLATNEKLFF"))
print(aln.pad("CELFF", "CFF"))
print(aln.score("CASS", "CASS"))
print(aln.score("CASSS", "CASS-"))
print(aln.score("CASA", "CASS"))
print(aln.score("CASR", "CASR"))
print(aln.alns("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))
print(aln.score("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))
print(aln.score_norm("CASSLAPGATNEKLFF", "CASSLATNEKLFF"))

seqdict = {'trbv1': "AGGGA", 'trbv2': 'AEFGHHW', 'trbj1': 'FKLHW'}
print(aligner.AlignGermline.from_seqs(seqdict).dist)

print(mir.get_resource_path("segments.txt"))

libd = segments.Library.load_default()
print(libd)
print(libd.get_summary())