# S3: embedding distances from real (Britanova) vs model (OLGA/Murugan) prototypes agree.
# Reads appendix/data/s3.tsv. Dark2 palette (gnuplot-palettes).
paldir = system("echo ${GNUPLOT_PALETTES:-../../gnuplot-palettes}")
load paldir.'/dark2.pal'

set terminal pdfcairo enhanced font "Helvetica,11" size 4.8in,4.6in
set output 'fig_s3_prototype_source.pdf'

r_s3 = real(system("grep '^r_s3' data/theory_stats.txt | cut -f2"))

set grid lc rgb '#e2e2e2'
set xlabel 'D_{ij}  (real prototypes)'
set ylabel 'D_{ij}  (model prototypes)'
set title sprintf('S3:  R = %.3f', r_s3)
set size ratio 1
plot 'data/s3.tsv' using 1:2 with points ls 3 pt 7 ps 0.15 notitle
