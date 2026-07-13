# S2 / T1: embedding distance D_ij vs dissimilarity d_ij, Smith-Waterman and gapblock.
# Reads appendix/data/s2_{sw,gapblock}.tsv (from gen_theory_data.py). Colours: ColorBrewer
# Dark2 via gnuplot-palettes (set $GNUPLOT_PALETTES, or clone to ../../gnuplot-palettes).
paldir = system("echo ${GNUPLOT_PALETTES:-../../gnuplot-palettes}")
load paldir.'/dark2.pal'

set terminal pdfcairo enhanced font "Helvetica,11" size 9in,4.2in
set output 'fig_s2_correlation.pdf'

r_sw = real(system("grep '^r_sw' data/theory_stats.txt | cut -f2"))
r_gb = real(system("grep '^r_gb' data/theory_stats.txt | cut -f2"))

set multiplot layout 1,2
set grid lc rgb '#e2e2e2'
set xlabel 'dissimilarity  d_{ij}'
set ylabel 'embedding distance  D_{ij}'

set title sprintf('Smith-Waterman:  R = %.3f', r_sw)
plot 'data/s2_sw.tsv' using 1:2 with points ls 1 pt 7 ps 0.15 notitle

set title sprintf('gapblock (v3):  R = %.3f', r_gb)
plot 'data/s2_gapblock.tsv' using 1:2 with points ls 2 pt 7 ps 0.15 notitle
unset multiplot
