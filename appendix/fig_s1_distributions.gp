# S1 / T4: distribution laws. d_ij ~ Gamma (not Normal); D_ij ~ GEV/Frechet (not Normal).
# Reads appendix/data/s1_{diss,dist}_{hist,curves}.tsv. Dark2 palette for the fitted curves.
paldir = system("echo ${GNUPLOT_PALETTES:-../../gnuplot-palettes}")
load paldir.'/dark2.pal'

set terminal pdfcairo enhanced font "Helvetica,11" size 9in,4.2in
set output 'fig_s1_distributions.pdf'

ks_dg = real(system("grep '^ks_d_gamma' data/theory_stats.txt | cut -f2"))
ks_Dg = real(system("grep '^ks_D_gev'   data/theory_stats.txt | cut -f2"))
ks_Dn = real(system("grep '^ks_D_normal' data/theory_stats.txt | cut -f2"))
xi    = real(system("grep '^xi'         data/theory_stats.txt | cut -f2"))

set multiplot layout 1,2
set style fill solid 0.30 noborder
set boxwidth 0.9 relative
set grid lc rgb '#e2e2e2'
set key top right
set ylabel 'density'

set title sprintf('d_{ij}:  Gamma  (KS %.3f)', ks_dg)
set xlabel 'dissimilarity  d_{ij}'
plot 'data/s1_diss_hist.tsv'   using 1:2 with boxes lc rgb '#c8c8c8' notitle, \
     'data/s1_diss_curves.tsv' using 1:2 with lines ls 1 lw 2.5 title 'Gamma', \
     'data/s1_diss_curves.tsv' using 1:3 with lines ls 3 lw 2.0 dt 2 title 'Normal'

set title sprintf('D_{ij}:  GEV ({/Symbol x}=%+.2f)  KS %.3f vs %.3f', xi, ks_Dg, ks_Dn)
set xlabel 'embedding distance  D_{ij}'
plot 'data/s1_dist_hist.tsv'   using 1:2 with boxes lc rgb '#c8c8c8' notitle, \
     'data/s1_dist_curves.tsv' using 1:2 with lines ls 1 lw 2.5 title 'GEV/Frechet', \
     'data/s1_dist_curves.tsv' using 1:3 with lines ls 3 lw 2.0 dt 2 title 'Normal'
unset multiplot
