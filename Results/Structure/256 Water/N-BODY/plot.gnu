set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Local Number Density'
set ylabel 'Normalized Frequency'
set xr [6:20]
set xtics 2
set yrange [:10.5]
set style fill transparent solid 0.4

pl "ucg.out" u 1:($2/5000) w filledcurves lw 9 lc rgb '#1e90ff' title 'BUMPer CG', "aa.out" u 1:($2/5000) w l lw 7 lc rgb '#c71eff' dt '...' title 'All-Atom (Ref.)'
