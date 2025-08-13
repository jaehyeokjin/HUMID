set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Angle ({/Symbol q})'
set ylabel 'Normalized Frequency'
set yr [:4]
set format y "%1.0t{/Symbol \264}10^{%L}"
set xr [20:180]
set xtics 20
set yrange [:0.006]
pl "angular_dist.dat" u 1:2 w l lw 9 lc rgb '#1e90ff' title 'BUMPer CG', "aa/angular_dist.dat" u 1:2 w l lw 7 lc rgb '#c71eff' dt '...' title 'All-Atom (Ref.)'
