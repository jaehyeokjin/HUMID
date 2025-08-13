set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Angle ({/Symbol q})'
set ylabel 'Normalized Frequency'
set format y "%1.1t{/Symbol \264}10^{%L}"
set yr [:0.025]

pl "aa.out" u 1:2 w l lw 9 lc rgb 'red' title 'All-Atom', "trim.out" every 2 u 1:2 w l lw 9 lc rgb '#fa8072' title 'HUMID', "all_long.out" every 2 u 1:2 w l lw 7 dt '...' lc rgb '#1e90ff' notitle

set key Left
set key spacing 1.5

pl "aa.out" u 1:2 w l lw 9 lc rgb 'red' title 'H_3O^+ (All-Atom)', "trim.out" every 2 u 1:2 w l lw 9 lc rgb '#fa8072' title 'H_3O^+ (HUMID)', "all_long.out" every 2 u 1:2 w l lw 7 dt '...' lc rgb '#1e90ff' title 'H_2O  (All-Atom)'
