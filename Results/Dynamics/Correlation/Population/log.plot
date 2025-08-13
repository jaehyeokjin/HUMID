set terminal wxt size 1320,850 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Time (ps)'
set ylabel 'log(C_{HB} (t))'
set xr [0:100]
set xtics 20
set ytics format '%1.1f'

pl "aa.out" every 100 u ($1/1000):(log($2))  w l lw 9 lc rgb '#fa8072' title 'All-Atom', "continuous/250_long.out" u ($1/1000):(log($2))  w l lw 9 lc rgb '#1e90ff' title 'HUMID'
