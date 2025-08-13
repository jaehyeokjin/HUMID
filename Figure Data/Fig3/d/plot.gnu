set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Time (ps)'
set ylabel 'MSD ({\305}^2)'
set yr [0:50]
set xr [0:10]
set ytics 5
set key spacing 1.5
set key right
set key Left
pl "11.out" u ($1/1000):2 w l lw 9 lc rgb '#1e90ff' title 'H_2O  ', "250.out" u ($1/1000):2 w l lw 9 lc rgb '#fa8072' title 'H_3O^+', "../FG/New/11.out"  u ($1/1000):2 w l lw 7 lc rgb 'blue' dt '...' notitle, "../FG/New/22.out"  u ($1/1000):2 w l lw 7 lc rgb 'red' dt '...' notitle
