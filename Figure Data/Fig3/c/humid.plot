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
set key left
set key Left
gnuplot> pl "11_part.out" u ($1/500):2 w l lw 9 lc rgb '#8A2BE2' title 'H_2O: Naive CG',  "../FG/New/11.out"  u ($1/1000):2 w l lw 9 lc rgb 'blue' title 'H_2O: All-Atom', "../HUMID/11.out" u ($1/1000):2 w l lw 7 dt '...' lc rgb '#1e90ff' title 'H_2O: HUMID'
