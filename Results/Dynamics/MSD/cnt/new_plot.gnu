set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Time (ps)'
set ylabel 'MSD ({\305}^2)'
set xr [0:10]
set key spacing 1.5
set key right
set key Left
set key maxrows 2
set xr [:4]
set yr [:8]
set xtics 1
pl "11.out" u ($1/1000):2 w l lw 9 lc rgb '#1e90ff' title 'H_2O  (CNT)', "humid-250.out" u ($1/1000):2 w l lw 9 lc rgb '#fa8072' title 'H_3O^+ (CNT) ', "../../../humid/Dynamics/MSD/HUMID/11.out" u ($1/1000):($2/3) w l lw 7 dt '...' lc rgb 'blue' notitle, "/Users/jaehyeok/humid/Dynamics/MSD/final-HUMID/cg/msd_output.txt" u ($1/1000):($2/3) w l dt '...' lw 7 lc rgb 'red' notitle

