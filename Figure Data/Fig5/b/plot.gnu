set terminal wxt size 450,400 enhanced font 'Whitney,16' persist

set border lw 1.3333
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Time (ps)'
set ylabel 'Relative MSD'
set xr [0:10]
set key spacing 1.5
set ytics 0.2
set key right
set key Left
set key left
pl "one-1.25.out" u ($1/1000):($2/39431.3) w l lw 3 lc rgb '#fa8072' title '  1 H_3O^+', "two-1.20.out" u ($1/1000):($2/39431.3) w l lw 3 lc rgb '#d5fb56' title '  2 H_3O^+', "four-1.14.out" u ($1/1000):($2/39431.3) w l lw 3 lc rgb '#3afd81' title '  4 H_3O^+', "eight-1.10.out" u ($1/1000):($2/39431.3) w l lw 3 lc rgb '#1e90ff' title '  8 H_3O^+', 
