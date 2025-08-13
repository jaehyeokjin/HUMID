set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Distance ({\305})'
set ylabel 'Molecular-wise g(R)'
set yr [:4]
set format y '%1.1f'
set xr [2:8]
pl "bumper.dat" u 1:2 w l lw 9 lc rgb '#1e90ff' title 'BUMPer CG', "aa.dat" u 1:2 w l lw 7 lc rgb '#c71eff' dt '...' title 'All-Atom (Ref.)'
