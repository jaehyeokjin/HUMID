set terminal wxt size 1320,1200 enhanced font 'Whitney,42' persist

set border lw 4
set termoption enhanced
set encoding iso_8859_1
set xlabel 'Time (ps)'
set ylabel 'C_{vv} ({\305}^2/ps)'
set xtics 0.2
set xtics format '%1.1f'
set ylabel 'C_{{/Symbol d}Fv} (cal/mol)'

pl "cfv.out" u ($1/1000):($2*1000) w l lw 9 lc rgb '#6abf31' title 'Mapped All-Atom'
