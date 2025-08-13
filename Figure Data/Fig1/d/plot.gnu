set terminal wxt size 3000,1200 enhanced font 'Whitney,64' persist

set border lw 6
set termoption enhanced
set encoding iso_8859_1
set yr [:2]
set xr [:1000]
set ytics format '%1.1f'
set xlabel 'Time (ps)'
set ylabel 'Averaged Proton Values'
y1(x) = 0.94699404375
y2(x) = 0.614647016502
set xr [-1:1001]
pl "./average/boundary_histo_0_1250.out" u ($1/200):2 w l lw 9 lc rgb '#FA8072' title 'Condition 2: {/Symbol S}_I p_I', "" u ($1/200):3 w l lw 9 lc rgb '#1e90ff' title 'Condition 3: max(p_I)', y1(x) w l lw 16 dt '...' lc rgb '#8072fa' notitle, y2(x) w l lw 16 dt '...' lc rgb '#ff1e90' notitle
