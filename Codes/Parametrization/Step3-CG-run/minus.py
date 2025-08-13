import sys
from numpy import *
from pylab import *
from scipy.optimize import curve_fit

# Fitting function
def func(x, a, b, c):
    return a/(x**b)-c
v_threshold = 5000.0
r_threshold = 0.8
if len(sys.argv)!=2:
    sys.exit()
inputname = sys.argv[1]
inputfile = "corr_"+inputname+".table"
inp = open(inputfile)
print_statement = "Start fitting the potential: "+ inputname
print(print_statement)
# Initial parameters
i = 0
r = []
r_og = []
v = []
f = []
header = ""

for line in inp:
    if i <= 4:
        header += line
    line = line.strip()
    if i > 4 and line!="" and line[0] != "#":
        line = line.split()
        if len(line) != 4:
            sys.exit()
        r.append(float(line[1]))
        r_og.append(float(line[1]))
        v.append(float(line[2]))
        f.append(float(line[3]))
    i += 1
inp.close()

# Fitting part
## Rcut < reading
r_cut_tb = open("r_cut.in", 'r')
for line in r_cut_tb:
    line_element = line.split()
    if line_element[0] == inputname:
        rcut = float(line_element[1])
rr = []
vv = []
## Setting the fitting parameters
for i in range(len(r)):
    if r[i] <= rcut:
        rr.append(r[i])
        vv.append(v[i])
r_small=r[0]
## Start the fitting
par, pcov = curve_fit(func, rr, vv,maxfev=2000)

## Dump the data
v2 = []
f2 = []
rmax_count = max(r)
r_bin_rough = int(50.0*(rmax_count-1.0) + 1.0)
r=[]
for i in range(0, r_bin_rough):
    rvalue = "%.2f" % (1.0 + 0.02 * float(i))
    r.append(float(rvalue))
r_final = [] # Final r after checking the value < threshold
r_correction_site = 2.68
f_correction_factor = f[r_og.index(r_correction_site)] - (-(func(r[r.index(r_correction_site)], par[0], par[1], par[2]) - func(r[r.index(r_correction_site)+1], par[0], par[1], par[2]))/(r[r.index(r_correction_site)] - r[r.index(r_correction_site)+1]))

for i in range(len(r)):
    if r[i] < r_small:
        vtmp = func(r[i], par[0], par[1], par[2])
        ftmp = -(func(r[i], par[0], par[1], par[2]) - func(r[i+1], par[0], par[1], par[2]))/(r[i] - r[i+1])+f_correction_factor
        if vtmp < v_threshold:
            r_final.append(r[i])
            v2.append(vtmp)
            f2.append(ftmp)
    else:
        r_final.append(r[i])
        original_index = r_og.index(r[i])
        v2.append(v[original_index])
        f2.append(f[original_index])

outname = "fitting_"+inputname+".table"
outfile = open(outname, 'w')
outfile.write(header)
for i in range(len(r_final)):
    outfile.write("%d %.6f %.6f %.6f\n" % (i+1, r_final[i], v2[i], f2[i]))
outfile.close()
