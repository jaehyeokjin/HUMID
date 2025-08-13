import sys

filename = sys.argv[1]
f=open(filename,'r')
count = len(open(filename).readlines(  ))
filename2 = "corr_"+filename
f1=open(filename2,'w')
i=0
totnum = 0
for line in f:
    if i <= 2:
        f1.write(line)
    if i == 3:
        line_element = line.split()
        totnum = int(line_element[1])
    if i == 5:
        line_element = line.split()
        rmin = line_element[1]
        deduct = int(line_element[0])-1
        changed_tot = totnum-deduct-1
        final_line = "N "+str(count-5)+" R "+rmin+" 9.000000\n\n"+ "1 "+line_element[1] + " " +line_element[2]+ " " +line_element[3]+"\n"
        f1.write(final_line)
    if i >= 6:
        line_element = line.split()
        index = int(line_element[0]) - deduct
        final_out = str(index) + " " + line_element[1] + " " + line_element[2] +" " + line_element[3] + "\n"
        f1.write(final_out)
    i += 1

f.close()
f1.close()
