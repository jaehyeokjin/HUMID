import sys
fname=sys.argv[1]
f=open(fname,'r')
outname="histo_"+fname.split(".")[0]+".out"
f1=open(outname, 'w')

t=[]
index=[]
hop=[]
h=[]
for line in f:
	t.append(int(line.split()[0]))
	index.append(int(line.split()[1]))
h_val=0
h_increment=0
for i in range(1,len(t)):
	h_increment=0
	if index[i] != index[i-1]:
		indicator = 0
		if (len(hop)) !=0:
			for j in range(0,len(hop)):
				if ((index[i-1] == hop[j][0]) and (index[i] == hop[j][1])):
					h_increment = 1
					indicator = 1
				else:
					if ((index[i-1] == hop[j][1]) and (index[i] == hop[j][0])):
						h_increment = -1
						indicator = 1
			if indicator != 1:
				hop.append([index[i-1], index[i]])
				h_increment = 1
				#check2 = "%d -> %d with %d hop length:%d\n" %(index[i-1], index[i], h_increment, len(hop))
				#print(check2)
		else:
			hop.append([index[i-1], index[i]])
			h_increment = 1
	else:
		h_increment = 0
#	check = "%d -> %d with %d hop length:%d\n" %(index[i-1], index[i], h_increment, len(hop))
	h_val += h_increment
	st_out="%d %d\n" % (t[i], h_val)
	f1.write(st_out)
print(hop)
f.close()
f1.close()

