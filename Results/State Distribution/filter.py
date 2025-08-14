import sys
name=sys.argv[1]
freq = int(sys.argv[2])
f=open(name,'r')
outname="./average/"+name.split(".")[0]+"_"+str(freq)+".out"
out=open(outname,'w')
x=[]
y=[]
z=[]
avg_y=[]
avg_z=[]
for line in f:
	l_e=line.split()
	x.append(int(l_e[0]))
	y.append(float(l_e[1]))
	z.append(float(l_e[2]))
for i in range(0,len(x)):
	if (i<freq):
		sum_y=0.0
		sum_z=0.0
		id_y = 0
		for j in range(0,i+1):
			sum_y += y[j]
			sum_z += z[j]
			id_y +=1 
		avg_y.append(sum_y/float(id_y))
		avg_z.append(sum_z/float(id_y))
	elif (i>(len(x)-freq)):
		sum_y=0.0
		sum_z=0.0
		id_y = 0
		for j in range(len(x)-freq,i+1):
			sum_y += y[j]
			sum_z += z[j]
			id_y +=1
			avg_y.append(sum_y/float(id_y))
			avg_z.append(sum_z/float(id_y))
	else:
		sum_y=0.0
		sum_z=0.0
		id_y = 0
		for j in range(i-freq,i+freq):
			sum_y += y[j]
			sum_z += z[j]
			id_y += 1
		avg_y.append(sum_y/float(id_y))
		avg_z.append(sum_z/float(id_y))

for i in range(0,len(x)):
	str_out = "%d %.6f %.6f\n" % (x[i],avg_y[i],avg_z[i])
	out.write(str_out)
f.close()
out.close()
