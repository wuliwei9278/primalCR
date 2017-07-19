import random
n = 6040
m = 3952
k = 10
U_file_name = "initial.U"
V_file_name = "initial.V"
with open(U_file_name, 'wb') as f:
	for i in range(n):
		tmp_list = []
		for i in range(k):
			tmp_list.append(random.uniform(0,1))
		line = ' '.join([str(x) for x in tmp_list]) + '\n'
		f.write(line)
with open(V_file_name, 'wb') as f:
    for i in range(m):
        tmp_list = []
        for i in range(k):
            tmp_list.append(random.uniform(0,1))
        line = ' '.join([str(x) for x in tmp_list]) + '\n'
        f.write(line)


