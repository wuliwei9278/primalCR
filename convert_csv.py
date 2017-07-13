def convert_csv(input_file_name, output_file_name):
	fo = open(output_file_name, 'w')
	user_id_max = -1
	item_id_max = -1
	cc = 0
	with open(input_file_name, 'r') as f:
		for line in f:
			tmp = line.split(',')
			user_id_max = max(user_id_max, int(tmp[0]))
			item_id_max = max(item_id_max, int(tmp[1]))
			cc += 1
			new_line = ' '.join(tmp)
			fo.write(new_line)
	fo.close()
	print("in file " + output_file_name + " there are " + str(cc) + " lines, " \
		+ " max user id: " + str(user_id_max) + "; max item id: " + str(item_id_max))
	return cc, user_id_max, item_id_max

data_folder = "/Users/wuliwei/Documents/primalCR/ml1m/"
# training the first, test the second
input_file_name_l = ["ml1m_train_ratingsALL.csv", "ml1m_test_ratings10.csv"]
output_file_name_l = ["training.ratings", "test.ratings"]
num_lines = []
user_id_max = -1
item_id_max = -1
for i in range(len(input_file_name_l)):
	tmp = convert_csv(data_folder + input_file_name_l[i], data_folder + output_file_name_l[i])
	num_lines.append(tmp[0])
	user_id_max = max(user_id_max, tmp[1])
	item_id_max = max(item_id_max, tmp[2])
	
fo = open(data_folder + "meta", 'w')
fo.write(' '.join([str(user_id_max), str(item_id_max)]) + '\n')
fo.write(' '.join([str(num_lines[0]), output_file_name_l[0]]) + '\n')
fo.write(' '.join([str(num_lines[1]), output_file_name_l[1]]) + '\n')
fo.close