path = './q_weight_max'
i = 1
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) == True:
        file = open(os.path.join(path,file), "r", encoding="utf-8")
        mystr1 = file.readline()  # 表示一次读取一行
        file_max = open('./q_weight_max/q_weight_max.txt', "a", encoding="utf-8")
        if i == 60:
            print(mystr1)
        file_max.write(mystr1[:-1]+'\n')
        file_max.close()
        file.close()
        i += 1

path = './q_activation_max'
i = 1
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) == True:
        new_name = file.replace(file, "max_activation-modulelist_Conv2d_%d.txt" % (151 - i))
        os.rename(os.path.join(path, file), os.path.join(path, new_name))
        #########合并最大值文档150layers
        '''file = open(os.path.join(path, new_name), "r", encoding="utf-8", errors="ignore")
        mystr1 = file.readline()  # 表示一次读取一行
        file_max = open('./q_activation_max/q_activation_max.txt', "a", encoding="utf-8", errors="ignore")
        file_max.write(mystr1[:-1] + '\n')
        file_max.close()
        file.close()'''
        i += 1
