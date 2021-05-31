import os

rootdir = r'E:\pythontest\CNN\images\test'



fc = open(os.path.join(rootdir,'table.txt'),'w')

dir_list = os.listdir(rootdir)
i = 0
length = len(dir_list)-1
while i < length:
    path = os.path.join(rootdir,dir_list[i])
    temp_list = os.listdir(path)
    print(temp_list)
    j = 0
    temp_length = len(temp_list)
    while j<temp_length:
        fc.write(temp_list[j]+" "+str(i)+'\n')
        j+=1
    i+=1

fc.close()