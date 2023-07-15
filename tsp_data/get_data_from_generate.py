import os

batch_size = [1]
nodes_num = [10,20,50,100]
random_mode = ['uniform','gaussian']

for i in range(len(batch_size)):
    for j in range(len(nodes_num)):
        for k in range(len(random_mode)):
            command = "python tsp_data/generate_tsp_data.py --batch_size={} --nodes_num={}\
                --random_mode={}".format(batch_size[i],nodes_num[j],random_mode[k])
            os.system(command)

    

    