import os
import numpy as np

dataset='i2b2'
setting = '8-2'
baseline='i2b2_8-2_RDP'
split = "Test"



path = f"{dataset}_{setting}_{baseline}"
entity_num = {"i2b2":16, "onto":18, "conll":4}
runs = 5
tasks = int((entity_num[dataset]-eval(setting[0]))/(eval(setting[-1])) +1)

MiF1s = np.zeros((runs,tasks))
MaF1s = np.zeros((runs,tasks))

assert len(os.listdir(path)) == runs

for i, dir in enumerate(os.listdir(path)):
    log_file_path = os.path.join(os.path.join(path,dir),'train.log')
    count = 0
    MiF1 = 0
    MaF1 = 0
    with open(log_file_path, encoding='utf-8') as f:
        for line in f:
            idx_beg = line.rfind(split+"_f1=")
            if idx_beg != -1: 
                idx_end = line.find(',', idx_beg)
                new_str = line[idx_beg+len(split+"_f1="):idx_end]
                score = eval(new_str)
                MiF1s[i][count] = score

            idx_beg = line.rfind(split+"_ma_f1=")
            if idx_beg != -1: 
                idx_end = line.find(',', idx_beg)
                new_str = line[idx_beg+len(split+"_ma_f1="):idx_end]
                score = eval(new_str)
                MaF1s[i][count] = score

                count += 1


    assert count == tasks




print(MiF1s)
print(MaF1s)

print("*********************************************************")
tw_avg_mi_f1 = np.mean(MiF1s,axis=0)
tw_std_mi_f1 = np.std(MiF1s,axis=0)
tw_avg_ma_f1 = np.mean(MaF1s,axis=0)
tw_std_ma_f1 = np.std(MaF1s,axis=0)
print("Task-wise Results:")
print("Task-wise avg Mi F1: ", tw_avg_mi_f1)
print("Task-wise std Mi F1: ", tw_std_mi_f1)
print("Task-wise avg Ma F1: ", tw_avg_ma_f1)
print("Task-wise std Ma F1: ", tw_std_ma_f1)

print("*********************************************************")
print("Final Main Results:")
print("Final avg Mi F1: ", np.mean(np.mean(MiF1s,axis=1)))
print("Final std Mi F1: ", np.std(np.mean(MiF1s,axis=1)))
print("Final avg Ma F1: ", np.mean(np.mean(MaF1s,axis=1)))
print("Final std Ma F1: ", np.std(np.mean(MaF1s,axis=1)))


print("Final avg Mi F1: ", np.mean(MiF1s,axis=1))
print("Final avg Ma F1: ", np.mean(MaF1s,axis=1))



