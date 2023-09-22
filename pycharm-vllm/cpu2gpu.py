import torch
import os

dir_name = os.path.dirname(__file__) 
dir_name = os.path.join(dir_name, "../batch_seq")    
dir_list = os.listdir(dir_name)
seq_len = [11,22,45]

def count_time(*size):
    a=torch.ones(*size,dtype=torch.float16,device="cpu")
    a.to("cuda")

    b=torch.ones(*size,dtype=torch.float16,device="cpu")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True) 
    start.record()
    b.to("cuda")
    end.record()
    torch.cuda.synchronize() 
    return start.elapsed_time(end)

for txt_name in dir_list:
    if txt_name == "cpu2gpu.txt":
        continue
    txt_path = os.path.join(dir_name, txt_name) 
    batch_size = int(txt_name.split("_")[0])
    seq_id = int(txt_name.split("_")[1].split(".")[0])
    total_time = 0
    for _ in range(3):
        total_time += count_time(batch_size*seq_len[seq_id], 12, 64)
    print(batch_size,seq_len[seq_id],total_time/3)