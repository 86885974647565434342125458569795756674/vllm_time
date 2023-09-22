import matplotlib.pyplot as plt
import os

dir_name = os.path.dirname(__file__) 
dir_name = os.path.join(dir_name, "../batch_seq")    
dir_list = os.listdir(dir_name)

data = []

for txt_name in dir_list:
    if txt_name == "cpu2gpu.txt":
        continue
    txt_path = os.path.join(dir_name, txt_name) 
    batch_size = int(txt_name.split("_")[0])
    seq_id = int(txt_name.split("_")[1].split(".")[0])
    with open(txt_path, mode='r', encoding='utf-8') as f:
        line = f.readline()          
        line = f.readline()   
        line = f.readline()   
        line = f.readline()   
        prompt_len = line.split(" ")[-1][:-1]
        line = f.readline()  
        state = "init"
        while line:
            if line[0] == "*":
                line = f.readline()
                line = f.readline() 
                v1 = 0
                v2 = 0
                count = 0
                while line and line[0] == "-":
                    line = f.readline()   
                    k1 = line.split(" ")[0]
                    if count > 0:
                        v1 += float(line.split(" ")[1][:-1])
                    line = f.readline()
                    k2 = line.split(" ")[0]
                    if count > 0:
                        v2 += float(line.split(" ")[1][:-1])
                    line = f.readline()   
                    count += 1
                v1 /= count - 1
                v2 /= count - 1
                data.append((batch_size,seq_id,state,prompt_len,k1,v1,k2,v2,v1/v2))
                state = "add"

data.sort(key=lambda x:(x[2],x[3],x[0]))

# for d in data:
#     print(d)

data1 = []
txt_path=os.path.join(dir_name,"cpu2gpu.txt") 
with open(txt_path, mode='r', encoding='utf-8') as f:
    line = f.readline() 
    while line:
        batch_size = int(line.split(" ")[0])
        seq_id = int(line.split(" ")[1])
        time = float(line.split(" ")[2][:-1])
        data1.append((batch_size,seq_id,time))
        line = f.readline() 

data1.sort(key=lambda x:(x[1],x[0]))

times = []
for i in range(3):
    x = []
    y = []
    for j in range(8):
        x.append(data1[i*8+j][0])
        y.append(data1[i*8+j][2])
    times.append((data1[i*8][1],x,y))

for d in times:
    plt.plot(d[1],d[2],label=d[0],marker=".")

qkv_pro = []
attn = []
ra = []
for i in range(6):
    qx = []
    qy = []
    ax = []
    ay = []
    rx = []
    ry = []
    for j in range(8):
        qx.append(data[i*8+j][0])
        qy.append(data[i*8+j][5])
        ax.append(data[i*8+j][0])
        ay.append(data[i*8+j][7])
        rx.append(data[i*8+j][0])
        ry.append(data[i*8+j][8])
    qkv_pro.append((data[i*8][2],data[i*8][3],qx,qy))
    attn.append((data[i*8][2],data[i*8][3],ax,ay))
    ra.append((data[i*8][2],data[i*8][3],rx,ry))

plt.title("qkv_pro")
for d in qkv_pro:
    plt.plot(d[2],d[3],label=d[0]+"-"+d[1],marker=".")



plt.legend()
plt.savefig('qkv_pro.png')

plt.figure()
plt.title("attn")
for d in attn:
    plt.plot(d[2],d[3],label=d[0]+"-"+d[1],marker=".")

plt.legend()
plt.savefig('attn.png')

plt.figure()
plt.title("ratio")
for d in ra:
    plt.plot(d[2],d[3],label=d[0]+"-"+d[1],marker=".")

plt.legend()
plt.savefig('ra.png')