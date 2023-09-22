docker run --privileged -it -p 15000:22 --name cyy_vllm --gpus '"device=1"'  -v /data/cyy/vllm:/vllm cyy/vllm

docker exec --privileged -it cyy_vllm /bin/bash 

passwd

apt update

apt install openssh-server

/etc/ssh/sshd_config

service ssh start

scp -r optm-125 cyy@172.18.xxxx:/data/cyy/vllm

# c++

name="vllm.cache_ops",sources=["csrc/cache.cpp", "csrc/cache_kernels.cu"]

name="vllm.attention_ops",sources=["csrc/attention.cpp", "csrc/attention/attention_kernels.cu"]

name="vllm.pos_encoding_ops",sources=["csrc/pos_encoding.cpp", "csrc/pos_encoding_kernels.cu"]

name="vllm.layernorm_ops",sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"]

name="vllm.activation_ops",sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"]

# time

[Hello-SimpleAI/HC3 · Datasets at Hugging Face](https://huggingface.co/datasets/Hello-SimpleAI/HC3/viewer/all/train?row=35)



/usr/local/lib/python3.8/dist-packages/vllm/model_executor/models/opt.py

OPTDecoder

OPTDecoderLayer

OPTAttention

/usr/local/lib/python3.8/dist-packages/vllm/model_executor/layers/attention.py

PagedAttention



```
        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            # Prompt run.
            assert input_metadata.num_generation_tokens == 0
```



/usr/local/lib/python3.8/dist-packages/vllm/core/scheduler.py

self.scheduler_config.max_model_len

self.scheduler_config.max_num_batched_tokens

self.scheduler_config.max_num_seqs

2048 2560 256



2048

2048*32768=67108864

32768

# run

执行脚本

画图
