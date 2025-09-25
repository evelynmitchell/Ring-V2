# Ring-Linear-V2
<p align="center"><img src="../figures/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>

## News
* [2025-09]:🎉 Add [Ring-flash-linear-2.0](https://huggingface.co/inclusionAI/Ring-flash-linear-2.0) Model
* [2025-09]:🎉 Add [Ring-mini-linear-2.0](https://huggingface.co/inclusionAI/Ring-mini-linear-2.0) Model

## Introduction
We are excited to announce the official open-source release of Ring-linear-V2 series! Building on the success of our [Ling-V2](https://github.com/inclusionAI/Ling-V2) series, these models continues to leverage a powerful hybrid architecture of linear and standard attention, perfectly balancing high performance with superior efficiency. 

## Model Downloads


|       **Model**        | **Context Length** |                                                                             **Download**                                                                             |
|:----------------------:| :----------------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ring-flash-linear-2.0  |        128k         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-flash-linear-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-flash-linear-2.0)  |
| Ring-mini-linear-2.0  |        128k         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-mini-linear-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-mini-linear-2.0)  |

Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).

## Quickstart

### 🤗 Hugging Face Transformers
Installation requirements:

```shell
pip install flash-linear-attention==0.3.2
pip install transformers==4.56.1
```

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-mini-linear-2.0"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


prompts = [
    "Give me a short introduction to large language models."
]
input_texts = []
for prompt in prompts:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_texts.append(text)

print(input_texts)

model_inputs = tokenizer(input_texts, return_tensors="pt", return_token_type_ids=False, padding=True, padding_side='left').to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192,
    do_sample=False,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("*" * 30)
print(responses)
print("*" * 30)
```

### 🚀 vLLM

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:
```shell
pip install torch==2.7.0 torchvision==0.22.0 
```

Then you should install our vLLM wheel package:
```shell
pip install ./vllm-0.8.5+cuda12_8_gcc10_2_1-cp310-cp310-linux_x86_64.whl --no-deps --force-reinstall
```

#### Offline Inference

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-mini-linear-2.0")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)

llm = LLM(model="inclusionAI/Ring-mini-linear-2.0", dtype='bfloat16', enable_prefix_caching=False, max_num_seqs=128)
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)
```

#### Online Inference
```shell
vllm serve inclusionAI/Ring-mini-linear-2.0 \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --gpu-memory-utilization 0.90 \
              --max-num-seqs 512 \
              --no-enable-prefix-caching
```


### 🚀 SGLang

#### Environment Preparation

We will later submit our model to SGLang official release, now we can prepare the environment following steps:
```shell
pip3 install sgl-kernel==0.3.9.post2 vllm==0.10.2 torch==2.8.0 torchvision==0.23.0
```

Then you should install our sglang wheel package:
```shell
pip install ./whls/sglang-0.5.2-py3-none-any.whl
```

#### Run Inference

BF16 and FP8 models are supported by SGLang now, it depends on the dtype of the model in ${MODEL_PATH}. They both share the same command in the following:  

- Start server:
```shell
python -m sglang.launch_server \
    --model-path <model_path> \
    --trust-remote-code \
    --tp-size 1 \
    --disable-radix-cache \
    --json-model-override-args "{\"linear_backend\": \"seg_la\"}"
```

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)



## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring-V2/blob/master/LICENSE).

## Citation

If you find our work helpful, feel free to give us a cite.

