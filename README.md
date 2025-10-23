# Ring-V2
<p align="center"><img src="./figures/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>

## News
* [2025-10]:🎉 Add [Ring-linear-2.0](https://arxiv.org/abs/2510.19338) 📖 Technical Report
* [2025-10]:🎉 Add [Ring-1T](https://huggingface.co/inclusionAI/Ring-1T) Model
* [2025-09]:🎉 Add [Ring-linear-2.0](https://github.com/inclusionAI/Ring-V2/tree/main/hybrid_linear) Series
* [2025-09]:🎉 Add [Ring-flash-2.0](https://huggingface.co/inclusionAI/Ring-flash-2.0) Model
* [2025-09]:🎉 Add [Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0) Model

## Introduction
Ring-V2 is a family of  reasoning MoE LLMs with a range of sizes provided and open-sourced by InclusionAI, derived from [Ling-V2](https://github.com/inclusionAI/Ling-V2). These models achieve leading performance in complex reasoning at similar sizes, while maintaining high inference speed thanks to their highly sparse architecture.

## Model Downloads


|       **Model**        | **Context Length** |                                                                             **Download**                                                                             |
|:----------------------:| :----------------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ring-1T  |        64K -> 128K (YaRN)	         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-1T) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-1T)  |
| Ring-1T-FP8  |        64K -> 128K (YaRN)	         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-1T-FP8) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-1T-FP8)  |
| Ring-flash-2.0  |        32K -> 128K (YaRN)	         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-flash-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-flash-2.0)  |
| Ring-mini-2.0  |        32K -> 128K (YaRN)	         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-mini-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-mini-2.0)  |

Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-flash-2.0"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment

### vLLM

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:

```bash
git clone -b v0.10.0 https://github.com/vllm-project/vllm.git
cd vllm
wget https://raw.githubusercontent.com/inclusionAI/Ring-V2/refs/heads/main/inference/vllm/bailing_moe_v2.patch
git apply bailing_moe_v2.patch
pip install -e .
```

#### Offline Inference:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-flash-2.0")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)

llm = LLM(model="inclusionAI/Ring-flash-2.0", dtype='bfloat16')
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)

```

#### Online Inference:

```bash
vllm serve inclusionAI/Ring-flash-2.0 \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --use-v2-block-manager \
              --gpu-memory-utilization 0.90
```

To handle long context in vLLM using YaRN, we need to follow these two steps:
1. Add a `rope_scaling` field to the model's `config.json` file, for example:
```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```
2. Use an additional parameter `--max-model-len` to specify the desired maximum context length when starting the vLLM service.

For detailed guidance, please refer to the vLLM [`instructions`](https://docs.vllm.ai/en/latest/).


### SGLang

#### Environment Preparation

We will later submit our model to SGLang official release, now we can prepare the environment following steps:
```shell
pip3 install sglang==0.5.2rc0 sgl-kernel==0.3.7.post1
```
You can use docker image as well:
```shell
docker pull lmsysorg/sglang:v0.5.2rc0-cu126
```
Then you should apply patch to sglang installation:
```shell
# patch command is needed, run `yum install -y patch` if needed
patch -d `python -c 'import sglang;import os; print(os.path.dirname(sglang.__file__))'` -p3 < inference/sglang/bailing_moe_v2.patch
```

#### Run Inference

BF16 and FP8 models are supported by SGLang now, it depends on the dtype of the model in ${MODEL_PATH}. They both share the same command in the following:  

- Start server:
```shell
python -m sglang.launch_server \
    --model-path $MODLE_PATH \
    --host 0.0.0.0 --port $PORT \
    --trust-remote-code \
    --attention-backend fa3
```
MTP is supported for base model, and not yet for chat model. You can add parameter `--speculative-algorithm NEXTN`
to start command.

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)



### Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to [finetune Ring](https://github.com/inclusionAI/Ring-V2/blob/main/docs/llamafactory_finetuning.md).

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring-V2/blob/master/LICENSE).

## Citation

If you find our work helpful, feel free to give us a cite.

```

```
