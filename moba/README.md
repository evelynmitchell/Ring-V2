# Ring-sparse-2.0-exp
<p align="center"><img src="../figures/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>

## News
* [2025-10]:🎉 Add [Ring-mini-sparse-2.0-exp](https://huggingface.co/inclusionAI/Ring-mini-sparse-2.0-exp) Model

## Introduction
We're excited to announce the open-source release of **Ring-mini-sparse-2.0-exp**! Building on our [Ring-mini-2.0](https://github.com/inclusionAI/Ling-V2), this new version leverages block-wise sparse attention to achieve superior performance with significantly improved efficiency.

## Model Downloads



|       **Model**        | **Maximum Supported Length** |                                                                             **Download**                                                                             |
|:----------------------:| :----------------: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ring-mini-sparse-2.0-exp  |        128k         |  [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-mini-sparse-2.0-exp) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ring-mini-sparse-2.0-exp)  |

## Quickstart

### 🤗 Hugging Face Transformers
Installation requirements:

```shell
pip install transformers==4.56.1
```

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-mini-sparse-2.0-exp"

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
        add_generation_prompt=True
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

### 🚀 SGLang

#### Environment Preparation

We have submitted our PR to SGLang official release and it will be merged later, for now we can prepare the environment following steps, firstly install the community version SGLang and required packages:
```shell
pip install sglang==0.5.3 sgl-kernel==0.3.15 torch==2.8.0 torchvision==0.23.0 torchao
```

Then you should install our sglang wheel package:
```shell
pip install http://raw.githubusercontent.com/inclusionAI/Ring-V2/blob/main/moba/whls/sglang-0.5.3.post1-py3-none-any.whl --no-deps --force-reinstall
```

#### Run Inference

Our model is supported by SGLang now. You can launch the sever with the command in the following:  

- Start server:
```shell
python -m sglang.launch_server \
    --model-path <model_path> \
    --trust-remote-code \
    --tp-size 4 \  ## Currently MoBA with GQA only supports tp size = group size
    --disable-radix-cache \
    --chunked-prefill-size 0 \
    --attention-backend moba
```

- Client:

```shell
curl -s http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "temperature": 0.6, "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}]}'
```

More usage can be found [here](https://docs.sglang.ai/basic_usage/send_request.html)

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring-V2/blob/master/LICENSE).

## Citation

If you find our work helpful, feel free to give us a cite.
