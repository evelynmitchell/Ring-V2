FROM nvcr.io/nvidia/pytorch:25.08-py3
RUN pip install transformers==4.45.0
RUN pip install accelerate
RUN pip install datasets==3.6.0
RUN pip install peft
RUN pip install dependency_injector
RUN pip install nvidia-nvshmem-cu12
RUN git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && export TORCH_CUDA_ARCH_LIST=9.0 && python setup.py build && python setup.py install

RUN pip uninstall -y transformer_engine transformer_engine_cu12 transformer_engine_torch
RUN pip install setuptools>=61.0 && pip install einops && pip install onnxscript==0.3.1 && pip install onnx
RUN pip install transformer_engine==2.6.0.post1
RUN pip install transformer_engine_torch==2.6.0.post1 --no-build-isolation --no-deps

RUN pip install jsonlines
RUN pip install bitsandbytes