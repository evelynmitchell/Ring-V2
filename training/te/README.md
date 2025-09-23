## TransformerEngine (TE) Patching

This is a patch for transpormer_engine package with version 2.6.0.post1 to add a GPU memory optimization for blockwise fp8 training.

By default, each blockwise fp8 weight will create a rowwise fp8 weight and a columnwise fp8 weight in advance and keep them during the training. This optimization will not create columnwise fp8 weight in advance, but create it when needed in backward pass. Since blockwise fp8 quantizes weight in 128x128 blocks, columnwise fp8 weight can be created from rowwise fp8 weight by using a triton kernel to speed up the transpose. This is a tradeoff between computation and storage.

## How to Patch
Make sure transformer_engine with version 2.6.0.post1 is already installed. Then run:
```
bash apply_te_patch.sh
```

## How to Use

To turn on this fp8 weight gpu optimization feature, set envionment variable:
```
TE_ON_DEMAND_FP8_WEIGHT_T_CREATION=1
```
