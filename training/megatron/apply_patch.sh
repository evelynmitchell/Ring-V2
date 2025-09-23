#!/bin/bash

set -e # 如果任何命令失败，则退出脚本

SCRIPT_DIR="$(dirname "$0")"
PATCH_FILE="bailing_moe_v2.patch"
ZIP_FILE="Megatron-LM-core_v0.13.0.zip"
EXTRACTED_DIR="Megatron-LM-core_v0.13.0"

echo "步骤 1/4: 下载 Megatron-LM..."
wget https://github.com/NVIDIA/Megatron-LM/archive/refs/tags/core_v0.13.0.zip -O $ZIP_FILE

echo "步骤 2/4: 解压..."
unzip $ZIP_FILE

echo "步骤 3/4: 移动补丁文件..."
cp ${SCRIPT_DIR}/$PATCH_FILE $EXTRACTED_DIR/

echo "步骤 4/4: 应用补丁..."
cd $EXTRACTED_DIR
patch -p1 < $PATCH_FILE

echo "补丁应用成功！"
cd ..