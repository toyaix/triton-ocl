# Triton-SPIRV

> 更多Triton相关请参考[英文文档](./README.md) - English Documentation

> 项目编译安装在文末，初次使用请划到文档末尾。

## 项目目标

本项目参考和拷贝了[triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu) 和 [microsoft/triton-shared](https://github.com/microsoft/triton-shared) 的大部份代码，目标是提供一个接入`Triton`的样本，完成`tutorials`的适配，大概率是个`Toy`。并跑在`Nvidia`的卡上，目标达到仅使用`CUDA CORE`的`Kernel`性能的`50%`及以上。目前项目的[代码变动](https://github.com/triton-lang/triton/compare/main...OpenMLIR:triton-spirv-preview:develop)


## 项目文档

[Triton SPIR-V 后端开发：向量加实现验证](https://zhuanlan.zhihu.com/p/1914987075771561349)

[Triton SPIR-V 后端开发：PyBind绑定](https://zhuanlan.zhihu.com/p/1914770426808206615)

[Triton SPIR-V 后端开发：新增Pass](https://zhuanlan.zhihu.com/p/1914706253290120299)

[Triton SPIR-V 后端开发：backend 初始化](https://zhuanlan.zhihu.com/p/1907141200789602446)


## 作者相关技术文章

[浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)

[从零开始教你写一个MLIR Pass](https://zhuanlan.zhihu.com/p/708819963)

[LeetGPU入门教程 (CUDA guide最佳实践)](https://zhuanlan.zhihu.com/p/1899956367734867434)


## 项目运行

仅需要添加`TRITON_SPIRV_BACKEND=1`做为环境变量，即可使用本后端

```bash
TRITON_SPIRV_BACKEND=1 python python/tutorials/spirv_demo/01-vector-add.py
```

如已编译`spirv-opt`，可以运行下面的命令来执行单个/多个 Pass

```bash
build-opt/third_party/spirv/tool/spirv-opt third_party/spirv/test/add_kernel.ttir  --triton-to-linalg
```

## 里程碑

### 2025.6.7
[vector-add.py](https://github.com/OpenMLIR/triton-spirv/blob/develop/python/tutorials/spirv_demo/01-vector-add.py) 跑通对上答案了

### 2025.5.17

添加了空的SPIR-V后端，来获取`ttir`。


## 项目展望

这个项目的源起在2023年5月，当时我在将我们自有的`gpgpu`芯片接入`Pytorch`，希望完成对大模型的训练，此时我了解到了`Triton`。在[自有AI芯片接入AI框架Pytorch的方案](https://www.cnblogs.com/BobHuang/p/17879241.html)中，我记录了我对`Triton`的看好。2024年6月随着清华智源的推动，`Triton`也在国内火了起来。我也心痒痒，然后写了[浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，并在文章中提到了想做一个`SPIR-V`后端的玩具。今年我正式投入到了`Triton`的开发中，空闲之余我也进行了算子开发的学习，并写了[LeetGPU入门教程 (CUDA guide最佳实践)](https://zhuanlan.zhihu.com/p/1899956367734867434)。我慢慢意识到了`Triton`的伟大，而不仅是初次了解时的扩充生态以及快速交差的期望。**Pythonic is future!**，世界上会写Python的人越来越多，`AI`模型的发展可能会产生更多的算子，让算法的人能直接写算子带来的收益很可能抹平算子性能的差距。我用业余时间维护`triton-spirv`一方面是为了完成当年想做一个`toy`的心愿，另外也希望降低`Triton`的接入成本，为大家提供一个简明教程。我也衷心地希望`Triton`可以越来越强，既好写性能又过得去，愿`Python kernel mode`生态越来越强大，或许可以打破一部分`CUDA`的护城河。


## 项目编译和安装

由于我需要看MLIR的源码实现，所以这里`clone`了`llvm/llvm-project`，只安装的话`setup.py`会帮你自动下载的，第二块可以忽略，下载好本项目直接`pip install -e .`即可。

```bash
# 克隆 本项目
git clone https://github.com/OpenMLIR/triton-spirv.git
cd triton-spirv
# 查看其依赖的LLVM版本
cat cmake/llvm-hash.txt
# 依赖LLVM版本号为3c709802d31b5bc5ed3af8284b40593ff39b9eec

cd ~
# 克隆LLVM
git clone https://github.com/llvm/llvm-project.git
# 切换到 cat 出的依赖的LLVM版本
git checkout 3c709802d31b5bc5ed3af8284b40593ff39b9eec
mkdir build; cd build
# 设置CMake 参数
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
# 编译LLVM，需要蛮久的
ninja
export LLVM_BUILD_DIR=~/llvm-project/build

# 进入Triton文件夹
cd <triton install>
# 设置debug模式，可选
export DEBUG=1
# build-time dependencies
pip install -r python/requirements.txt
# 调用 setup.py 安装，使用 --no-build-isolation 可以加快编译速度
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -e . --no-build-isolation
```

## 编译 spirv-opt 方便调试

提供了`BUILD_SPIRV_OPT`选项来控制是否编译二进制的命令行工具。

```bash
mkdir build-opt; cd build-opt
cmake -G Ninja .. -DLLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include  -DLLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib -DTRITON_CODEGEN_BACKENDS="nvidia;amd;spirv" -DCMAKE_BUILD_TYPE=Debug -DBUILD_SPIRV_OPT=ON -DTEST_SPIRV_CC=ON
```


## 测试机配置

先安装clinfo查看输出是否正常，并使用项目中的 `python/tutorials/spirv_demo/12-ctypes-cl.py` 测试
```bash
git clone https://github.com/OpenMLIR/triton-spirv.git
cd triton-spirv
python3 python/tutorials/spirv_demo/12-ctypes-cl.py
```
编译llvm还需要`cmake`和`ninja-build`，测试机就直接下载二进制了。`triton`对Python版本有要求，所以用conda环境，具体命令如下所示
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
conda create --name triton python=3.12
conda activate triton
pip install -r python/requirements.txt
pip install -e . --no-build-isolation
```

---

## [Upstream README](https://github.com/OpenMLIR/triton-spirv?tab=readme-ov-file#upstream-readme)
