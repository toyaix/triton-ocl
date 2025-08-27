<h3 align="center">
Triton OpenCL backend 💡
</h3>

<p align="center">
<a href="https://www.zhihu.com/column/c_1906884474676945862"><b>开发文档</b></a> ｜ <a href="https://triton-ocl.top"><b>🔗 triton-ocl.top</b></a>
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

> 项目编译安装在文末，初次使用请划到文档末尾。

## 项目目标

本项目参考和拷贝了[triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu) 和 [microsoft/triton-shared](https://github.com/microsoft/triton-shared) 的大部份代码，目标是提供一个接入`Triton`的样本，完成`tutorials`的适配，大概率是个`Toy`。并跑在`Nvidia`的卡上，目标达到仅使用`CUDA CORE`的`Kernel`性能的`50%`及以上。目前项目的[代码变动](https://github.com/OpenMLIR/triton-ocl/compare/base-develop...OpenMLIR:triton-ocl:develop)


## 项目文档

[Triton SPIR-V 后端开发：矩阵乘实现验证(953bff6)](https://zhuanlan.zhihu.com/p/1925309765489230184)

[Triton SPIR-V 后端开发：向量加实现验证(f66f77c)](https://zhuanlan.zhihu.com/p/1914987075771561349)

[Triton SPIR-V 后端开发：PyBind绑定(457f0aa)](https://zhuanlan.zhihu.com/p/1914770426808206615)

[Triton SPIR-V 后端开发：新增Pass(08b0e35)](https://zhuanlan.zhihu.com/p/1914706253290120299)

[Triton SPIR-V 后端开发：backend 初始化(02ff396)](https://zhuanlan.zhihu.com/p/1907141200789602446)


## 作者相关技术文章

[深度剖析 Triton编译器 MatMul优化（三）—— TMA](https://zhuanlan.zhihu.com/p/1924011555437155686)

[深度剖析 Triton编译器 MatMul优化（二）—— MMA](https://zhuanlan.zhihu.com/p/1922921325296615496)

[深度剖析 Triton编译器 MatMul优化（一）—— FMA](https://zhuanlan.zhihu.com/p/1922542705797465957)

[浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)

[新兴 Python 算子开发：Triton、CuTeDSL、MOJO 🔥等概览](https://zhuanlan.zhihu.com/p/1919816304271028292)

[Triton 社区首贡献：Bug 修复实录](https://zhuanlan.zhihu.com/p/1917136776885174369)

[从零开始教你写一个MLIR Pass](https://zhuanlan.zhihu.com/p/708819963)

[MLIR学习可以参考的项目](https://zhuanlan.zhihu.com/p/1924384457349132481)

[LeetGPU入门教程 (CUDA guide最佳实践)](https://zhuanlan.zhihu.com/p/1899956367734867434)


## 项目运行

仅需要添加`TRITON_SPIRV_BACKEND=1`做为环境变量，即可使用本后端

```bash
TRITON_SPIRV_BACKEND=1 python python/tutorials/spirv_demo/01-vector-add.py
```

如已编译`spirv-opt`，可以运行下面的命令来执行单个/多个 Pass

```bash
build-opt/third_party/spirv/tool/spirv-opt/spirv-opt third_party/spirv/test/add_kernel.ttir  --triton-to-linalg
```

## 里程碑

### 2025.7.6
[matrix-multiplication.py](https://github.com/OpenMLIR/triton-spirv/blob/develop/python/tutorials/spirv_demo/02-matrix-multiplication.py) 跑通对上答案了

### 2025.6.7
[vector-add.py](https://github.com/OpenMLIR/triton-spirv/blob/develop/python/tutorials/spirv_demo/01-vector-add.py) 跑通对上答案了

### 2025.5.17

添加了空的SPIR-V后端，来获取`ttir`。


## 项目展望

这个项目的源起在2023年5月，当时我在将我们自有的`gpgpu`芯片接入`Pytorch`，希望完成对大模型的训练，此时我了解到了`Triton`。在[自有AI芯片接入AI框架Pytorch的方案](https://www.cnblogs.com/BobHuang/p/17879241.html)中，我记录了我对`Triton`的看好。2024年6月随着清华智源的推动，`Triton`也在国内火了起来。我也心痒痒，然后写了[浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，并在文章中提到了想做一个`SPIR-V`后端的玩具。今年我正式投入到了`Triton`的开发中，空闲之余我也进行了算子开发的学习，并写了[LeetGPU入门教程 (CUDA guide最佳实践)](https://zhuanlan.zhihu.com/p/1899956367734867434)。我慢慢意识到了`Triton`的伟大，而不仅是初次了解时的扩充生态以及快速交差的期望。**Pythonic is future!**，世界上会写Python的人越来越多，`AI`模型的发展可能会产生更多的算子，让算法的人能直接写算子带来的收益很可能抹平算子性能的差距。我用业余时间维护`triton-spirv`一方面是为了完成当年想做一个`toy`的心愿，另外也希望降低`Triton`的接入成本，为大家提供一个简明教程。我也衷心地希望`Triton`可以越来越强，既好写性能又过得去，愿`Python kernel mode`生态越来越强大，或许可以打破一部分`CUDA`的护城河。


## 项目编译和安装

由于我需要看MLIR的源码实现，所以这里`clone`了`llvm/llvm-project`，只安装的话`setup.py`会帮你自动下载的，第二块可以忽略，下载好本项目直接`pip install -e .`即可。

`Python 3.10` 的 ctypes 有**bug**，建议使用`Python 3.12`。

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

先安装clinfo查看输出是否正常，并使用项目中的 `python/tutorials/spirv_demo/12-ctypes-cl.py` 测试。`Python 3.10`的`ctypes`会`Segmentation fault`，所以也可以先安装conda。

```bash
git clone https://github.com/OpenMLIR/triton-spirv.git
cd triton-spirv
pip install numpy
python3 python/tutorials/spirv_demo/12-ctypes-cl.py
```

编译llvm还需要`cmake`和`ninja-build`，测试机就直接下载二进制了，如果系统glibc版本低还需要自己编译LLVM，我这里是`Ubuntu 22.04`。`Triton`对Python版本有要求，所以用conda环境，具体命令如下所示。

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
conda create --name triton python=3.12
conda activate triton
pip install -r python/requirements.txt
pip install torch
pip install -e . --no-build-isolation

# GLIBCXX_3.4.30' not found解决
conda install -c conda-forge libstdcxx-ng=12

TRITON_SPIRV_BACKEND=1 TRITON_CACHE_DIR=$PWD/.cache python python/tutorials/spirv_demo/01-vector-add.py
# 可能会缺少 ZLIB::ZLIB，target "LLVMSupport" contains，需要安装
# sudo apt install zlib1g zlib1g-dev
```

---

## [Upstream README](https://github.com/OpenMLIR/triton-spirv?tab=readme-ov-file#upstream-readme)
