# Mind-KernelInfra

#### 介绍
算子公共平台，Mind-KernelInfra，缩写MKI。它统一定义了算子调用的L0级接口。南向对接不同组织开发的算子库，北向支撑不同加速库应用，实现M x N算子能力复用。 

#### 软件架构
软件架构说明
1. 调用关系
 ATB(or other boost) --> MKI --> Ascend-op_common-lib(or other op libs)

#### 安装教程
无需安装， 直接与算子包一起编译。见下一章使用说明。  

#### 使用说明
三种典型使用场景：
场景1. 与算子库、加速库一起编译出包使用
下面假设 Mind-KernelInfra、算子库（本例中是ascend-op-common-lib）、加速库（本例中是ascend-transformer-boost）代码都已经准备好，且处于统一级目录下。
1.  使用算子命名空间作为参数， 编译MKI, 并将编译产物拷贝到算子库和加速库的3rdparty目录：在这里例子中， 命名空间参数是AtbOps 

-     cd Mind-KerenlInfra
-     bash scripts/build.sh --namespace=AtbOps testframework
-     cp -r output/mki ../ascend-op-common-lib/3rdparty/
-     cp -r output/mki ../ascend-transformer-boost/3rdparty/


2.  编译算子库， 并将编译产物拷贝到加速库的3rdparty目录

-     cd ascend-op-common-lib
-     bash scripts/build.sh  testframework --no_werror
-     cd output/
-     cp -r output/atbops ../ascend-transformer-boost/3rdparty/
-     cp -r output/asdops ../ascend-transformer-boost/3rdparty/


3.  编译加速库

-     cd ascend-transformer-boost/
-     source scripts/set_env.sh
-     bash scripts/build.sh  testframework
-     cd output/
-     ls
-     cd atb
-     source set_env.sh

4.  运行模型或算子测试用例

场景2：仅与算子库一起编译后， 在算子层面进行调用
下面假设 Mind-KernelInfra、算子库（本例中是ascend-op-common-lib）代码都已经准备好，且处于统一级目录下。
1.  使用算子命名空间作为参数， 编译MKI, 并将编译产物拷贝到算子库目录：在这里例子中， 命名空间参数是AtbOps 

-     cd Mind-KerenlInfra
-     bash scripts/build.sh --namespace=AtbOps testframework
-     cp -r output/mki ../ascend-op-common-lib/3rdparty/

2.  编译算子库

-     cd ascend-op-common-lib/
-     bash scripts/build.sh  testframework --no_werror

3.  设施算子库环境， 并开始测试

-     source output/asdops/set_env.sh
-     cd tests/pythontest/optest/
-     msprof --application="python3 mix/test_kvcache.py"


场景3：单算子工程（待补充）
适用于仅简单测试新写的单个算子， 而不想构建完整算子库的用户。
1. 参照example中的算子用例， 实现算子并编写测试用例
2. 编译带example的MKI
3. 测试算子 

#### 参与贡献
0. 申请权限， 在 https://onebox.huawei.com/v/ccd7a576641532864964b7d1ff3c7c15?type=1 中填写账号信息，通知何太航添加权限
1.  Fork 本仓库
- git clone https://gitee.com/ascend/Mind-KernelInfra.git
2.  新建开发分支
- git checkout -B dev
3.  提交代码
- git push origin dev:dev
4.  新建 Pull Requst

#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
