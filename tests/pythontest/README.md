# 算子Python测试框架

基于unittest的测试框架，提供算子python调用接口和完善的测试流程，开发者和测试者只需编写用例和几个函数就可以测试对应算子。

目前测试框架转为pypi包调用，不必再担心格式化测试代码后path混乱的问题。

## 快速入门

### 1. 编写算子测试类继承`OpTest`基类

需重载以下函数：

- `calculate_times`: 计算算子计算次数。
- `golden_calc`: 生成标杆数据。
- `data_generate`: 自定义Tensor中数值生成逻辑，在csv的`DataGenerate`字段中填写`custom`
  以指定使用此函数生成用例数据。可以按`custom(key1=value1,key2=value2)`这样的格式进行编写，以传递进一些自定义参数。
- `golden_compare`: 目前绝大多数算子无需重载该函数，现在已经对接完善的新精度标准。在特殊情况下，也可重载。

### 2. 编写`test_case`

#### 方法一、手写`test`函数体

定义`test`开头的函数，在其中生成op_param和tensor等数据，然后执行execute。
execute会自动处理golden比较等逻辑。

#### 方法二、构造test对象，然后用run_case运行

方法三中的csv加载实质上是通过一系列代码将csv文件转换成case对象，然后用run_case运行。因此，可以手动构造case对象，然后运行case。

#### 方法三、编写csv用例并使用装饰器加载

使用装饰器`case_inject`装饰测试类，从csv文件读取op_param和tensor等数据，然后执行execute。

- 装饰器`case.injector.case_inject`可以接收参数`csv_path`
  ，用于加载指定csv文件或者指定目录下的csv文件。不指定时，默认加载同级目录下的`.csv`文件。

- 以上两种样例读取方式可以混用。

- 为兼容外部调用情况，当python3运行的文件和测试类所在文件不是同一文件时，注入器将自行失效。

- 非必要不建议手写test函数。如需要手写test_case，请注意以下几点，以便问题排查：
  - 注意代码规范
  - 遵循测试框架编程范式
  - 不使用非必要的额外类变量
  - 不打印非必要的info级别日志

### 3. 运行测试

使用python直接运行测试代码。

### 4. 注册入口

为支持类似ATB的混合Operation测试，设计了测试类管理器，使用测试类管理器，可一次运行多个Operation的测试用例。

测试类以OpName和部分OpParam为查找键。

下面给出例程：

```python
manager = BatchTestManager()
manager.register_test_class(TestOp1Operation, "Op1")
manager.register_test_class(TestOp2OperationKernel1, "Op2", {'KernelType': 1})
manager.register_test_class(TestOp2OperationKernel2, "Op2", {'KernelType': 2})
manager.add_test_csv("test_cases_1.csv")
manager.add_test_csv("test_cases_2.csv")
manager.all_test_run()
```

## 框架目录结构

```bash
tests/pythontest/
├── example             # 测试编写示例
│   ├── __init__.py
│   └── test_example_function.csv
├── case
│   ├── __init__.py     # 用例类
│   ├── injector.py     # 用例注入器
│   └── parser.py       # 用例加载器
├── utils               # 工具
│   └── binfile.py  # 读写二进制Tensor工具
├── constant.py         # 一些常量，例如算子种类、数据类型
├── op_test.py          #测试基类
└── README.md

```

## 样例字段说明

| 名称             | 描述                | 形式                                                                                                                           |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| CaseNum          | 用例编号            | `递增的int`                                                                                                                    |
| CaseName         | 用例名称            | `str`                                                                                                                          |
| OpName           | Operation名称       | `str`                                                                                                                          |
| OpType           | 算子类型            | 参考`constant.py`                                                                                                              |
| OpParam          | Operation参数       | `dict`/`json`                                                                                                                  |
| In(Out)Num       | 输入/输出数量       | `int`                                                                                                                          |
| In(Out)DType     | 输入/输出数据类型   | 参见`constant.py`，多个Tensor间用;分隔                                                                                         |
| In(Out)Shape     | 输入/输出形状       | `tuple[str, int]`，多个Tensor间用;分隔                                                                                         |
| In(Out)Format    | 输入/输出格式       | 保留字段，尚不支持nd以外的格式，可留空                                                                                         |
| In(Out)TensorBin | 输入/输出二进制文件 | `str`，文件名，会自动读取文件。不为空时，直接忽略之前的参数，转而读取二进制文件。                                              |  |  |
| DataGenerate     | 数据生成方式        | 可使用`torch`，`numpy`或`numpy.random`中的函数，会自动应用shape；若留空或输入无效内容，则使用测试类中重载的`data_generate`函数 |
| TestType         | 测试类型            | `"Function"`/`"Generalization"`/`Performance`                                                                                  |
| SocVersion       | 芯片型号            | `Ascend310P`/`Ascend910B`/`Ascend310B`/`Ascend910`                                                                             |
| ExpectedError    | 期望错误类型        | 保留字段，尚未支持                                                                                                             |

## 进阶使用

### shape泛化机

- shape可填写`占位符`，以进行验收标准测试。编写规则如下：
  - 占位符遵循大多数编程语言中的变量名称原则，即非数字开头的大小写字母加数字加`_`的组合。
  - 同时，也可以输入`占位符表达式`，完成更复杂的测试。
  - 表达式支持以下运算符：+ - * / % ** //

- 生成泛化shape时，会根据输入输出shape中的所有不同维度表达式中的占位符进行全排列。

- 当使用字母+符号+数字组合时，将会在生成占位符全排列后，用这个表达式进行一次计算，计算结果为最终shape中的dim。
  - 例1：InShape为`X,Y`，OutShape为`X,1`，泛化数组为`[2,4]`
    - (2,2);(2,1)
    - (2,4);(2,1)
    - (4,2);(4,1)
    - (4,4);(4,1)
  - 例2：InShape为`X*2,3`，OutShape为`X/2,3`，泛化数组为`[2,4,6]`
    - (4,3);(1,3)
    - (8,3);(2,3)
    - (12,3);(3,3)
  - 例3：InShape为`X,3;Y,3;Z;3`，OutShape为`X+Y+Z,3`，泛化数组为`[2]`
    - (2,3) (2,3) (2,3);(6,3)

### 特殊生成器

numpy/torch中有许多方便的tensor生成工具，DataGenerate字段中，可以编写函数调用式语句，以调用这些丰富的生成规则。

例：均匀分布：`uniform(low=-5,high=5)`

系统将自动传入shape并将结果转换成torch.Tensor。

-
    1. 你必须使用kwargs传入参数。
-
    2. 系统将从torch、numpy.random、numpy中寻找对应函数。

### dump数据测试

csv的In(Out)BinFile字段可支持Tensor数据的传入传出。

有以下使用场景：

1. 保存某一次的测试输入输出到文件
   这种情况下，需要同时填写这两个字段之前的用例数据和这两个路径。用例运行完后，会将dump出的tensor文件写入两个路径。
2. 以特定bin文件作为用例进行测试
    这种情况下，将这两个字段直接留白。框架将自动加载数据。

### 使用第三方用例

参考`case/parser.py`，解析第三方用例到`Case`类。

（待补充）

## TODOList

- [ ] format转换
- [ ] bin读写
- [ ] 总测试entry，自动扫描测试类 这个放commonlib
- [ ] 期望出错
- [ ] 新精度标准完全介入
- [ ] 第二标杆读取
