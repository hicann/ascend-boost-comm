# 算子Python测试框架

基于unittest的测试框架，提供算子python调用接口和完善的测试流程，开发者和测试者只需编写用例和几个函数就可以测试对应算子。

## 使用说明

### 1. 编写算子测试类继承optest基类

- `golden_calc`: 生成标杆数据。除了随机生成类算子外，都需要重载这个函数。
- `golden_compare`: 算子结果与标杆比较。默认为`torch.allclose(rtol=0, atol=0)`,若允许一定误差，或有更高标准，可以重载。
- `custom`: 自定义Tensor中数值生成逻辑。如果numpy的语句无法满足需求，可以在csv样例中指定custom重载custom函数。

### 2. 编写`test_case`

#### 方法一、手写`test`函数体

定义`test`开头的函数，在其中生成op_param和tensor等数据，然后执行execute。
execute会自动处理golden比较等逻辑。

#### 方法二、编写csv用例并使用装饰器加载

使用装饰器`CaseInject`装饰测试类，从csv文件读取op_param和tensor等数据，然后执行execute。

- 装饰器`case.injector.CaseInject`可以接收参数`csv_path`，用于加载指定csv文件或者指定目录下的csv文件。不指定时，默认加载同级目录下的`.csv`文件。

- 以上两种样例读取方式可以混用。

- 并不推荐在手写test_case函数时进行随机shape的生成。这样每次的测试用例不一致，容易引起ci偶挂等问题。请务必使用算子验收标准编写基本覆盖全面的测试用例。

### 3. 运行测试

使用python直接运行测试代码。

## 框架目录结构

```bash
tests/pythontest/
├── example             # 测试编写示例
│   ├── __init__.py
│   └── test_example_function.csv
├── case
│   ├── __init__.py     # 用例类
│   ├── injector.py     # 用例注入器
│   ├── parser.py       # 用例加载器
│   └── runner.py       # 用例运行函数
├── constant.py         # 一些常量，例如算子种类、数据类型
├── op_test.py          #测试基类
└── README.md

```

## 样例字段说明

| 名称          | 描述              | 形式                                                                                        |
| ------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| CaseNum       | 用例编号          | `递增的int`                                                                                   |
| CaseName      | 用例名称          | `str`                                                                                         |
| OpName        | Operation名称     | `str`                                                                                         |
| TacticName    | Tactic名称        | `str`                                                                                         |
| OpParam       | Operation参数     | `dict`/`json`                                                                                   |
| In(Out)Num    | 输入/输出数量     | int                                                                                         |
| In(Out)DType  | 输入/输出数据类型 | 参见`constant.py`，多个Tensor间用;分隔                                                      |
| In(Out)Shape  | 输入/输出形状     | `tuple[int]`，多个Tensor间用;分隔                                                             |
| In(Out)Format | 输入/输出格式     | 保留字段，尚不支持nd以外的格式，可留空                                                      |
| DataGenerate  | 数据生成方式      | 可使用`numpy`或`numpy.random`中的函数，会自动应用shape；若留空或输入无效内容，则使用测试类中重载的`custom`函数 |
| TestType      | 测试类型          | `"Function"`                                                                                    |
| SocVersion    | 芯片型号          | `Ascend310P`/`Ascend910B`                                                                       |
| ExpectedError | 期望错误类型      | 保留字段，尚未支持                                                                          |

## 高级用法

### 使用第三方用例

参考`case/parser.py`，解析第三方用例到`Case`类。

（待补充）