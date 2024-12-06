# csv字段说明

测试用例的字段如下：

| 名称            | 描述          | 形式                                                                                                               |
|---------------|-------------|------------------------------------------------------------------------------------------------------------------|
| CaseNum       | 用例编号        | `递增的int`                                                                                                         |
| CaseName      | 用例名称        | `str`                                                                                                            |
| OpName        | Operation名称 | `str`                                                                                                            |
| OpParam       | Operation参数 | `dict`/`json`                                                                                                    |
| In(Out)Num    | 输入/输出数量     | `int`                                                                                                            |
| In(Out)DType  | 输入/输出数据类型   | 参见`mkipythontest.constant.TENSOR_DTYPE_DICT`，多个Tensor间用;分隔                                                       |
| In(Out)Shape  | 输入/输出形状     | `tuple[str, int]`，多个Tensor间用;分隔；当使用 str时，                                                                        |
| In(Out)Format | 输入/输出格式     | 参见`mkipythontest.constant.TensorFormat`，多个Tensor间用;分隔；并非将输入转化为对应格式，只是在运行时指定输入的格式                                 |
| DataGenerate  | 数据生成方式      | 可使用`torch`，`numpy`或`numpy.random`中的函数，会自动应用shape；若留空或输入无效内容，则使用测试类中重载的`data_generate`函数                          |
| Env           | 环境变量        | `dict`，某些特殊算子需要使用环境变量                                                                                            |
| IOMux         | 输入输出复用      | 原地操作型算子使用，以 `O:I`形式编写，表示 第 `O` 个输出复用第 `I` 个输入的地址空间；多个输入复用用`;`分割                                                  |
| SocVersion    | 芯片型号        | `Ascend310P`/`Ascend910B`/`Ascend910`，多个芯片型号用`;`分隔，支持型号可在`mkipythontest.utils.soc`查看；型号前添加`!`表示对此平台跳过用例，两种写法不可混用 |
| ExpectedError | 期望错误类型      | 参见`mkipythontest.constant.ErrorType`，该文件定义与MKI CPP 侧一致，可拦截指定错误；目前尚未支持精细的错误类型，只能拦截错误与否                            |
