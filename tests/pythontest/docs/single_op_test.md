# 单算子测试场景

## 场景说明

单算子测试场景：指`csv用例`中只存在一个`Operation`，对这个单独的`Opeartion`进行测试。

以下步骤基于您已经完成了算子的开发，需要进行测试。
下面我们假设算子 IR 如下：
>
> - `OperationName`: `MathOperation`
> - `Kernels`: `AddKernel`
> - `OpParam`: `mathType = MATH_ADD(1)`
> - 输入/输出
>
> |名称|输入/输出|形状|数据类型|说明|
> |-|-|-|-|-|
> |x|in|$(n_0,n_1,...n_i)$|float16|第一个输入|
> |y|in|$(n_0,n_1,...n_i)$|float16|第二个输入|
> |out|out|$(n_0,n_1,...n_i)$|float16|输出|
>
> - 计算公式：$out=x+y$

## 测试步骤

### 1.编写unittest测试类

使用继承，搭建最简单的测试脚本：

```python
import unittest
from mkipythontest import OpTest


class TestMathOperationAddKernel(OpTest):
    pass


if __name__ == "__main__":
    unittest.main()
```

注意事项：

1. 脚本中，需要继承`OpTest`类，并且类名必须为`TestXXXOperationYYY`，其中XXX为算子名称。
2. 如果一个测试类专门用于测试一个 Operation 中特定的一个 Kernel，则可在类名之后添加对应标识，即 1 中的YYY。

### 2.补充测试函数

```python
# ...
from mkipythontest.constant import OpType


class TestMathOperationAddKernel(OpTest):
    op_type = OpType.COMPUTE_FLOAT

    def calculate_times(self, in_tensors):
        return in_tensors[0].numel()

    def golden_calc(self, in_tensors):
        return [in_tensors[0] + in_tensors[1]]

    def data_generate(self, in_tensors, **kwargs):
        if 'value' in kwargs:
            return [
                in_tensors[0].fill(kwargs['value']),
                in_tensors[1].fill(kwargs['value'])
            ]
        else:
            return [
                in_tensors[0].uniform(-1, 1),
                in_tensors[1].uniform(-1, 1)
            ]
# ...
```

您需要实现以下函数：

1. `golden_calc`: 用于计算标杆。
2. `data_generate`: 用于生成输入数据。这个函数的默认实现会用$U(-5,5)$的分布随机填充输入张量。 如果自行重载的话，可以在生成一个张量时读取另一个张量的
   shape 等信息。同时，也可通过 csv 用例中的数据生成字段对数据生成流程进行影响。
3. `calculate_times`： 用于计算算子的计算次数，用于精度标准的选取。
4. `op_type`：算子的类型，用于精度标准的选取。也可以使用`@property`动态返回。

- 这些函数的定义可在`mkipythontest.optest`中查看，也可参考已迁移的代码。
- 如果需要测试双标杆算子，第二标杆需要重载`golden_calc_gpu`函数，其定义与`golden_calc`一致。双标杆目前为实验性功能。

### 3. 编写测试用例

我们推荐编写 csv 测试用例，其定义在这里[](./csv_param.md))

同时，也可参考示例代码构建手动用例。

### 4. 将csv用例加载进测试类中

```python
# ...
if __name__ == "__main__":
    load_csv_to_optest(TestMathOperationAddKernel, "./test_add_kernel.csv")
# ...
```

然后运行 python 文件即可。
