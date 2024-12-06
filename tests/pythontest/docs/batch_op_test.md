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
