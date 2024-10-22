from mkipythontest.manager import BatchTestSuite

manager = BatchTestManager()
manager.register_test_class(TestOp1Operation, "Op1")
manager.register_test_class(TestOp2OperationKernel1, "Op2", {'KernelType': 1})
manager.register_test_class(TestOp2OperationKernel2, "Op2", {'KernelType': 2})
manager.add_test_csv("test_cases_1.csv")
manager.add_test_csv("test_cases_2.csv")
manager.all_test_run()

if __name__ == "__main__":
    pass