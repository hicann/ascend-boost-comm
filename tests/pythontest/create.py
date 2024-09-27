import os

if __name__ == '__main__':
    op_name = "TestOperation"
    dir_name = op_name.replace("Operation", "").lower()
    os.system(f"cp -r example {dir_name}")
    os.system(f"mv {dir_name}/test_example.py {dir_name}/test_{dir_name}.py")
    os.system(f"mv {dir_name}/test_example.csv {dir_name}/test_{dir_name}.csv")
    with open(f"{dir_name}/test_{dir_name}.csv", "w") as f:
        first_line = f.readline(1)