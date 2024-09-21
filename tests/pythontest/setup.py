from setuptools import setup, find_packages

setup(
    name='mkipythontest',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch","numpy","pandas"
    ],
    author='yuantao',
    author_email='taoyuan15@h-partners.com',
    description='Mind-KernelInfra python test framework',
    license='Mulan PSL',
    keywords='mki python test',
    url='https://gitee.com/Ascend/Mind-KernelInfra'
)
