import unittest 

from mkipythontest.case.parser import DefaultCsvParser


class TestDefaultParser(unittest.TestCase):
    
    def test_case(self):
        parser = DefaultCsvParser()
        case_list = parser.parse("./test_default_parser.csv")
        case_0 = case_list[0]
        
if __name__ == "__main__":
    unittest.main()