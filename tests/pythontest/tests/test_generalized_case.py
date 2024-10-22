import unittest
from mkipythontest.case import GeneralizedCase


class TestGeneralizedCase(unittest.TestCase):
    def test_single_placeholder(self):
        generalized_case = GeneralizedCase(specific_case_name="test_single_placeholder", in_num=1, out_num=1)
        generalized_case.in_shapes = [('X',)]
        generalized_case.out_shapes = [('X',)]
        dims = [1, 4, 16, 64, 256, 1024, 16384, 131072, 1048576]
        cases = generalized_case.to_case_list(dims)
        in_shapes = [c.in_shapes for c in cases]
        out_shapes = [c.out_shapes for c in cases]
        golden_shapes = [[(dim,)] for dim in dims]
        self.assertEqual(in_shapes, golden_shapes)
        self.assertEqual(out_shapes, golden_shapes)

    def test_double_placeholder(self):
        generalized_case = GeneralizedCase(specific_case_name="test_double_placeholder", in_num=1, out_num=1)
        generalized_case.in_shapes = [('X', 'Y')]
        generalized_case.out_shapes = [('X',)]
        dims = [1, 4, 16]
        cases = generalized_case.to_case_list(dims)
        in_shapes = sorted([c.in_shapes for c in cases])
        out_shapes = sorted([c.out_shapes for c in cases])
        golden_in_shapes = sorted(
            [[(1, 1)], [(1, 4)], [(1, 16)], [(4, 1)], [(4, 4)], [(4, 16)], [(16, 1)], [(16, 4)], [(16, 16)]])
        golden_out_shapes = sorted([[(1,)], [(4,)], [(16,)]] * 3)
        self.assertEqual(in_shapes, golden_in_shapes)
        self.assertEqual(out_shapes, golden_out_shapes)

    def test_op_single(self):
        generalized_case = GeneralizedCase(specific_case_name="test_op", in_num=1, out_num=1)
        generalized_case.in_shapes = [('X',)]
        generalized_case.out_shapes = [('X+1', 'X-1', 'X*2', 'X//2', 'X%2')]
        dims = [5, 16]
        cases = generalized_case.to_case_list(dims)
        shapes = [c.all_shapes for c in cases]
        golden_shapes = [
            [(5,), (6, 4, 10, 2, 1)], [(16,), (17, 15, 32, 8, 0)]
        ]
        self.assertEqual(shapes, golden_shapes)

    def test_op_param(self):
        generalized_case = GeneralizedCase(specific_case_name="test_op_param", in_num=1, out_num=1)
        generalized_case.in_shapes = [('X',)]
        generalized_case.out_shapes = [('X',)]
        generalized_case.op_param = {
            'test_param': 'X'
        }
        dims = [5, 16]
        cases = generalized_case.to_case_list(dims)
        shapes = [c.all_shapes for c in cases]
        op_params = [c.op_param for c in cases]
        golden_shapes = [
            [(5,), (5,)], [(16,), (16,)]
        ]
        golden_op_params = [
            {'test_param': 5}, {'test_param': 16}
        ]
        self.assertEqual(shapes, golden_shapes)
        self.assertEqual(op_params, golden_op_params)


if __name__ == '__main__':
    unittest.main()
