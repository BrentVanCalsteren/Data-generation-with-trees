
from interval import *

def test_intervals():
    int_1 = (False, 0,1,True)
    int_2 = (True, -1,0,False)
    int_3 = (False, 0.5,1.5,True)
    int_4 = (True, -1.5,0.5,False)
    int_list_1 = [int_1, int_2]
    int_list_2 = [int_3, int_4]

    print(f'result simple intersect:{intersect(int_1, int_2)}')
    print(f'result simple intersect:{intersect(int_1, int_3)}')
    print(f'result union intersect:{intersect_union_interval_lists(int_list_1, int_list_2)}')
    print(f'result union intersect:{intersect_union_interval_lists(int_list_1, int_list_2)}')



if __name__ == '__main__':
    test_intervals()