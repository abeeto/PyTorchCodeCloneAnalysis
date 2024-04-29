# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# <Project Name>     Ziang.Chen@kcl.ac.uk
#------------------------------------------------------------------
import argparse
import tqdm
import unittest
import colour


def test_Function():
    assert 1 == 1




def arg_parsing():
    parser = argparse.ArgumentParser(description="Ziang's Template Python Programme")
    parser.add_argument('file', help='File path')
    parser.add_argument('--name', type=str, help="some demo name" ,default='name')
    parser.add_argument('--save-path', type=str, default='./')
    parser.add_argument('--mode', type=str, help="program model", default='evaluation')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args=arg_parsing()

    if args.mode == 'evaluation':
        with tqdm.tqdm(total=100) as bar:
            import time
            for i in range(100):
                time.sleep(0.5)
                bar.update(1)
    
    if args.mode == 'test':
        unittest.FunctionTestCase(test_Function)
        print("Test Passed")










