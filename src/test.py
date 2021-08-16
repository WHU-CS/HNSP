import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--parameters', nargs='+', type=float)

args = parser.parse_args()

for i in args.parameters:
    print(i)
