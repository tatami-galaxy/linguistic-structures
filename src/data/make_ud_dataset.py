from argparse import ArgumentParser
from datasets import load_dataset

# node distance
# node depth


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('--', type=str)
    cli_args = argp.parse_args() 
    print(cli_args.test_arg)

