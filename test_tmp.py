import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
        '--storage', '-m', default=20,
        help='The Memory Storage')
parser.add_argument(
        '--mini_batch', '-batch', default=512,
        help='minibatch size')
parser.add_argument(
        '--whole_gradient', action='store_true',default=False,
        help='Compute the gradient using all data')
args = parser.parse_args()

m = int(args.storage)
# minibatch size
minibatch = int(args.mini_batch)
# use entire data to compute gradient
use_whole_data_for_gradient = args.whole_gradient

print(m,minibatch)