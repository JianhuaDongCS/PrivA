import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run model.")
    parser.add_argument('--weights_path', nargs='?', default='./pretrain/',
                        help='Store model path.')
    parser.add_argument('--ageset_path', nargs='?', default='./Generate_data/',
                        help='Store ageset path.')
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k/',
                        help='Choose a dataset from {Book-Crossing/, ml-100k/,ml-1m/}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight_decay.')
    parser.add_argument('--regs1', nargs='?', default='[1e-2]',
                        help='Regularizations1.')
    parser.add_argument('--regs2', nargs='?', default='[1e-5]',
                        help='Regularizations2.')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate.')
    parser.add_argument('--G_lr', type=float, default=0.0001,
                        help='Generator learning rate.')
    parser.add_argument('--c', type=float, default=0.0,
                        help='privacy settings.')
    parser.add_argument('--lambdal', type=float, default=1.0,
                        help='privacy settings.')
    parser.add_argument('--lambdas', type=float, default=1.0,
                        help='privacy settings.')
    parser.add_argument('--attribute_dim', type=int, default=21,
                        help='attribute_dim.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--Ks', nargs='?', default='[10]',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
