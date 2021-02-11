import os

from pine import MNIST_PINE

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

### parsing and configuration ###

def parse_args():
    desc = "PINE implementation with Tensorflow"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--main_model', type=str, default='mnist_model_no1', help='Select the Main Model', choices=['mnist_model_no1', 'mnist_model_no2'], required=True)
    parser.add_argument('--interpreter', type=str, default='mnist_interpreter_no1', choices=['mnist_interpreter_no1','mnist_interpreter_no2'], help='Select the Interpreter', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'mnist', 'fashion-mnist', 'celebA'],
                        help='Select the dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the interpretation')

    return check_args(parser.parse_args())

### checking arguments ###

def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)


    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args

### main ###

def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    main_models = ['mnist_model_no1', 'mnist_model_no2', 'mnist_model_no3']
    interpreters = ['mnist_interpreter_no1', 'mnist_interpreter_no2', 'mnist_interpreter_no3']

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        pine = None
        for main_model in main_models:
            for interpreter in interpreters:
                if args.main_model == main_model and args.interpreter == interpreter :
                    pine = MNIST_PINE(sess, main_model=args.main_model, interpreter=args.interpreter,
                                epoch=args.epoch,
                                batch_size=args.batch_size,
                                dataset_name=args.dataset,
                                checkpoint_dir=args.checkpoint_dir)
        if pine is None:
            raise Exception("[!] There is no option for " + args.main_model)

        # build graph
        pine.build_pine()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        pine.train()
        print(" [*] Training finished!")

        # visualize learned generator
        pine.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
