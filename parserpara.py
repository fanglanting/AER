import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('RAP-AD')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        default='mooc')
    parser.add_argument('--n_degree', nargs='*', default=['32', '32'],
                        help='[l, s], history length and partner size')
    parser.add_argument('-s', '--step', type=int, help='number of interactions used for prediction', default=5)
    parser.add_argument('--raps', type=str, help='using source node encoding (on/None)', default='on')
    parser.add_argument('--rapd', type=str, help='using destination node encoding (on/None)', default='on')
    parser.add_argument('--mask', nargs='*', default=[0,0], help='mask numbers')
    parser.add_argument('-g','--g_neg', type=bool, default=False, help='load augmentated data')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv