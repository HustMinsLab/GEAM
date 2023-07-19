from train import *
import argparse

def parse_args():
    # input arguments    
    parser = argparse.ArgumentParser(description='GEAM') 
    
    # aucs mLFR500_mu0.2
    parser.add_argument('--dataset', nargs='?', default='mLFR500_mu0.2')
    # ppr heat
    parser.add_argument('--diffusion', nargs='?', default='ppr')
    parser.add_argument('--initial_feat', nargs='?', default='deepwalk')

    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--nb_epochs', type=int, default=101)
    parser.add_argument('--hid_units', type=int, default=512)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--student_t_v', type=int, default=1)
    parser.add_argument('--num_cluster', type=int, default=3)
    parser.add_argument('--with_gt', default=False)
    parser.add_argument('--test_Q', default=True)
    parser.add_argument('--perEpoch_Q', type=int, default=10)

    return parser.parse_known_args()



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    args, unknown = parse_args()

    if 'mLFR' in args.dataset:
        args.num_layer = 4
        args.num_cluster = 3
        args.sample_size = 500
        args.with_gt = True
    if 'aucs' in args.dataset:
        args.num_layer = 5
        args.num_cluster = 6
        args.sample_size = 61
        args.with_gt = False
      
    print(args)
    train(args)
