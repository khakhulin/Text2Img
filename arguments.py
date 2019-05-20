import argparse


def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, action='store_true')
    #  Experiment
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--multi-gpu', default=False, action='store_true')
    parser.add_argument('--num_devices', default=1, type=int, help='numbers of gpus')
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--exp_name', default='default', type=str, help='name of the experiments')

    parser.add_argument('--resume_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--gen_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--disc_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--encoder_path', default=None, type=str, help='path to the weights of experiment')

    parser.add_argument('--save_path', default='./models_backup/exp1', type=str, help='path to save models')
    parser.add_argument('--datasets', default='birds', type=str, help='type of the dataset')

    #  Training
    parser.add_argument('--log_every', default=1, type=int, help='frequency of logging')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch in buffer')
    parser.add_argument('--branch_num', default=3, type=int, help='numpber of phase with upscale images')
    parser.add_argument('--max_epoch', default=20, type=int, help='num of training epoch ')
    parser.add_argument('--generator_lr', default=1e-3, type=float, help='learning rate for  generator')
    parser.add_argument('--discriminator_lr', default=1e-3, type=float, help='learning rate for discriminator')
    parser.add_argument('--encoder_lr', default=2e-3, type=float, help='learning rate for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='learning rate for discriminator')
    parser.add_argument('--beta2', default=0.999, type=float, help='learning rate for discriminator')

    #  Model
    parser.add_argument('--embd_size', default=64, type=int, help='size of embedding vectors in gcn')
    parser.add_argument('--encoder_type', default='lstm', type=str, help='type of the text encoder')

    args = parser.parse_args()

    return args