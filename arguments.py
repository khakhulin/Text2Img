import argparse

def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, action='store_true')
    #  Experiment
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--multi-gpu', default=False, action='store_true')
    parser.add_argument('--num_devices', default=1, type=int, help='numbers of gpus')
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--exp_name', default='default', type=str, help='name of the experiments')

    parser.add_argument('--resume_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--gen_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--disc_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--encoder_path', default=None, type=str, help='path to the weights of experiment')

    parser.add_argument('--save_path', default='./models_backup/exp1', type=str, help='path to save models')
    parser.add_argument('--dataset_type', default='birds', type=str, help='type of the dataset')
    parser.add_argument('--data_path', default='dataset/CUB_200_2011', type=str, help='path to dataset')
    #  Training
    parser.add_argument('--use_average_weights', default=False, action='store_true', help='')
    parser.add_argument('--average_weights', default='', type=str, help='path to average generator weights')
    parser.add_argument('--pretrained_text_enc', default='', type=str, help='path to load text encoder weights')
    parser.add_argument('--pretrained_image_enc', default='', type=str, help='path to load image encoder weights')
    parser.add_argument('--pretrained_generator', default='', type=str, help='path to load generator weights')
    parser.add_argument('--continue_from', default='', type=str, help='checkpoint path to start from')
    parser.add_argument('--log_every', default=200, type=int, help='frequency of logging')
    parser.add_argument('--snapshot_every', default=2000, type=int, help='frequency of model saving')
    parser.add_argument('--batch_size', default=8, type=int, help='size of batch in buffer')
    parser.add_argument('--branch_num', default=3, type=int, help='numpber of phase with upscale images')
    parser.add_argument('--max_epoch', default=40, type=int, help='num of training epoch ')
    parser.add_argument('--generator_lr', default=2e-4, type=float, help='learning rate for  generator')
    parser.add_argument('--discriminator_lr', default=2e-4, type=float, help='learning rate for discriminator')
    parser.add_argument('--encoder_lr', default=2e-3, type=float, help='learning rate for discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, help='learning rate for discriminator')
    parser.add_argument('--beta2', default=0.999, type=float, help='learning rate for discriminator')
    parser.add_argument('--smooth_lambda', default=0.3, type=float, help='smooth constant for ranking loss')
    parser.add_argument('--inception_score_on_validation', default=False, type=bool, help='measure inception score on validation')

    #  Model
    parser.add_argument('--embd_size', default=256, type=int, help='size of embedding vectors in gcn')
    parser.add_argument('--text_enc_emb_size', default=128, type=int, help='size of embedding vectors in text encoder')
    parser.add_argument('--encoder_type', default='lstm', type=str, help='type of the text encoder')
    parser.add_argument('--loss_type', default='ls-gan', type=str, help='type of the text encoder')
    parser.add_argument('--is_bert', default=False, action='store_true')
    parser.add_argument('--use_sagan', default=False, action='store_true', help='use self-attention model')
    parser.add_argument('--base_size', default=64, type=int)

    # DAMSM
    parser.add_argument('--gamma1', default=4.0, type=float, help='')
    parser.add_argument('--gamma2', default=5.0, type=float, help='')
    parser.add_argument('--gamma3', default=10.0, type=float, help='')
    parser.add_argument('--damsm_batch_size', default=50, type=int, help='')
    parser.add_argument('--damsm_lr', default=2e-4, type=float, help='')
    parser.add_argument('--damsm_snapshot_interval', default=10, type=int, help='')
    parser.add_argument('--damsm_n_epoch', default=300, type=int, help='')
    parser.add_argument('--damsm_rnn_grad_clip', default=0.25, type=float, help='')
    parser.add_argument('--damsm_text_encoder', default='', type=str, help='')
    parser.add_argument('--damsm_image_encoder', default='', type=str, help='')

    args = parser.parse_args()

    return args