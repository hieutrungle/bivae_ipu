import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import configargparse
from utils import utils
# from hard_code_test_model import VariationalAutoencoder
from model import VariationalAutoencoder
import data_io
import timeit
import tensorflow as tf
from tensorflow.python import ipu


def main(args):
    
    num_ipus = args.num_ipus
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = num_ipus
    config.configure_ipu_system()
    # strategy = tf.distribute.MirroredStrategy()
    strategy = ipu.ipu_strategy.IPUStrategy()
    global_batch_size = args.batch_size
    # global_batch_size = batch_size * strategy.num_replicas_in_sync
    # global_batch_size = batch_size * num_ipus
    print(f"global batch size: {global_batch_size}")
    print(f'Number of IPUs: {num_ipus}')

    with strategy.scope():

        dataio = data_io.Data(args.data_path, global_batch_size, args.tile_size)
        data = dataio.load_data(args.dataset)
        
        # Model Initialization
        in_shape = list(dataio.data_dim[1:])
        model_arch = utils.get_model_arch(args.model_arch)
        print(f"model_arch: {args.model_arch}")

        vae = VariationalAutoencoder(args, model_arch, global_batch_size, in_shape)
        print(f"is using se: {vae.use_se}\n")
        vae.model().summary()

        # Set up for training, evaluation, or generation
        model_path = args.model_path
        print(f"\nlogging information to: {model_path}\n")
        
            
        # sys.exit()
        # Training
        # Training parameters
        epochs = args.epochs
        steps_per_execution = dataio.data_dim[0] // global_batch_size

        start_training = timeit.default_timer()
        vae.compile(optimizer='adamax', loss="mse", steps_per_execution=steps_per_execution)
        history = vae.fit(
                data,
                batch_size=global_batch_size,
                epochs=epochs,
                verbose=2)
        print(f"Training time: {timeit.default_timer()-start_training}")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    # IPUs
    parser.add_argument('--num_ipus', type=int, default=1,
                        help='Number of IPUs')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cesm', 'isabel'],
                        help='which dataset to use, default="mnist')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--tile_size', type=int, default=64,
                        help="tile size after partitioning scientific dataset")
    # Genral training options
    parser.add_argument('--eval', action='store_true', default=False,
                        help="run evaluation on testing dataset")
    parser.add_argument('--save_encoding', action='store_true', default=False,
                        help="save encoding vectors during eval")
    parser.add_argument('--generate', action='store_true', default=False,
                        help="run generation")
    parser.add_argument('--model_path', default="./model_output/distributed_training",
                        help="Path to model folder")
    parser.add_argument('--path_img_output', default=None,
                        help="Path to image output folder when generating new images")
    parser.add_argument('--train_portion', type=float, default=0.95,
                        help="train portion after spliting the original dataset")
    # logging options
    # parser.add_argument('--experiment_name', type=str, required=True,
    #                help='path to directory where checkpoints & tensorboard events will be saved.')
    parser.add_argument('--epochs_til_ckpt', type=int, default=5,
                        help="Epochs until checkpoint is saved")
    parser.add_argument('--steps_til_summary', type=int, default=50,
                        help="Number of iterations until tensorboard summary is saved")
    parser.add_argument('--logging_root', type=str, default='./logs',
                        help="root for logging")
    # optimization
    parser.add_argument('--batch_size', type=int, default=10, 
                        help="batch size. default=32")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=5e-5,
                        help='min learning rate')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--model_arch', type=str, default='res_wnelu',
                        help='which model architecture to use')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.4,
                        help='The portions epochs that KL is annealed')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=1,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    # latent variables
    parser.add_argument('--num_channels_of_latent', type=int, default=1,
                        help='number of channels of latent variables')
    # Initial channel
    parser.add_argument('--num_initial_channel', type=int, default=36,
                        help='number of channels in pre-enc and post-dec')
    # Share parameter of preprocess and post-process blocks
    parser.add_argument('--num_process_blocks', type=int, default=1,
                        help='number of preprocessing and post-processing blocks')
    # Preprocess cell
    parser.add_argument('--num_preprocess_cells', type=int, default=2,
                        help='number of cells per proprocess block')
    # Encoder and Decoder Tower
    parser.add_argument('--num_scales', type=int, default=2,
                        help='the number of scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=1,
                        help='number of groups per scale')
    parser.add_argument('--is_adaptive', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_cell_per_group_enc', type=int, default=1,
                        help='number of cells per group in encoder')
    # decoder parameters
    parser.add_argument('--num_cell_per_group_dec', type=int, default=1,
                        help='number of cell per group in decoder')
    # Post-process cell
    parser.add_argument('--num_postprocess_cells', type=int, default=2,
                        help='number of cells per post-process block')
    # Squeeze-and-Excitation
    parser.add_argument('--use_se', action='store_true', default=True,
                        help='This flag enables squeeze and excitation.')
    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--iter', type=int, default=0,
                        help='resume iteration')
    args = parser.parse_args()

    if (args.generate and (args.model_path is None or args.path_img_output is None or args.iter is None)):
        parser.error('The --generate argument requires the --model_path and --path_img_output')

    if (args.resume and args.iter is None):
        parser.error('The --resume argument requires the --iter')

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print()
    
    devices = tf.config.list_physical_devices()
    print(devices)
    print(f"Tennsorflow version: {tf.__version__}\n")

    main(args=args)
