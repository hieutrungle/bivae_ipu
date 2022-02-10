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
    
    strategy = ipu.ipu_strategy.IPUStrategy()
    global_batch_size = args.batch_size
    print(f"global batch size: {global_batch_size}")
    print(f'Number of IPUs: {num_ipus}')

    with strategy.scope():

        dataio = data_io.Data(args.data_path, global_batch_size)
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
                epochs=epochs
                )
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
    # Genral training options
    parser.add_argument('--model_path', default="./model_output/distributed_training",
                        help="Path to model folder")
    # optimization
    parser.add_argument('--batch_size', type=int, default=10, 
                        help="batch size. default=10")
    parser.add_argument('--epochs', type=int, default=2,
                        help='num of training epochs')
    parser.add_argument('--model_arch', type=str, default='res_wnelu',
                        help='which model architecture to use')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=1,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    # latent variables
    parser.add_argument('--num_channels_of_latent', type=int, default=2,
                        help='number of channels of latent variables')
    # Initial channel
    parser.add_argument('--num_initial_channel', type=int, default=32,
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
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print()
    
    devices = tf.config.list_physical_devices()
    print(devices)
    print(f"Tennsorflow version: {tf.__version__}\n")

    main(args=args)
