import tensorflow as tf
from utils import mkdir_p, fMRI
from model import vaegan
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_integer("batch_size" , 16, "batch size")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent space")
flags.DEFINE_string("path", './data/S1/', "the directory of your data")
flags.DEFINE_string("phase", 'val', "train or val chose dataset")
flags.DEFINE_string("save_path", './output/infer/', "the validation result")
flags.DEFINE_string("model_path", './model/stage3/d-vae-gan/', " path to best model dir")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    batch_size = FLAGS.batch_size
    latent_dim = FLAGS.latent_dim
    model_path = FLAGS.model_path
    save_path  = FLAGS.save_path
    data_db_val = fMRI(FLAGS.path, "val", batch_size)

    vaeGan = vaegan(batch_size= batch_size, data_db_val = data_db_val, 
                    latent_dim=latent_dim, save_path=save_path)


    vaeGan.build_model_vaegan()

    vaeGan.infer_batch(model_path=model_path)










