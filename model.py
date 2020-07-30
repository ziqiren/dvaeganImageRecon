import tensorflow as tf
from ops import batch_normal, de_conv, conv2d, fully_connect, lrelu
from utils import *
import math

from utils import fMRI
import time

TINY = 1e-8
d_scale_factor = 0.25
g_scale_factor = 1 - 0.75 / 2

class vaegan(object):
    # build model
    def __init__(self, batch_size, data_db_val=None, latent_dim=128, save_path=None):

        self.batch_size = batch_size
        self.data_db_val = data_db_val
        self.latent_dim = latent_dim
        self.channel = 3
        self.image_size = 100
        self.output_size = data_db_val.image_size
        self.vector_dim = 4466
        self.save_path = save_path
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.zp = tf.placeholder(tf.float32, [self.batch_size, self.vector_dim])

        self.ep = tf.random_normal(shape=[self.batch_size, self.latent_dim])
        self.zp2 = tf.random_normal(shape=[self.batch_size, self.latent_dim])

        self.dataset_name = 'S1'
        self.model_name = 'd-vae-gan'

    def build_model_vaegan(self):

        self.z_mean, self.z_sigm = self.Encode(self.images)
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm)) * self.ep)
        self.x_tilde = self.generate(self.z_x, reuse=False)
        self.l_x_tilde, self.De_pro_tilde = self.discriminate(self.x_tilde)

        self.vec_128_mean, self.vec_128_sigm = self.Encode2(self.zp)
        self.vec_128 = tf.add(self.vec_128_mean, tf.sqrt(tf.exp(self.vec_128_sigm)) * self.ep)
        self.x_p = self.generate(self.vec_128, reuse=True)
        self.v_x_tilde, self.G_pro_logits = self.discriminate(self.x_p, True)


        # image
        self.l_x, self.D_pro_logits = self.discriminate(self.images, True)

        # KL loss
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)
        self.kl_loss_v = self.KL_loss(self.vec_128_mean, self.vec_128_sigm)

        # D loss
        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_pro_logits) - d_scale_factor, logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))
        self.D2_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde) - d_scale_factor, logits=self.De_pro_tilde))

        # G loss
        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits) - g_scale_factor, logits=self.G_pro_logits))
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.De_pro_tilde) - g_scale_factor, logits=self.De_pro_tilde))

        # preceptual loss(feature loss)
        self.LL_loss = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.l_x_tilde, self.l_x), [1, 2, 3]))
        self.LL_loss_v2 = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.v_x_tilde, self.l_x_tilde), [1, 2, 3]))
        self.LL_loss_v = tf.reduce_mean(tf.reduce_sum(self.NLLNormal(self.v_x_tilde, self.l_x), [1, 2, 3]))


        # For encode
        self.encode_loss = self.kl_loss / (self.latent_dim * self.batch_size) - self.LL_loss / (4 * 4 * 64)
        # For encode2
        self.encode2_loss_2 = self.kl_loss_v / (self.latent_dim * self.batch_size) - self.LL_loss_v2 / (4 * 4 * 64)

        # for Generater
        self.G1_loss = self.G_tilde_loss - 1e-6 * self.LL_loss
        self.G3_loss = self.G_fake_loss - 1e-6 * self.LL_loss_v
        
        #For discriminater
        self.D1_loss = self.D_tilde_loss + self.D_real_loss
        self.D2_loss = self.D_fake_loss + self.D2_tilde_loss
        self.D3_loss = self.D_fake_loss + self.D_real_loss


        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name and 'e2_' not in var.name]
        self.e2_vars = [var for var in t_vars if 'e2_' in var.name]

        self.all_vars = self.d_vars + self.g_vars  + self.e_vars+ self.e2_vars

        self.saver = tf.train.Saver(max_to_keep=50)

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.relu(conv2d(x_var, output_dim=64, name='dis_conv1'))
            conv2 = tf.nn.relu(
                batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3 = tf.nn.relu(
                batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
            middle_conv = conv4
            conv4 = tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv4 = tf.reshape(conv4, [self.batch_size, -1])

            fl = tf.nn.relu(
                batch_normal(fully_connect(conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl, output_size=1, scope='dis_fully2')

            return middle_conv, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:
            if reuse == True:
                scope.reuse_variables()

            d1 = tf.nn.relu(
                batch_normal(fully_connect(z_var, output_size=13 * 13 * 256, scope='gen_fully1'), scope='gen_bn1',
                             reuse=reuse))
            d2 = tf.reshape(d1, [self.batch_size, 13, 13, 256])
            d2 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 25, 25, 128], name='gen_deconv2'),
                                         scope='gen_bn2', reuse=reuse))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 50, 50, 64], name='gen_deconv3'),
                                         scope='gen_bn3', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 100, 100, 32], name='gen_deconv4'),
                                         scope='gen_bn4', reuse=reuse))
            d5 = de_conv(d4, output_shape=[self.batch_size, 100, 100, 3], name='gen_deconv5', d_h=1, d_w=1)

            return tf.nn.tanh(d5)

    def Encode2(self, vec, reuse=False):

        with tf.variable_scope('encode_v') as scope:
            if reuse == True:
                scope.reuse_variables()

            fc1 = tf.nn.relu(batch_normal(fully_connect(vec, output_size=1024, scope='e2_v_1'), scope='e2_v_bn1'))
            z_mean = fully_connect(fc1, output_size=128, scope='e2_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e2_f3')

            return z_mean, z_sigma

    def Encode(self, x, reuse=False):

        with tf.variable_scope('encode') as scope:
            if reuse == True:
                scope.reuse_variables()

            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2, output_dim=256, name='e_c3'), scope='e_bn3'))
            conv3 = tf.reshape(conv3, [self.batch_size, 256 * 13 * 13])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1, output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

            return z_mean, z_sigma

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def load_training_vectors(self, files, idx):
        if len(files) >1:
            try:
                batch_files = files[idx * self.batch_size:(idx + 1) * self.batch_size]
            except:
                batch_files = files[idx * self.batch_size:]
        elif len(files)==1:
            batch_files = files
        else:
            print("files len is wrong")
            os._exit(0)
        batch_vectors = [load_vectors(f) for f in batch_files]
        batch_vectors = np.reshape(np.array(batch_vectors).astype(np.float32),
                                (len(batch_files), self.vector_dim))
        return batch_vectors

    def load_training_imgs(self, files, idx):
        if len(files) >1:
            try:
                batch_files = files[idx * self.batch_size:(idx + 1) * self.batch_size]
            except:
                batch_files = files[idx * self.batch_size:]
        elif len(files)==1:
            batch_files = files
        else:
            print("files len is wrong")
            os._exit(0)
        batch_imgs = [load_data(f, image_size=self.image_size) for f in batch_files]
        batch_imgs_ = np.reshape(batch_imgs, (len(batch_files), self.image_size, self.image_size, -1))

        return batch_imgs_, batch_files

    def save(self, sess, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def infer(self, model_path):
        val_images_name = self.data_db_val.data_list
        val_fmris_name = [name.replace("B", "A").replace("JPEG", "mat") for name in val_images_name]
        assert len(val_images_name) == len(val_fmris_name)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print('[*] test data loaded successfully')
        print("# test data len for stage 2 / 3 : %d  " % (len(val_fmris_name)))

        with tf.Session(config=config) as sess:
            sess.run(init)
            if model_path:
                print(" [*] Loading to  {}".format(model_path))
                ckpt = tf.train.get_checkpoint_state(model_path)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(sess, os.path.join(model_path, ckpt_name))
                print(" [*] Success to read {}".format(ckpt_name))
            else:
                print('[*] model is missed')
                os._exit(0)
            i = 0
            epoch_size = int(len(val_images_name)/self.batch_size)
            # remove overlap images:
            # name_set = set()
            for idx in range(0, epoch_size):
                batch_images, names = self.load_training_imgs(val_images_name, idx)

                batch_z = self.load_training_vectors(val_fmris_name, idx)

                next_x_images = batch_images

                fd = {self.zp: batch_z}

                recon_images = sess.run(self.x_p, feed_dict=fd)

                save_path_recon = os.path.join(self.save_path, "recon")
                save_path_real = os.path.join(self.save_path, "real")
                save_images_by_name(next_x_images,save_path_real,names)
                save_images_by_name(recon_images,save_path_recon,names)


