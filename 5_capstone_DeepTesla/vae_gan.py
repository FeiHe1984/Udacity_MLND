from keras import callbacks as cbks
import time
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
from layers import Deconv2D
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Activation, Convolution2D, LeakyReLU, Flatten, BatchNormalization as BN
from keras.models import Sequential, Model
from keras import initializations
from functools import partial
learning_rate = .0002
beta1 = .5
z_dim = 512
normal = partial(initializations.normal, scale=.02)

"""
Modified from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
"""
import os
import math
import random
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_images(images, size, image_path, gray=False):
    return imsave(inverse_transform(images), size, image_path, gray=gray)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size, gray=False):
    h, w = images.shape[1], images.shape[2]
    if gray:
      img = np.zeros((h * size[0], w * size[1]))
    else:
      img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


def imsave(images, size, path, gray=False):
    return scipy.misc.imsave(path, merge(images, size, gray=gray))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def save(sess, saver, checkpoint_dir, step, name):
  """Save tensorflow model checkpoint"""
  model_name = name
  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, saver, checkpoint_dir, name):
  """Load tensorflow model checkpoint"""
  print(" [*] Reading checkpoints: {}".format(checkpoint_dir))

  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Checkpoints read: {}".format(ckpt_name))
    return True
  else:
    print(" [!] Failed reading.")
    return False


def train_model(name, g_train, d_train, sampler, generator, samples_per_epoch, nb_epoch,
                z_dim=100, verbose=1, callbacks=[],
                validation_data=None, nb_val_samples=None,
                saver=None):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    self = {}
    epoch = 0
    counter = 0
    out_labels = ['g_loss', 'd_loss', 'd_loss_fake', 'd_loss_legit', 'time']  # self.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + callbacks + [history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)

    callbacks.set_params({
        'nb_epoch': nb_epoch,
        'nb_sample': samples_per_epoch,
        'verbose': verbose,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    while epoch < nb_epoch:
      callbacks.on_epoch_begin(epoch)
      samples_seen = 0
      batch_index = 0
      while samples_seen < samples_per_epoch:
        z, x = next(generator)
        # build batch logs
        batch_logs = {}
        if type(x) is list:
          batch_size = len(x[0])
        elif type(x) is dict:
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        t1 = time.time()
        d_losses = d_train(x, z, counter)
        z, x = next(generator)
        g_loss, samples, xs = g_train(x, z, counter)
        outs = (g_loss, ) + d_losses + (time.time() - t1, )
        counter += 1

        # save samples
        if batch_index % 100 == 0:
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/train_%s_%s.png' % (name, epoch, batch_index))

          samples, xs = sampler(z, x)
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/test_%s_%s.png' % (name, epoch, batch_index))
        
      
        for l, o in zip(out_labels, outs):
            batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        # construct epoch logs
        epoch_logs = {}
        batch_index += 1
        samples_seen += batch_size

      if saver is not None:
        saver(epoch)

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1
    
    # _stop.set()
    callbacks.on_train_end()
    return sampler

def mean_normal(shape, mean=1., scale=0.02, name=None):
    return K.variable(np.random.normal(loc=mean, scale=scale, size=shape), name=name)


def cleanup(data):
  X = data[0][:64, -1]
  X = np.asarray([cv2.resize(x.transpose(1, 2, 0), (160, 80)) for x in X])
  X = X/127.5 - 1.
  Z = np.random.normal(0, 1, (X.shape[0], z_dim))
  return Z, X


def generator(batch_size, gf_dim, ch, rows, cols):

    model = Sequential()

    model.add(Dense(gf_dim*8*rows[0]*cols[0], batch_input_shape=(batch_size, z_dim), name="g_h0_lin", init=normal))
    model.add(Reshape((rows[0], cols[0], gf_dim*8)))
    model.add(BN(mode=2, axis=3, name="g_bn0", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim*4, 5, 5, subsample=(2, 2), name="g_h1", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn1", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim*2, 5, 5, subsample=(2, 2), name="g_h2", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn2", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim, 5, 5, subsample=(2, 2), name="g_h3", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn3", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(ch, 5, 5, subsample=(2, 2), name="g_h4", init=normal))
    model.add(Activation("tanh"))

    return model


def encoder(batch_size, df_dim, ch, rows, cols):

    model = Sequential()
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))
    model = Convolution2D(df_dim, 5, 5, subsample=(2, 2), border_mode="same",
                          name="e_h0_conv", dim_ordering="tf", init=normal)(X)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*2, 5, 5, subsample=(2, 2), border_mode="same",
                          name="e_h1_conv", dim_ordering="tf")(model)
    model = BN(mode=2, axis=3, name="e_bn1", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*4, 5, 5, subsample=(2, 2), name="e_h2_conv", border_mode="same",
                          dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="e_bn2", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*8, 5, 5, subsample=(2, 2), border_mode="same",
                          name="e_h3_conv", dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="e_bn3", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)
    model = Flatten()(model)

    mean = Dense(z_dim, name="e_h3_lin", init=normal)(model)
    logsigma = Dense(z_dim, name="e_h4_lin", activation="tanh", init=normal)(model)
    meansigma = Model([X], [mean, logsigma])
    return meansigma


def discriminator(batch_size, df_dim, ch, rows, cols):
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))
    model = Convolution2D(df_dim, 5, 5, subsample=(2, 2), border_mode="same",
                          batch_input_shape=(batch_size, rows[-1], cols[-1], ch),
                          name="d_h0_conv", dim_ordering="tf", init=normal)(X)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*2, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h1_conv", dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="d_bn1", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*4, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h2_conv", dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="d_bn2", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim*8, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h3_conv", dim_ordering="tf", init=normal)(model)

    dec = BN(mode=2, axis=3, name="d_bn3", gamma_init=mean_normal, epsilon=1e-5)(model)
    dec = LeakyReLU(.2)(dec)
    dec = Flatten()(dec)
    dec = Dense(1, name="d_h3_lin", init=normal)(dec)

    output = Model([X], [dec, model])

    return output


def get_model(sess, image_shape=(80, 160, 3), gf_dim=64, df_dim=64, batch_size=64,
              name="autoencoder", gpu=0):
    K.set_session(sess)
    checkpoint_dir = './outputs/results_' + name
    with tf.variable_scope(name), tf.device("/gpu:{}".format(gpu)):
      # sizes
      ch = image_shape[2]
      rows = [image_shape[0]//i for i in [16, 8, 4, 2, 1]]
      cols = [image_shape[1]//i for i in [16, 8, 4, 2, 1]]

      # nets
      G = generator(batch_size, gf_dim, ch, rows, cols)
      G.compile("sgd", "mse")
      g_vars = G.trainable_weights
      print ("G.shape: ", G.output_shape)

      E = encoder(batch_size, df_dim, ch, rows, cols)
      E.compile("sgd", "mse")
      e_vars = E.trainable_weights
      print ("E.shape: ", E.output_shape)

      D = discriminator(batch_size, df_dim, ch, rows, cols)
      D.compile("sgd", "mse")
      d_vars = D.trainable_weights
      print ("D.shape: ", D.output_shape)

      Z2 = Input(batch_shape=(batch_size, z_dim), name='more_noise')
      Z = G.input
      Img = D.input
      G_train = G(Z)
      E_mean, E_logsigma = E(Img)
      G_dec = G(E_mean + Z2 * E_logsigma)
      D_fake, F_fake = D(G_train)
      D_dec_fake, F_dec_fake = D(G_dec)
      D_legit, F_legit = D(Img)

      # costs
      recon_vs_gan = 1e-6
      like_loss = tf.reduce_mean(tf.square(F_legit - F_dec_fake)) / 2.
      kl_loss = tf.reduce_mean(-E_logsigma + .5 * (-1 + tf.exp(2. * E_logsigma) + tf.square(E_mean)))

      d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_legit, tf.ones_like(D_legit)))
      d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake)))
      d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_dec_fake, tf.zeros_like(D_dec_fake)))
      d_loss_fake = d_loss_fake1 + d_loss_fake2
      d_loss = d_loss_legit + d_loss_fake

      g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.ones_like(D_fake)))
      g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_dec_fake, tf.ones_like(D_dec_fake)))
      g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
      e_loss = kl_loss + like_loss

      # optimizers
      print ("Generator variables:")
      for v in g_vars:
        print (v.name)
      print ("Discriminator variables:")
      for v in d_vars:
        print (v.name)
      print ("Encoder variables:")
      for v in e_vars:
        print (v.name)

      e_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(e_loss, var_list=e_vars)
      d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
      g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
      tf.global_variables_initializer().run()

    # summaries
    sum_d_loss_legit = tf.summary.scalar("d_loss_legit", d_loss_legit)
    sum_d_loss_fake = tf.summary.scalar("d_loss_fake", d_loss_fake)
    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_g_loss = tf.summary.scalar("g_loss", g_loss)
    sum_e_loss = tf.summary.scalar("e_loss", e_loss)
    sum_e_mean = tf.summary.histogram("e_mean", E_mean)
    sum_e_sigma = tf.summary.histogram("e_sigma", tf.exp(E_logsigma))
    sum_Z = tf.summary.histogram("Z", Z)
    sum_gen = tf.summary.image("G", G_train)
    sum_dec = tf.summary.image("E", G_dec)

    # saver
    saver = tf.train.Saver()
    g_sum = tf.summary.merge([sum_Z, sum_gen, sum_d_loss_fake, sum_g_loss])
    e_sum = tf.summary.merge([sum_dec, sum_e_loss, sum_e_mean, sum_e_sigma])
    d_sum = tf.summary.merge([sum_d_loss_legit, sum_d_loss])
    writer = tf.summary.FileWriter("/tmp/logs/"+name, sess.graph)

    # functions
    def train_d(images, z, counter, sess=sess):
      z2 = np.random.normal(0., 1., z.shape)
      outputs = [d_loss, d_loss_fake, d_loss_legit, d_sum, d_optim]
      with tf.control_dependencies(outputs):
        updates = [tf.assign(p, new_p) for (p, new_p) in D.updates]
      outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
      dl, dlf, dll, sums = outs[:4]
      writer.add_summary(sums, counter)
      return dl, dlf, dll

    def train_g(images, z, counter, sess=sess):
      # generator
      z2 = np.random.normal(0., 1., z.shape)
      outputs = [g_loss, G_train, g_sum, g_optim]
      with tf.control_dependencies(outputs):
        updates = [tf.assign(p, new_p) for (p, new_p) in G.updates]
      outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
      gl, samples, sums = outs[:3]
      writer.add_summary(sums, counter)
      # encoder
      outputs = [e_loss, G_dec, e_sum, e_optim]
      with tf.control_dependencies(outputs):
        updates = [tf.assign(p, new_p) for (p, new_p) in E.updates]
      outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
      gl, samples, sums = outs[:3]
      writer.add_summary(sums, counter)

      return gl, samples, images

    def f_load():
      try:
        return load(sess, saver, checkpoint_dir, name)
      except:
        print("Loading weights via Keras")
        G.load_weights(checkpoint_dir+"/G_weights.keras")
        D.load_weights(checkpoint_dir+"/D_weights.keras")
        E.load_weights(checkpoint_dir+"/E_weights.keras")

    def f_save(step):
      save(sess, saver, checkpoint_dir, step, name)
      G.save_weights(checkpoint_dir+"/G_weights.keras", True)
      D.save_weights(checkpoint_dir+"/D_weights.keras", True)
      E.save_weights(checkpoint_dir+"/E_weights.keras", True)

    def sampler(z, x):
      code = E.predict(x, batch_size=batch_size)[0]
      out = G.predict(code, batch_size=batch_size)
      return out, x

    return train_g, train_d, sampler, f_save, f_load, [G, D, E]
