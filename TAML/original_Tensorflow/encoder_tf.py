import tensorflow as tf
from layers_tf import *
# inference network for generating the three balancing variables

def kl_diagnormal_stdnormal_debug(q):
  # print("q dtype: ", q.dtype)
  qshape = q.mean().shape
  # p = normal(tf.zeros(qshape, dtype=tf.dtypes.float64), tf.ones(qshape, dtype=tf.dtypes.float64))
  p = normal(tf.zeros(qshape, dtype='float64'), tf.ones(qshape, dtype='float64'))
  # print("p dtype: ", p.dtype)
  return tf.distributions.kl_divergence(q, p)

class InferenceNetwork:
  def __init__(self, args):
    if args.id_dataset[0] == 'cifar' or len(args.id_dataset) > 1:
      self.xdim, self.input_channel, self.n_channel = 32, 3, 32
    elif args.id_dataset[0] == 'mimgnet':
      self.xdim, self.input_channel, self.n_channel = 84, 3, 32
    else:
      raise ValueError("Invalid in-dist. dataset: %s" % args.id_dataset)

    self.numclass = args.way
    self.max_shot = args.max_shot

    # turn on/off the balancing variables
    self.z_on = args.z_on
    self.gamma_on = args.gamma_on
    self.omega_on = args.omega_on

    self.s_on = True if len(args.id_dataset) > 1 else False
  # Compute element-wise sample mean, var., and set cardinality
  # then, return the concatenation of them.
  def _statistics_pooling(self, x, N):
    print("|||encoder: _statistics_pooling")
    mean, var = tf.nn.moments(x, 0)
    N = tf.tile(tf.reshape(N, [-1]), mean.shape.as_list())
    return tf.stack([mean, var, N], 1)

  # compute the posterior of balancing variables
  def get_posterior(self, inputs, name='encoder', reuse=None):
    print("|||encoder: get_posterior()")
    x, y = inputs
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)

    # encoder 1
    print("<<<<encoder 1>>>")
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    print("x reshaped before convolution: ", x.shape)
    x = conv(x, 10, name=name+'/conv1', reuse=reuse)
    print("x after convolution 1: ", x.shape)
    x = relu(x)
    x = pool(x)
    print("x after max pooling 1: ", x.shape)
    x = conv(x, 10, name=name+'/conv2', reuse=reuse)
    print("x after convolution 2: ", x.shape)
    x = relu(x)
    x = pool(x)
    print("x after max pooling 2: ", x.shape)
    x = flatten(x)
    print("x after flatten: ", x.shape)
    x = dense(x, 64, name=name+'/dense1', reuse=reuse)
    print("x after dense layer: ", x.shape)

    # statistics pooling 1
    print("<<<statistics pooling 1>>>")
    ysum = tf.reduce_sum(y, 1)
    print("y_sum: (sum reduced along dimension 1)", ysum)
    y_ = tf.argmax(y, 1)
    print("y_ (argmax of y on dimension 1) : ", y_)
    s = []; N_c_list = []
    print("iterate over number of classes", self.numclass)
    for c in range(self.numclass):
      idx_c = tf.logical_and(tf.equal(y_,c), tf.greater(ysum, 0))
      x_c = tf.boolean_mask(x, idx_c) # x's corresponding to class c
      y_c = tf.boolean_mask(y, idx_c)
      N_c = (tf.reduce_sum(y_c)-1.)/(self.max_shot-1.) # normalized set size
      N_c_list.append(N_c)
      print("\tN_c: ", N_c.shape)
      print("\tx_c shape: ", x_c.shape)
      s_c = self._statistics_pooling(x_c, N=N_c)
      print("\ts_c (from statistics pooling x_c: ", s_c)
      s.append(s_c)
    s = tf.stack(s, 0)
    print("final s (stacked from the 's_c's): ", s.shape)
    s = dense(s, 4, name=name+'/interact1', reuse=reuse)
    print("s after dense layer: ", s.shape)
    s = relu(s)
    s = tf.reshape(s, [self.numclass, -1])
    print("s reshape: ", s.shape)

    # encoder 2
    print("<<<encoder 2>>>")
    v = dense(s, 128, name=name+'/dense2', reuse=reuse)
    print("v, result of s after dense layer, ", v.shape)
    v = relu(v)
    v = dense(v, 32, name=name+'/dense3', reuse=reuse)
    print("v, another dense layer, ", v.shape)

    # statistics pooling 2
    print("<<<statistics pooling 2>>>")
    v = self._statistics_pooling(v, N=tf.reduce_mean(N_c_list))
    print("v after statistics pooling: ", v.shape)
    v = tf.expand_dims(v, 0)
    print("v after expanded dims, ", v.shape)
    v = dense(v, 4, name=name+'/interact2', reuse=reuse)
    print("v after dense layer: ", v.shape)
    v = relu(v)
    v = tf.reshape(v, [1, -1])
    print("v reshaped: ", v.shape)

    # generate omega (from statistics pooling 1)
    print("<<<omega>>>")
    print("using omega? ", self.s_on)
    s1 = dense(s, 64, name=name+'/dense_omega', reuse=reuse)
    print("\ts1, result of s after dense layer: ", s1.shape)
    s1 = relu(s1)
    odim = 1
    s_o = s if self.s_on else s1
    mu_omega = dense(s_o, odim, name=name+'/mu_omega', reuse=reuse)
    print("mu omega: from dense layer", mu_omega.shape)
    sigma_omega = dense(s_o, odim, name=name+'/sigma_omega', reuse=reuse)
    print("sigma omega: from dense layer", sigma_omega.shape)
    mu_omega, sigma_omega = tf.squeeze(mu_omega), tf.squeeze(sigma_omega)
    print("mu omega, squeezed: ", mu_omega.shape)
    print("sigma omega, squeezed: ", sigma_omega.shape)
    q_omega = normal(mu_omega, softplus(sigma_omega))
    print("-> acquired omega posterior distribution")


    # generate gamma (from statistics pooling 2)
    print("<<<gamma>>>")
    v1 = dense(v, 64, name=name+'/dense_gamma', reuse=reuse)
    print("\tv1, result of v after dense layer: ", v1.shape)
    v1 = relu(v1)
    gdim = 5
    mu_gamma = dense(v1, gdim, name=name+'/mu_gamma', reuse=reuse)
    print("mu gamma: from dense layer", mu_gamma.shape)
    sigma_gamma = dense(v1, gdim, name=name+'/sigma_gamma', reuse=reuse)
    print("sigma gamma: from dense layer", sigma_gamma.shape)
    mu_gamma, sigma_gamma = tf.squeeze(mu_gamma), tf.squeeze(sigma_gamma)
    print("mu gamma, squeezed: ", mu_gamma.shape)
    print("sigma gamma, squeezed: ", sigma_gamma.shape)
    q_gamma = normal(mu_gamma, softplus(sigma_gamma))
    print("-> acquired gamma posterior distribution")

    # generate z (from statistics pooling 2)
    print("<<<zeta>>>")
    v2 = dense(v, 64, name=name+'/dense_z', reuse=reuse)
    print("\tv2, result of v after dense layer: ", v2.shape)
    v2 = relu(v2)
    zdim = 2*self.n_channel*4
    mu_z = dense(v2, zdim, name=name+'/mu_z', reuse=reuse)
    print("mu zeta: from dense layer", mu_z.shape)
    sigma_z = dense(v2, zdim, name=name+'/sigma_z', reuse=reuse)
    print("sigma zeta: from dense layer", sigma_z.shape)
    mu_z, sigma_z = tf.squeeze(mu_z), tf.squeeze(sigma_z)
    print("mu zeta, squeezed: ", mu_z.shape)
    print("sigma zeta, squeezed: ", sigma_z.shape)
    q_z = normal(mu_z, softplus(sigma_z))
    print("-> acquired zeta posterior distribution")
    return q_omega, q_gamma, q_z

  def forward(self, inputs, sample, reuse=None):
    # compute posterior
    q_omega, q_gamma, q_z = self.get_posterior(inputs, reuse=reuse)

    # compute kl
    kl_omega = tf.reduce_sum(kl_diagnormal_stdnormal_debug(q_omega))
    print("kl omega shape: ", kl_omega.shape)
    kl_gamma = tf.reduce_sum(kl_diagnormal_stdnormal_debug(q_gamma))
    print("kl gamma shape: ", kl_gamma.shape)
    kl_z = tf.reduce_sum(kl_diagnormal_stdnormal_debug(q_z))
    print("kl zeta shape: ", kl_z.shape)

    # sample variables from the posterior
    omega, gamma, z = None, None, None
    kl = 0.
    if self.omega_on:
      kl = kl + kl_omega
      omega = q_omega.sample() if sample else q_omega.mean()
      print("omega shape: ", omega.shape)

    if self.gamma_on:
      kl = kl + kl_gamma
      g_ = q_gamma.sample() if sample else q_gamma.mean()
      print("g_ shape: ", g_.shape)
      g_ = tf.split(g_, [1,1,1,1,1], 0)
      print("g_ shape: ", g_[0].shape, "g_ length", len(g_))
      gamma = {}
      for l in [1,2,3,4]:
        gamma['conv%d_w'%l] = gamma['conv%d_b'%l] = g_[l-1]
      gamma['dense_w'] = gamma['dense_b'] = g_[4]

    if self.z_on:
      kl = kl + kl_z
      z_ = q_z.sample() if sample else q_z.mean()
      print("z_ shape: ", z_.shape)
      zw_ = tf.split(z_[:self.n_channel*4], [self.n_channel]*4, 0)
      print("how many zw_s: ", len(zw_), "range: ", self.n_channel*4, "num or size: ", self.n_channel)
      print("zw_ element shape: ", zw_[0].shape, zw_[1].shape, zw_[2].shape)
      zb_ = tf.split(z_[self.n_channel*4:], [self.n_channel]*4, 0)
      print("how many zb_s: ", len(zb_))
      print("zb_ element shape: ", zb_[0].shape)
      z = {}
      for l in [1,2,3,4]:
        z['conv%d_w'%l] = zw_[l-1]
        z['conv%d_b'%l] = zb_[l-1]

    return omega, gamma, z, kl
