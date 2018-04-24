from __future__ import print_function

import torch


'''
def normalize(x, axis=-1):

  """Normalizing to unit length along the specified dimension.

  Args:

    x: pytorch Variable

  Returns:

    x: pytorch Variable, same shape as input      

  """
 #torch.norm,返回输入张量input 的p 范数。  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12) 
  return x'''
def normalize(x, axis=-1):
  x=1.*x/(tf.sqrt(tf.reduce_sum(tf.pow(x,2),keep_dims=True))+1e-12)
  return x





'''def euclidean_dist(x, y):

  """

  Args:

    x: pytorch Variable, with shape [m, d]

    y: pytorch Variable, with shape [n, d]

  Returns:

    dist: pytorch Variable, with shape [m, n]

  """

  m, n = x.size(0), y.size(0)   #tf.shape()
  #x.expand(m,n)将矩阵扩充为m，n;x.t()矩阵的转置

  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)   #xx=tf.pow(x,2)
  #tf.reduce_sum()

  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()#.t()矩阵的转置

  dist = xx + yy
# \(out = (beta * M) + (alpha * mat1 @ mat2)\)
  dist.addmm_(1, -2, x, y.t())  #执行矩阵乘法
#clamp:小于min的数置为min,大于max的置为max
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

  return dist'''
def euclidean_dist(x,y):
  m=tf.shape(x)[0]
  n=tf.shape(y)[0]
  xx=tf.pow(x,2)
  xx=tf.reduce_sum(xx,1,keep_dims=True)
  xx=tf.tile(xx,(1,n))
  yy=tf.pow(y,2)
  yy=tf.reduce_sum(yy,1,keep_dims=True)
  yy=tf.tile(yy,(1,n))
  yy=tf.transpose(yy)
  dist=xx+yy
  dist=tf.add(dist,tf.matmul(-2*x,tf.transpose(y)))
  dist=tf.clip_by_value(dist,le-12,le+12)
  return dist





'''def batch_euclidean_dist(x, y):

  """

  Args:

    x: pytorch Variable, with shape [N, m, d]

    y: pytorch Variable, with shape [N, n, d]

  Returns:

    dist: pytorch Variable, with shape [N, m, n]

  """

  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)
  N, m, d = x.size()
  N, n, d = y.size()
  # shape [N, m, n]

  xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)

  yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)

  dist = xx + yy
#baddbmm_()批矩阵的乘法
  dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))

  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

  return dist
'''
def batch_euclidean_dist(x, y):
  assert tf.size(tf.shape(x)) == tf.constant(3)
  assert tf.size(tf.shape(y)) == 3
  assert tf.shape(x)[0] == tf.shape(y)[0]
  assert tf.shape(x)[2] == tf.shape(y)[2]
  N, m, d = tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]
  xx=tf.reduce_sum(tf.pow(x,2),-1,keep_dims=True)
  xx=tf.tile(xx,[1,1,n])
  yy=tf.reduce_sum(tf.pow(y,2),-1,keep_dims=True).transpose(0,2,1)
  yy=tf.tile(yy,[1,1,m])
  dist=xx+yy
  dist=tf.add(dist,tf.matmul(-2*x,tf.transpose(0,2,1)))
  dist=tf.clip_by_value(dist,le-12,le+12)
  return dist
  




'''def shortest_dist(dist_mat):

  """Parallel version.

  Args:

    dist_mat: pytorch Variable, available shape:

      1) [m, n]

      2) [m, n, N], N is batch size

      3) [m, n, *], * can be arbitrary additional dimensions

  Returns:

    dist: three cases corresponding to `dist_mat`:

      1) scalar

      2) pytorch Variable, with shape [N]

      3) pytorch Variable, with shape [*]

  """

  m, n = dist_mat.size()[:2]

  # Just offering some reference for accessing intermediate distance.

  dist = [[0 for _ in range(n)] for _ in range(m)]

  for i in range(m):

    for j in range(n):

      if (i == 0) and (j == 0):

        dist[i][j] = dist_mat[i, j]

      elif (i == 0) and (j > 0):

        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]

      elif (i > 0) and (j == 0):

        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]

      else:

        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]

  dist = dist[-1][-1]

  return dist'''
def shortest_dist(dist_mat):
   m,n=tf.shape(dist_mat)[0],tf.shape(dist_mat)[1]
   sess=tf.Session()
   init_op=tf.initialize_all_variables()
   m,n=sess.run(m),sess.run(n)
   dist_mat=sess.run(dist_mat)
   dist = [[0 for _ in range(n)] for _ in range(m)]
   for i in range(m):
      for j in range(n):
         if (i == 0) and (j == 0):
            dist[i][j] = dist_mat[i, j]
         elif (i == 0) and (j > 0):
           dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
         elif (i > 0) and (j == 0):
           dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
         else:
           dist[i][j] = tf.minimum(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
   dist = dist[-1][-1]
   return dist






'''def local_dist(x, y):

  """

  Args:

    x: pytorch Variable, with shape [M, m, d]

    y: pytorch Variable, with shape [N, n, d]

  Returns:

    dist: pytorch Variable, with shape [M, N]

  """

  M, m, d = x.size()

  N, n, d = y.size()
#返回一个内存连续的有相同数据的 tensor, 如果原 tensor 内存连续则返回原 tensor.
#返回一个有相同数据但大小不同的新的 tensor.
#返回的 tensor 与原 tensor 共享相同的数据，一定有相同数目的元素，但大小不同. 一个 tensor 必须是连续的 ( contiguous() ) 才能被查看.
  x = x.contiguous().view(M * m, d)

  y = y.contiguous().view(N * n, d)

  # shape [M * m, N * n]

  dist_mat = euclidean_dist(x, y)

  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)

  # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]

  dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)  #将tensor的维度换位 

  # shape [M, N]

  dist_mat = shortest_dist(dist_mat)

  return dist_mat'''
def local_dist(x, y):
  M,m,d= tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]
  N,n,d= tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]
  x=tf.reshape(x,[M*m,d])
  y=tf.reshape(y,[N*n,d])
  dist_mat=euclidean_dist(x,y)
  dist_mat=(tf.exp(dist_mat)-1.)/(tf.exp(dist_mat)+1)
  dist_mat=tf.transpose(dist_mat,perm=[1,3,0,2])
  dist_mat=shortest_dist(dist_mat)
  return dist_mat





'''def batch_local_dist(x, y):

  """

  Args:

    x: pytorch Variable, with shape [N, m, d]

    y: pytorch Variable, with shape [N, n, d]

  Returns:

    dist: pytorch Variable, with shape [N]

  """

  assert len(x.size()) == 3

  assert len(y.size()) == 3

  assert x.size(0) == y.size(0)

  assert x.size(-1) == y.size(-1)



  # shape [N, m, n]

  dist_mat = batch_euclidean_dist(x, y)

  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)

  # shape [N]

  dist = shortest_dist(dist_mat.permute(1, 2, 0))

  return dist'''
def batch_local_list(x,y):
  dist_mat=batch_euclidean_dist(x,y)
  dist_mat=(tf.exp(dist_mat)-1.)/(tf.exp(dist_mat)+1.)
  dist=shortest_dist(tf.transpose(dist_mat,perm=[1,2,0]))
  return dist






'''def hard_example_mining(dist_mat, labels, return_inds=False):

  """For each anchor, find the hardest positive and negative sample.

  Args:

    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]

    labels: pytorch LongTensor, with shape [N]

    return_inds: whether to return the indices. Save time if `False`(?)

  Returns:

    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]

    dist_an: pytorch Variable, distance(anchor, negative); shape [N]

    p_inds: pytorch LongTensor, with shape [N]; 

      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1

    n_inds: pytorch LongTensor, with shape [N];

      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1

  NOTE: Only consider the case in which all labels have same num of samples, 

    thus we can cope with all anchors in parallel.

  """



  assert len(dist_mat.size()) == 2  #是否是二维的

  assert dist_mat.size(0) == dist_mat.size(1)

  N = dist_mat.size(0)



  # shape [N, N]

  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())

  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())



  # `dist_ap` means distance(anchor, positive)

  # both `dist_ap` and `relative_p_inds` with shape [N, 1]

  dist_ap, relative_p_inds = torch.max(

    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

  # `dist_an` means distance(anchor, negative)

  # both `dist_an` and `relative_n_inds` with shape [N, 1]

  dist_an, relative_n_inds = torch.min(

    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

  # shape [N]

  dist_ap = dist_ap.squeeze(1) #去掉size为1的维度

  dist_an = dist_an.squeeze(1)



  if return_inds:

    # shape [N, N]

    ind = (labels.new().resize_as_(labels)

           .copy_(torch.arange(0, N).long())

           .unsqueeze( 0).expand(N, N))

    # shape [N, 1]

    p_inds = torch.gather(

      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)

    n_inds = torch.gather(

      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)

    # shape [N]

    p_inds = p_inds.squeeze(1)

    n_inds = n_inds.squeeze(1)

    return dist_ap, dist_an, p_inds, n_inds



  return dist_ap, dist_an'''

def hard_example_mining(dist_mat, labels, return_inds=False):
    N=tf.shape(dist_mat)[0]
    is_pos=tf.equal(tf.tile(lables,[N,N]),tf.transpose(tf.tile(lables,[N,N])))
    is_neg=tf.not_equal(tf.tile(lables,[N,N]),tf.transpose(tf.tile(lables,[N,N])))
    dist_ap, relative_p_inds = tf.maximum(tf.reshape(dist_mat[is_pos]),1, keep_dims=True)
    dist_an, relative_n_inds = tf.minimum(tf.reshape(dist_mat[is_neg]),1, keep_dims=True)
    dist_ap=tf.squeeze(dist_ap)
    dist_an=tf.squeeze(dist_an)
    return dist_ap,dist_an





def global_loss(tri_loss, global_feat, labels, normalize_feature=True):

  """

  Args:

    tri_loss: a `TripletLoss` object

    global_feat: pytorch Variable, shape [N, C]

    labels: pytorch LongTensor, with shape [N]

    normalize_feature: whether to normalize feature to unit length along the 

      Channel dimension

  Returns:

    loss: pytorch Variable, with shape [1]

    p_inds: pytorch LongTensor, with shape [N]; 

      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1

    n_inds: pytorch LongTensor, with shape [N];

      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1

    =============

    For Debugging

    =============

    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]

    dist_an: pytorch Variable, distance(anchor, negative); shape [N]

    ===================

    For Mutual Learning

    ===================

    dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]

  """

  if normalize_feature:

    global_feat = normalize(global_feat, axis=-1)

  # shape [N, N]

  dist_mat = euclidean_dist(global_feat, global_feat)

  dist_ap, dist_an, p_inds, n_inds = hard_example_mining(

    dist_mat, labels, return_inds=True)

  loss = tri_loss(dist_ap, dist_an)

  return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat



def local_loss(

    tri_loss,

    local_feat,

    p_inds=None,

    n_inds=None,

    labels=None,

    normalize_feature=True):

  """

  Args:

    tri_loss: a `TripletLoss` object

    local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)

    p_inds: pytorch LongTensor, with shape [N]; 

      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1

    n_inds: pytorch LongTensor, with shape [N];

      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1

    labels: pytorch LongTensor, with shape [N]

    normalize_feature: whether to normalize feature to unit length along the 

      Channel dimension

  

  If hard samples are specified by `p_inds` and `n_inds`, then `labels` is not 

  used. Otherwise, local distance finds its own hard samples independent of 

  global distance.

  

  Returns:

    loss: pytorch Variable,with shape [1]

    =============

    For Debugging

    =============

    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]

    dist_an: pytorch Variable, distance(anchor, negative); shape [N]

    ===================

    For Mutual Learning

    ===================

    dist_mat: pytorch Variable, pairwise local distance; shape [N, N]

  """

  if normalize_feature:

    local_feat = normalize(local_feat, axis=-1)

  if p_inds is None or n_inds is None:

    dist_mat = local_dist(local_feat, local_feat)

    dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds=False)

    loss = tri_loss(dist_ap, dist_an)

    return loss, dist_ap, dist_an, dist_mat

  else:

    dist_ap = batch_local_dist(local_feat, local_feat[p_inds])

    dist_an = batch_local_dist(local_feat, local_feat[n_inds])

    loss = tri_loss(dist_ap, dist_an)

    return loss, dist_ap, dist_an
