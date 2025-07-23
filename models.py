# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tensorflow.python.keras.layers.recurrent import activations
import u_net
import vgg
import collections
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
def build_model(model_type, batch_size):
  """Returns a Keras model specified by name."""
  if model_type == 'unet':
    return u_net.get_model(
        input_shape=(512, 512, 3),
        scales=4,
        bottleneck_depth=1024,
        bottleneck_layers=2)
  elif model_type == 'can':
    return vgg.build_can(
        input_shape=(512, 512, 3), conv_channels=64, out_channels=3)
  else:
    raise ValueError(model_type)

#Below are component of Uformer
class InputProj(tf.keras.Model):
  def __init__(self):
    super(InputProj, self).__init__()
    self.proj = tf.keras.layers.Conv2D(32, 3, 1, activation='LeakyReLU', padding = "same", input_shape=[512, 512, 3])
  def call(self, x):
    x = self.proj(x)
    return x
def window_partition(x, win_size):
  B, H, W, C = x.shape
  x = tf.reshape(x, (-1, H // win_size, win_size, W // win_size, win_size, C))
  windows = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, win_size, win_size, C))
  return windows
def window_reverse(windows, win_size, H, W):
    # B' ,Wh ,Ww ,C
    if windows.shape[0] != None:
      B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = tf.reshape(windows, (2, H // win_size, W // win_size, win_size, win_size, -1))
    x = tf.reshape(tf.transpose(x, (0, 1, 3, 2, 4, 5)),(2, H, W, -1))
    return x
class LinearProjection(tf.keras.Model):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super(LinearProjection, self).__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = tf.keras.layers.Dense(inner_dim, use_bias = bias, activation='LeakyReLU')
        self.to_kv = tf.keras.layers.Dense(inner_dim * 2, use_bias = bias, activation='LeakyReLU')
        self.dim = dim
        self.inner_dim = inner_dim
    def call(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x
        N_kv = attn_kv.shape[1] 
        q = tf.transpose(tf.reshape(self.to_q(x), (-1, N, 1, self.heads, C // self.heads)),(2, 0, 3, 1, 4))
        kv = tf.transpose(tf.reshape(self.to_kv(attn_kv), (-1, N_kv, 2, self.heads, C // self.heads)), (2, 0, 3, 1, 4))
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v
class WindowAttention(tf.keras.Model):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
      super(WindowAttention, self).__init__()
      self.dim = dim
      self.win_size = win_size  # Wh, Ww
      self.num_heads = num_heads
      head_dim = dim // num_heads
      self.scale = qk_scale or head_dim ** -0.5
      self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
      coords_h = tf.range(self.win_size[0]) # [0,...,Wh-1]
      coords_w = tf.range(self.win_size[1]) # [0,...,Ww-1]
      coords = tf.stack(tf.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
      coords_flatten = tf.keras.layers.Flatten()(coords)  # 2, Wh*Ww
      relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
      relative_coords = tf.transpose(relative_coords,(1, 2, 0))  # Wh*Ww, Wh*Ww, 2
      r1 = relative_coords[:, :, 0] + self.win_size[0] - 1
      r2 = relative_coords[:, :, 1] + self.win_size[1] - 1
      r1 = r1 * (2 * self.win_size[1] - 1)
      r4 = tf.stack([r1, r2], -1)
      relative_position_index =tf.math.reduce_sum(r4, -1)  # Wh*Ww, Wh*Ww

      self.relative_position_index = relative_position_index
      self.relative_position_bias_table =tf.Variable( tf.random.truncated_normal(((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads), stddev=.02) )
      self.proj = tf.keras.layers.Dense(dim, activation=None)
      self.softmax = tf.nn.softmax
    def call(self, x):
      B_, N, C = x.shape
      q, k, v = self.qkv(x)
      q = q * self.scale
      attn = (q @ tf.transpose(k, (0, 1, 3, 2)))
      relative_position_bias = tf.reshape( tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (tf.size(self.relative_position_index), 1) )),(self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1) ) # Wh*Ww,Wh*Ww,nH
      relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1) ) # nH, Wh*Ww, Wh*Ww
      ratio = attn.shape[-1]//relative_position_bias.shape[-1]
      relative_position_bias = tf.repeat(relative_position_bias, ratio, -1)
      attn = attn + tf.expand_dims(relative_position_bias, axis=0)
      attn = self.softmax(attn)
      x = tf.reshape(tf.transpose((attn @ v), (0, 2, 1, 3)), (-1, N, C))
      x = self.proj(x)
      return x
class LeFF(tf.keras.Model):
    def __init__(self, dim=32, hidden_dim=128, act_layer=tf.keras.activations.gelu, drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_dim, activation = None)
        self.g1 = act_layer
        self.dwconv = tf.keras.layers.Conv2D(hidden_dim,groups=hidden_dim,kernel_size=3,strides=1,padding="same")
        self.g2 = act_layer
        self.linear2 = tf.keras.layers.Dense(dim, activation = None)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def call(self, x):
        # bs x hw x c
        bs, hw, c = x.shape
        hh = int(math.sqrt(hw))

        x = self.linear1(x)
        x = self.g1(x)
        # spatial restore
        shape_0 = x.shape[0]
        shape_1 = x.shape[2]
        x  = tf.reshape(x, (-1, hh, hh, shape_1))
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        x = self.g2(x)
        x = tf.reshape(x, (-1, x.shape[1]*x.shape[2], x.shape[3]))
        x = self.linear2(x)
        return x
class LeWinTransformerBlock(tf.keras.Model):
  def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,act_layer=tf.keras.activations.gelu, norm_layer=tf.keras.layers.LayerNormalization, token_projection='linear',token_mlp='leff', modulator=False,cross_modulator=False):
    super(LeWinTransformerBlock, self).__init__()
    self.dp = drop_path
    self.dropout = tfa.layers.StochasticDepth()

    self.dim = dim
    self.input_resolution = input_resolution
    self.num_heads = num_heads
    self.win_size = win_size
    self.shift_size = shift_size
    self.mlp_ratio = mlp_ratio
    self.token_mlp = token_mlp
    self.norm1 = norm_layer()
    self.attn = WindowAttention(
                  dim, win_size=(self.win_size, self.win_size), num_heads=num_heads,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                  token_projection=token_projection)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
    self.norm2 = norm_layer()
  def call(self, x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    shortcut = x
    x = self.norm1(x)
    x = tf.reshape(x, (-1, H, W, C))
    x_windows = window_partition(x, self.win_size) #return windows with size             (N, win, win, C)
    x_windows = tf.reshape(x_windows, (-1, self.win_size * self.win_size, C))#reshape to (N, win* win, C)
    wmsa_in = x_windows
    attn_windows = self.attn(wmsa_in)
    shifted_x = window_reverse(attn_windows, self.win_size, H, W)
    x = shifted_x
    x = tf.reshape(x, (-1, H * W, C))
    if self.dp == 0:
      x = shortcut + x
      x = x + self.mlp(self.norm2(x))
    else:
      x = self.dropout([shortcut, x])
      x = self.dropout([x, self.mlp(self.norm2(x))])
    return x
class BasicUformerLayer(tf.keras.Model):
  def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size, mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, use_checkpoint=False, token_projection='linear',token_mlp='ffn', shift_flag=True, modulator=False,cross_modulator=False):
    super(BasicUformerLayer, self).__init__()
    self.dim = dim
    self.input_resolution = input_resolution
    self.depth = depth
    self.use_checkpoint = use_checkpoint
    self.blocks = [LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, win_size=win_size,
                                shift_size=0,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                modulator=modulator,cross_modulator=cross_modulator)
                for i in range(2)]
  def call(self, x):
    for blk in self.blocks:
      x = blk(x)
    return x
class Downsample(tf.keras.Model):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_channel,kernel_size=4,strides=2,padding="same")
        self.in_channel = in_channel
        self.out_channel = out_channel

    def call(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = tf.reshape(x, (-1, H, W, C))
        out = self.conv(x)
        out = tf.reshape(out, (-1, out.shape[1]*out.shape[2], out.shape[3]))
        return out
class Upsample(tf.keras.Model):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = tf.keras.layers.Conv2DTranspose(out_channel, kernel_size=2, strides=2)
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def call(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = tf.reshape(x, (-1, H, W, C))
        x1 = self.deconv(x)
        out = tf.reshape(x1, (-1, x1.shape[1]*x1.shape[2], x1.shape[3])) # B H*W C
        return out
class OutputProj(tf.keras.Model):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = tf.keras.layers.Conv2D(out_channel, kernel_size=3, strides=stride, padding="same")

        self.in_channel = in_channel
        self.out_channel = out_channel

    def call(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        y = tf.reshape(x, (-1, H, W, C))
        y = self.proj(y)
        return y
class Uformer(tf.keras.Model):
  def __init__(self, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],  num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2], embed_dim=32, win_size=8, img_size=512, mlp_ratio=4., qk_scale=None, drop_path_rate=0.1, attn_drop_rate = 0, shift_flag = True, norm_layer = tf.keras.layers.LayerNormalization, drop_rate = 0, qkv_bias=True, use_checkpoint=False,token_projection='linear', token_mlp='leff', dowsample=Downsample, upsample=Upsample):
    super(Uformer, self).__init__()
    self.num_enc_layers = len(depths)//2
    self.mlp_ratio = mlp_ratio
    enc_dpr = [x.item() for x in np.linspace(0, 0.1, sum(depths[:self.num_enc_layers]))] 
    conv_dpr = [drop_path_rate]*depths[4]
    dec_dpr = enc_dpr[::-1]
    self.input_proj = InputProj()
    self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=3, kernel_size=3, stride=1)
    self.encoderlayer_0 = BasicUformerLayer(dim=32,
                            output_dim=32,
                            input_resolution=(512,
                                                512),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=None,
                            drop=0, attn_drop=0,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=tf.keras.layers.LayerNormalization,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp)
    self.downsample_0 = dowsample(embed_dim, embed_dim*2)
    self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.downsample_1 = dowsample(embed_dim*2, embed_dim*4)
    self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.downsample_2 = dowsample(embed_dim*4, embed_dim*8)
    self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.downsample_3 = dowsample(embed_dim*8, embed_dim*16)
    self.conv = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
    self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
    self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
    self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)
    self.upsample_3 = upsample(embed_dim*4, embed_dim)
    self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

  def call(self, x):
    y = self.input_proj(x)
    y = tf.transpose(y, (0, 3, 1, 2))
    y = tf.reshape(y, (2, 32, 262144))
    y = tf.transpose(y, (0, 2, 1))
    conv0 = self.encoderlayer_0(y)
    pool0 = self.downsample_0(conv0)
    conv1 = self.encoderlayer_1(pool0)
    pool1 = self.downsample_1(conv1)
    conv2 = self.encoderlayer_2(pool1)
    pool2 = self.downsample_2(conv2)
    conv3 = self.encoderlayer_3(pool2)
    pool3 = self.downsample_3(conv3)
    conv4 = self.conv(pool3)
    up0 = self.upsample_0(conv4)
    deconv0 = tf.concat([up0,conv3],-1)
    deconv0 = self.decoderlayer_0(deconv0)
        
    up1 = self.upsample_1(deconv0)
    deconv1 = tf.concat([up1,conv2],-1)
    deconv1 = self.decoderlayer_1(deconv1)

    up2 = self.upsample_2(deconv1)
    deconv2 = tf.concat([up2,conv1],-1)
    deconv2 = self.decoderlayer_2(deconv2)

    up3 = self.upsample_3(deconv2)
    deconv3 = tf.concat([up3,conv0],-1)
    deconv3 = self.decoderlayer_3(deconv3)

    # Output Projection
    y = self.output_proj(deconv3)
    return y + x
