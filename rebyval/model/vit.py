import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

from dnn import DNN


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class VIT(Model):
    
    def __init__(self, patch_size, num_patches, 
                 dims_emb, num_heads, trans_layers=8, mlp_head_units=[2048,1024], regularizer=None, name='vit', **kwargs):
        super(VIT, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.dims_emb = dims_emb
        self.trans_units = [self.dims_emb*2, self.dims_emb]
        self.trans_layers = trans_layers
        self.mlp_head_units = [2048,1024]
        
        self.projection = layers.Dense(units=dims_emb)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=dims_emb)
        
        
    def _build_trans_block(self):
        trans_block = {}
        trans_block["LBN_1"] = layers.LayerNormalization(epsilon=1e-6)
        trans_block["MHA"] = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dims_emb, dropout=0.1)
        trans_block["skip_1"] = layers.Add()
        trans_block["LBN_2"] = layers.LayerNormalization(epsilon=1e-6)
        trans_block["mlp"] = DNN(units=self.trans_units, activations=["gelu"]*len(self.trans_units))
        trans_block["skip_2"] = layers.Add()
        return trans_block
    
    def _build_trans(self):
        trans = {}
        for i in range(self.trans_layers):
            trans["trans_layer_{]".format(i)] = self._build_trans_block()
        trans["LBN"] = layers.LayerNormalization(epsilon=1e-6)
        trans["flatten"] = layers.Flatten()
        trans["DP"] = layers.Dropout(0.5)
        trans["mlp"] = DNN(units=self.mlp_head_units, activations=["gelu"]*len(self.mlp_head_units))
        
        
        
            