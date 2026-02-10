# pip install flax==0.8.* jax jaxlib optax
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

# ----------------------------
# Utilities
# ----------------------------


class ActorCritic(nn.Module):
    in_channels: int
    out_channels: int
    out_directions: int

    grid_height: int
    grid_width: int
    layer_width: int

    add_coords: bool = False
    h: int = 8

    def setup(self):
        ## V-linear layers
        self.vr_linear1 = nn.Dense(features=self.layer_width)
        self.vr_linear2 = nn.Dense(features=1)

        ## Dip-policy
        self.dip_conv1 = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))    # reduse spatial shape by dividing by 2. grid_height+1 --> grid_height/2+1
        self.dip_conv2 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))    # reduse spatial shape by dividing by 2. grid_height/2+1 --> grid_height/4+1
        self.dip_deconv3 = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2))
        self.dip_deconv2 = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2))
        self.dip_conv_p = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.group_norm1 = nn.GroupNorm(num_groups=self.h)
        self.group_norm2 = nn.GroupNorm(num_groups=self.h)
    # *,

        ## Sub-policy
        self.sub_conv1 = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2))    # reduse spatial shape by dividing by 2. grid_height+1 --> grid_height/2+1
        self.sub_conv2 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))    # reduse spatial shape by dividing by 2. grid_height/2+1 --> grid_height/4+1
        self.sub_deconv3 = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2))
        self.sub_deconv2 = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2))
        self.sub_conv_p = nn.Conv(features=self.out_directions, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.sub_group_norm1 = nn.GroupNorm(num_groups=self.h)
        self.sub_group_norm2 = nn.GroupNorm(num_groups=self.h)


        ## V-map deconvolutions
        self.v_deconv2 = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2))
        self.v_group_norm2 = nn.GroupNorm(num_groups=self.h)
        self.v_conv_p = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='same')


    def __call__(self, x):  # (N, D+2, N_y, N_x)

        x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')   # (N, D+2, N_y, N_x) --> (N, D+2, N_y+1, N_x+1)
        x_pad_T = jnp.transpose(x_pad, axes=(0, 2, 3, 1))   # (N, D, N_y, N_x) --> (N, N_y, N_x, D)

        N, Ny, Nx, C = x_pad_T.shape


        dip_down2 = nn.gelu(self.dip_conv1(x_pad_T))  # (N, N_y, N_x, 1) --> (N, N_y/2, N_x/2, 16)
        dip_down3 = nn.gelu(self.dip_conv2(dip_down2))  # (N, N_y/2, N_x/2, 16) --> (N, N_y/4, N_x/4, 32)
        dip_down3 = self.group_norm1(dip_down3)
        dip_upp2 = nn.gelu(self.dip_deconv3(dip_down3))
        dip_upp1 = nn.gelu(self.dip_deconv2(dip_upp2+dip_down2))
        dip_upp1 = self.group_norm2(dip_upp1)
        sv_logit = self.dip_conv_p(dip_upp1)    # (N, N_y+1, N_x+1, 1)
        sv_logit = sv_logit[:, :-1, :-1, :]       # (N, N_y+1, N_x+1, 1) --> (N, N_y, N_x, 1)
        sv_logit = sv_logit[:,:,:,0]              # (N, N_y, N_x, 1) --> (N, N_y, N_x)


        sub_down2 = nn.gelu(self.sub_conv1(x_pad_T))  # (N, N_y, N_x, 1) --> (N, N_y/2, N_x/2, 16)
        sub_down3 = nn.gelu(self.sub_conv2(sub_down2))  # (N, N_y/2, N_x/2, 16) --> (N, N_y/4, N_x/4, 32)
        sub_down3 = self.sub_group_norm1(sub_down3)
        sub_upp2 = nn.gelu(self.sub_deconv3(sub_down3))
        sub_upp1 = nn.gelu(self.sub_deconv2(sub_upp2+sub_down2))
        sub_upp1 = self.sub_group_norm2(sub_upp1)
        sub_logit = self.sub_conv_p(sub_upp1)    # (N, N_y+1, N_x+1, C)
        sub_logit = sub_logit[:, :-1, :-1, :]       # (N, N_y+1, N_x+1, C) --> (N, N_y, N_x, C)
        sub_logit = sub_logit.transpose(0, 3, 1, 2)     # (N, N_y, N_x, C) --> (N, C, N_y, N_x)
        sub_logit = jnp.expand_dims(sub_logit, axis=1)  # (N, C, N_y, N_x) --> (N, 1, C, N_y, N_x)
        sub_logit = jnp.repeat(sub_logit, repeats=self.out_directions , axis=1)     # (N, 1, C, N_y, N_x) --> (N, D, C, N_y, N_xv_map

        
        v_map = nn.gelu(self.v_deconv2(sub_upp2+sub_down2))
        v_map = self.v_group_norm2(sub_upp1)
        v_map = self.v_conv_p(sub_upp1)    # (N, N_y+1, N_x+1, 1)
        v_map = v_map[:, :-1, :-1, :]       # (N, N_y+1, N_x+1, 1) --> (N, N_y, N_x, 1)
        v_map = v_map[:,:,:,0]              # (N, N_y, N_x, 1) --> (N, N_y, N_x)


        low_level_embedding = sub_down3
        # print(f"low_level_embedding: {low_level_embedding.shape}")
        flat_embedding = low_level_embedding.reshape((low_level_embedding.shape[0], -1))  # Flatten along spatial dimensions
        # print(f"flat_embedding: {flat_embedding.shape}")
        v1 = self.vr_linear1(flat_embedding)
        v = self.vr_linear2(nn.gelu(v1))

        # print(f"sub_logit: {sub_logit.shape}")
        # print(f"sv_logit: {sv_logit.shape}")
        # print(f"v_map: {v_map.shape}")
        # print(f"v: {v.shape}")


        return sub_logit, sv_logit, v, v_map
    


import pathlib
import flax
from flax.serialization import msgpack_restore
from flax.core import unfreeze
from flax.traverse_util import flatten_dict
def load_model(model, model_path, dummy_input, key=jax.random.PRNGKey(0)):
    print(f"model_path: {model_path}")
    model_path = pathlib.Path(model_path)
    print(f"model_path: {model_path}")
    with (model_path).open("rb") as f:
        param_bytes = f.read()

        def param_names_from_bytes(param_bytes, only_trainable=True):
            state = msgpack_restore(param_bytes)             # nested dict/FrozenDict with arrays
            tree = state.get("params", state) if only_trainable and isinstance(state, dict) else state
            flat = flatten_dict(unfreeze(tree))              # keys are tuples like ("Dense_0","kernel")

            names = []
            for k_tuple, v in flat.items():
                names.append("/".join(k_tuple))
            return names
        
        names = param_names_from_bytes(param_bytes)
        print(f"Params-names: {names}")

    param_template = model.init(key, dummy_input)
    params = flax.serialization.from_bytes(param_template, param_bytes)

    return params