from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models.transformer import transformer_base

import tensorflow as tf


@registry.register_model
class UniversalTransformerMohd(t2t_model.T2TModel):
    def body(self, features):
        inputs = features["inputs"]
        original_shape = common_layers.shape_list(inputs) 
        squeeze_shape = [x for x in common_layers.shape_list(inputs) if x != 1]
        inputs = tf.reshape(inputs,squeeze_shape)
        outputs = self.adaptive_computation(inputs,self.encode)
        outputs = tf.reshape(outputs,original_shape)
        return outputs

    def encode(self, encoder_inputs,timestep):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            #positional encoding
            x = common_attention.add_timing_signal_1d(encoder_inputs)
            #timestep encoding
            timestep = tf.fill(common_layers.shape_list(x)[:-1],timestep)
            x = common_attention.add_timing_signal_1d_given_position(x,timestep)
            #attention
            y = common_attention.multihead_attention(query_antecedent=x,
                    memory_antecedent=None,
                    bias=None,
                    total_key_depth=self.hparams.hidden_size,
                    total_value_depth=self.hparams.hidden_size,
                    output_depth=self.hparams.hidden_size,
                    num_heads=self.hparams.num_heads,
                    dropout_rate=self.hparams.attention_dropout)
            #residual connection and dropout
            #hparams.layer_postprocess_sequence = "da" (add,dropout)
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            #transition function as fc
            x = tf.layers.dense(x,self.hparams.hidden_size, name="transition")
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            return x

    def adaptive_computation(self, inputs, funct):
        with tf.variable_scope("ACT_" + funct.__name__, reuse=tf.AUTO_REUSE):
            halting_probability = tf.zeros(common_layers.shape_list(inputs)[:2] + [1],
                    name="halting_probability")
            timestep = tf.constant(0, dtype=tf.int32)
            def act_step(state, halting_probability, timestep):
                funct_output = funct(inputs,timestep)
                units_to_update = tf.less(halting_probability,self.hparams.act_threshold)
                units_to_keep = tf.logical_not(units_to_update)
                units_to_update = tf.cast(units_to_update,tf.float32)
                units_to_keep = tf.cast(units_to_keep,tf.float32)
                state = (units_to_keep * state) + (units_to_update * funct_output)
                new_halting_probability = tf.layers.dense(state,1,activation=tf.sigmoid,
                        name="new_halting_probability")
                halting_probability += new_halting_probability
                timestep += 1
                return state,halting_probability,timestep
            
            def halting_cond(_,halting_probability,__):
                c = tf.reduce_any(tf.less(halting_probability,self.hparams.act_threshold))
                return c

            outputs,halting_probability,timestep = tf.while_loop(halting_cond,
                    act_step,
                    [inputs, halting_probability, timestep])
        return outputs

@registry.register_hparams
def universal_transformer_mohd_hparams():
    hparams = transformer_base()
    hparams.add_hparam("act_threshold",0.99)
    return hparams
