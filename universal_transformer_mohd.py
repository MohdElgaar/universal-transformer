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
        targets = features["targets"]

        original_shape = common_layers.shape_list(inputs) 
        squeeze_shape_inputs = [x for x in common_layers.shape_list(inputs) if x != 1]
        squeeze_shape_targets = [x for x in common_layers.shape_list(targets) if x != 1]

        #squeeze unneeded dimensions
        inputs = tf.reshape(inputs,squeeze_shape_inputs)
        targets = tf.reshape(targets,squeeze_shape_targets)
        decoder_inputs = common_layers.shift_right_3d(targets)

        #encoder bias causes padding to be ignored
        inputs_embedding_mask = common_attention.embedding_to_padding(inputs)
        self.encoder_attention_bias = \
                common_attention.attention_bias_ignore_padding(inputs_embedding_mask)
        #decoder bias causes targets to only attend to previous positions only (and itself)
        self.decoder_attention_bias = \
                common_attention.attention_bias_lower_triangle\
                (common_layers.shape_list(targets)[1])



        self.encoder_outputs = self.adaptive_computation(inputs,self.encode)
        outputs = self.adaptive_computation(decoder_inputs,self.decode)
        outputs = tf.reshape(outputs,original_shape)
        return outputs

    def encode(self, encoder_inputs, timestep):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            #positional encoding
            x = common_attention.add_timing_signal_1d(encoder_inputs)
            #timestep encoding
            timestep = tf.fill(common_layers.shape_list(x)[:-1],timestep)
            x = common_attention.add_timing_signal_1d_given_position(x,timestep)
            #encdoer-encoder attention
            y = common_attention.multihead_attention(query_antecedent=x,
                    memory_antecedent=None,
                    bias=self.encoder_attention_bias,
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
            y = tf.layers.dense(x,self.hparams.hidden_size, name="transition")
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            return x

    def decode(self, decoder_inputs, timestep):
        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            #positional encoding
            x = common_attention.add_timing_signal_1d(decoder_inputs)
            #timestep encoding
            timestep = tf.fill(common_layers.shape_list(x)[:-1],timestep)
            x = common_attention.add_timing_signal_1d_given_position(x,timestep)
            #decoder-decoder attention
            y = common_attention.multihead_attention(query_antecedent=x,
                    memory_antecedent=None,
                    bias=self.decoder_attention_bias,
                    total_key_depth=self.hparams.hidden_size,
                    total_value_depth=self.hparams.hidden_size,
                    output_depth=self.hparams.hidden_size,
                    num_heads=self.hparams.num_heads,
                    dropout_rate=self.hparams.attention_dropout)
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            #encoder-decoder attention
            y = common_attention.multihead_attention(query_antecedent=x,
                    memory_antecedent=self.encoder_outputs,
                    bias=self.decoder_attention_bias,
                    total_key_depth=self.hparams.hidden_size,
                    total_value_depth=self.hparams.hidden_size,
                    output_depth=self.hparams.hidden_size,
                    num_heads=self.hparams.num_heads,
                    dropout_rate=self.hparams.attention_dropout)
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            #transition function as fc
            y = tf.layers.dense(x,self.hparams.hidden_size, name="transition")
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            return x


    def adaptive_computation(self, inputs, funct):
        with tf.variable_scope("ACT_" + funct.__name__, reuse=tf.AUTO_REUSE):
            halting_probability = tf.zeros(common_layers.shape_list(inputs)[:-1] + [1],
                    name="halting_probability")
            timestep = tf.constant(0, dtype=tf.int32)
            def act_step(old_state, halting_probability, timestep):
                new_state = funct(old_state,timestep)
                units_to_update = tf.less(halting_probability,self.hparams.act_threshold)
                units_to_keep = tf.logical_not(units_to_update)
                units_to_update = tf.cast(units_to_update,tf.float32)
                units_to_keep = tf.cast(units_to_keep,tf.float32)
                state = (units_to_keep * old_state) + (units_to_update * new_state)
                new_halting_probability = tf.layers.dense(state,1,activation=tf.sigmoid,
                        name="new_halting_probability")
                halting_probability += new_halting_probability
                timestep += 1
                return state,halting_probability,timestep
            
            def halting_cond(_,halting_probability,timestep):
                c1 = tf.reduce_any(tf.less(halting_probability,self.hparams.act_threshold))
                c2 = tf.reduce_any(tf.less(timestep,self.hparams.act_max_steps))
                return tf.logical_and(c1,c2)

            outputs,halting_probability,timestep = tf.while_loop(halting_cond,
                    act_step,
                    [inputs, halting_probability, timestep])
        return outputs

@registry.register_hparams
def universal_transformer_mohd_hparams():
    hparams = transformer_base()
    hparams.add_hparam("act_threshold",0.99)
    hparams.add_hparam("act_max_steps",2*hparams.hidden_size)
    return hparams
