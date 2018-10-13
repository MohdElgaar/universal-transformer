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
        """
        Args:
            features["inputs"]:
            features["targets"]:
                tensors with shape [batch_size, ..., hidden_size]
        Return:
            decoder_outputs: pre-softmax activations of same size as inputs

        I assume that the input is a time series such that input size is
        [batch_size,sequence_length,hidden_size]
        """
            
        inputs = features["inputs"]
        targets = features["targets"]

        #tensor2tensor provides 4d tensors and axis=2 is useless
        #so I remove it for ease of handling
        original_shape = common_layers.shape_list(inputs) 
        squeeze_shape_inputs = [x for x in \
                common_layers.shape_list(inputs) if x != 1]
        squeeze_shape_targets = [x for x in \
                common_layers.shape_list(targets) if x != 1]

        #squeeze unneeded dimensions
        inputs = tf.reshape(inputs,squeeze_shape_inputs)
        targets = tf.reshape(targets,squeeze_shape_targets)
        decoder_inputs = common_layers.shift_right_3d(targets)

        #encoder bias causes padding to be ignored
        inputs_embedding_mask = common_attention.\
                embedding_to_padding(inputs)
        self.encoder_attention_bias = common_attention.\
                attention_bias_ignore_padding(inputs_embedding_mask)
        #decoder bias causes targets to only attend to previous positions only (and itself)
        self.decoder_attention_bias = \
                common_attention.attention_bias_lower_triangle\
                (common_layers.shape_list(targets)[1])



        #process encoder and save the result for decoder to use
        #and process decoder
        self.encoder_outputs = self.adaptive_computation(inputs,self.encode)
        outputs = self.adaptive_computation(decoder_inputs,self.decode)
        #reshape output back to 4d
        outputs = tf.reshape(outputs,original_shape)
        return outputs

    def encode(self, encoder_inputs, timestep):
        """
        Args:
            encoder_inputs: inputs of shape [batch_size,sequence_length,
                            hidden_size]
            timestep: used for timestep encoding during ACT
        Return:
            encoder_outputs: the result of passing the encoder_input
                             through the encoder layers.
                             Input shape is preserved.

        This function is one step of encoding.
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            #positional encoding
            x = common_attention.add_timing_signal_1d(encoder_inputs)
            #timestep encoding
            #positional encoding with the same position for every unit
            #(position=timestep) is equivalent to timestep encoding
            x = common_attention.add_timing_signal_1d_given_position(x,
                    timestep)
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
            y = tf.layers.dense(x,self.hparams.hidden_size,
                    name="transition")
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            return x

    def decode(self, decoder_inputs, timestep):
        """
        Args:
            decoder_inputs: targets of shape [batch_size,sequence_length,
                            hidden_size]. Sequence is shifter right
                            by one.
            timestep: used for timestep encoding during ACT
        Return:
            decoder_outputs: the result of passing the decoder_input
                             through the edecoderlayers.
                             Input shape is preserved.

        This function is one step of decoding.
        """
        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            #positional encoding
            x = common_attention.add_timing_signal_1d(decoder_inputs)
            #timestep encoding
            x = common_attention.add_timing_signal_1d_given_position(x,
                    timestep)
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
                    bias=self.encoder_attention_bias,
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
            y = tf.layers.dense(x,self.hparams.hidden_size,
                    name="transition")
            #residual connection and dropout
            x = common_layers.layer_postprocess(x,y,self.hparams)
            #layer norm
            x = common_layers.layer_norm(x)
            return x


    def adaptive_computation(self, inputs, funct):
        """
        Args:
            inputs: inputs of funct (one of encode,decode)
            funct: function object
        Return:
            outputs: the result of recurrent funct using ACT
        """
        with tf.variable_scope("ACT_" + funct.__name__,
                reuse=tf.AUTO_REUSE):
            #treat hidden_size as one unit
            reduced_shape = common_layers.shape_list(inputs)[:-1] + [1] 
            halting_probability = tf.zeros(reduced_shape,
                    name="halting_probability") #[batch_size,sequence_length,1]
            timestep = tf.zeros(reduced_shape[:-1], dtype=tf.int32,
                    name="timestep") #[batch_size,sequence_length]
            active_mask = tf.ones(reduced_shape,
                    name="active_mask") #[batch_size,sequence_length,1]
            accumulate_outputs = tf.zeros_like(inputs,
                    name="accumulate_outputs") #[batch_size,sequence_length,hidden_size]
            def act_step(inputs, accumulate_outputs, active_mask, halting_probability, timestep):
                #steps outputs and state
                #state is analogous to RNN state s.t state=f^-1(outputs)
                outputs = funct(inputs,timestep)
                state = tf.layers.dense(outputs,self.hparams.hidden_size,
                        activation=tf.nn.relu,
                        name="outputs_to_state")

                #halting probability is a function of the state
                new_halting_probability = tf.layers.dense(state,1,
                        activation=tf.sigmoid,
                        name="new_halting_probability")
                #these are units still active after this step
                new_active_mask = tf.to_float(tf.less(halting_probability + \
                        new_halting_probability,
                        self.hparams.act_threshold))

                #these are units the were active and halted at this step
                #in this case accumulate using remainders
                #(affected by threshold)
                newly_halted = active_mask * new_active_mask
                remainders = 1.0 - halting_probability
                accumulate_outputs += newly_halted * remainders * outputs

                #for active units, accumulate using halting_probability
                halting_probability += new_halting_probability
                accumulate_outputs += new_active_mask * \
                        new_halting_probability * outputs

                #timestep holds timestep up to N(t)-1
                timestep += tf.to_int32(tf.squeeze(new_active_mask,-1))
                return outputs, accumulate_outputs, new_active_mask, new_halting_probability, timestep
            
            def halt_cond(_,__,___,halting_probability,timestep):
                #some probability is less than threshold
                c1 = tf.reduce_any(tf.less(halting_probability,
                    self.hparams.act_threshold))
                #some timestep is less than max time
                c2 = tf.reduce_any(tf.less(timestep,
                    self.hparams.act_max_steps))
                return tf.logical_and(c1,c2)
            
            outputs, accumulate_outputs, new_active_mask, new_halting_probability, timestep = \
                    tf.while_loop(halt_cond, act_step,
                            [inputs, accumulate_outputs, active_mask, halting_probability, timestep])
        return accumulate_outputs

@registry.register_hparams
def universal_transformer_mohd_hparams():
    """
    registry of hparams set of tensor2tensor
    This is the hparams of transformer in addition to ACT parameters
    """
    hparams = transformer_base()
    hparams.add_hparam("act_threshold",0.99) #default value from ACT paper

    #small value of max_steps restricts pondering during training
    #this is value used in ACT paper experiments
    hparams.add_hparam("act_max_steps",100) 
                                            
    return hparams
