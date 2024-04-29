


args.maxlen = 50

# variable 
self.is_training = tf.placeholder(tf.bool, shape=())   # bool
self.u = tf.placeholder(tf.int32, shape=(None))        # (bz,)
self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))    # (bz, 50)
self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))          # (bz, 50)
self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))          # (bz, 50)

mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) # (bz, 50, 1). ex. [True, False, ....] (有值為True, 沒值為False)




# embedding layer - (item lookup table, seq emb)
self.seq, item_emb_table = embedding(self.input_seq,
                                     vocab_size=itemnum + 1,
                                     num_units=args.item_hidden_units, 
                                     zero_pad=True,     # if True, the first row of item_emb_table will be 0
                                     scale=True,        # if True, embedding value will be multiply by (dim^0.5)
                                     l2_reg=args.l2_emb, # NOW, we have no it, but it must have
                                     scope="input_embeddings", 
                                     with_t=True, # if with_t: return outputs,lookup_table  else: return outputs
                                     reuse=False
                                    )   # self.seq (bz, 50, dim) ; item_emb_table (vocab, dim)

# Positional Encoding
t, pos_emb_table = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                             vocab_size=args.maxlen,
                             num_units=args.item_hidden_units + args.user_hidden_units,
                             zero_pad=False,
                             scale=False,
                             l2_reg=args.l2_emb,
                             scope="dec_pos",
                             reuse=False,
                             with_t=True
                            ) # t (bz, 50, dim+dim) ; pos_emb_table dont need


# User Encoding
u0_latent, user_emb_table = embedding(self.u[0],
                                      vocab_size=usernum + 1,
                                      num_units=args.user_hidden_units,
                                      zero_pad=False,
                                      scale=True,
                                      l2_reg=args.l2_emb,
                                      scope="user_embeddings",
                                      with_t=True,
                                      reuse=False
                                    ) # u0_latent (dim,) 

u_latent = embedding(self.u,
                     vocab_size=usernum + 1,
                     num_units=args.user_hidden_units,
                     zero_pad=False,
                     scale=True,
                     l2_reg=args.l2_emb,
                     scope="user_embeddings",
                     with_t=False,
                     reuse=True
                    ) # u_latent (bz, dim)


# Change dim to B by T by C
self.u_latent = tf.tile(tf.expand_dims(u_latent, 1), [1, tf.shape(self.input_seq)[1], 1])  # (bz, 50, dim)



# Concat item embedding with user embedding
self.hidden_units = args.item_hidden_units + args.user_hidden_units
self.seq = tf.reshape(tf.concat([self.seq, self.u_latent], 2),
                                [tf.shape(self.input_seq)[0], -1, self.hidden_units])  # (bz, 50, dim+dim)

# add position embedding
self.seq += t  # (bz, 50, dim+dim)

# Dropout
self.seq = tf.layers.dropout(self.seq,
                             rate=args.dropout_rate,
                             training=tf.convert_to_tensor(self.is_training))


self.seq *= mask # ????   # (bz, 50, dim+dim)


# Self-Attention Layer
# Build blocks
self.attention = []
for i in range(args.num_blocks):
    with tf.variable_scope("num_blocks_%d" % i):
        # Self-attention
        self.seq, attention = multihead_attention(queries=normalize(self.seq),
                                                  keys=self.seq,
                                                  num_units=self.hidden_units,
                                                  num_heads=args.num_heads,
                                                  dropout_rate=args.dropout_rate,
                                                  is_training=self.is_training,
                                                  causality=True,
                                                  scope="self_attention")
        self.attention.append(attention)
        # Feed forward
        self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                               dropout_rate=args.dropout_rate, is_training=self.is_training)
        self.seq *= mask
self.seq = normalize(self.seq) # (bz, 50, dim+dim)



###
user_emb = tf.reshape(self.u_latent, [tf.shape(self.input_seq)[0] * args.maxlen, 
                      args.user_hidden_units])  # (bz, dim)

###
pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
pos_emb = tf.nn.embedding_lookup(item_emb_table, pos) # (bz, dim)
neg_emb = tf.nn.embedding_lookup(item_emb_table, neg) # (bz, dim)

pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units]) # (bz, dim+dim)
neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units]) # (bz, dim+dim)


seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units])  # (bz * 50, dim+dim)


### TEST DATA
self.test_item = tf.placeholder(tf.int32, shape=(101))                   #(101, )
        
test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)   #(101, dim)
        
test_user_emb = tf.tile(tf.expand_dims(u0_latent, 0), [101, 1])          #(101, dim)
        
# combine item and user emb
test_item_emb = tf.reshape(tf.concat([test_item_emb, test_user_emb], 1), [-1, self.hidden_units])     #(101, dim)

self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))          #(bz, 101)
self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])  #(bz, 50, 101)
self.test_logits = self.test_logits[:, -1, :]                           #(bz, 101)



# TRAIN DATA
# prediction layer
self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1) # (bz, )
self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1) # (bz, )


# ignore padding items (0)
istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])  # (bz, )

self.loss = tf.reduce_sum(
                          -tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget
                          - tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
                          )
                           / tf.reduce_sum(istarget)   # (value)


reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

self.loss += sum(reg_losses)

tf.summary.scalar('loss', self.loss)
self.auc = tf.reduce_sum(
                         ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
                        )
                         / tf.reduce_sum(istarget)


tf.summary.scalar('auc', self.auc)
self.global_step = tf.Variable(0, name='global_step', trainable=False)


self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
self.merged = tf.summary.merge_all()











