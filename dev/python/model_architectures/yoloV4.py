import tensorflow as tf


def conv2d_bn_leaky(x, filters, kernel_size, strides=(1,1), padding='same',name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, name=name+"_conv2d")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_batch_normalization")(x)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	
def tiny_block(x,name):
    x = conv2d_bn_leaky(x, x.shape[-1],(3, 3),name=name+"_1")
    x1 = x[..., x.shape[-1]//2:]
    x2 = conv2d_bn_leaky(x1, x1.shape[-1], (3, 3),name=name+"_2")
    x3 = conv2d_bn_leaky(x2, x2.shape[-1], (3, 3),name=name+"_3")
    x3 = tf.keras.layers.Concatenate()([x3,x2])
    x3 = conv2d_bn_leaky(x3, x3.shape[-1], (1, 1),name=name+"_4")
    x4 = tf.keras.layers.Concatenate()([x, x3])
    return x4,x3
	
def backbone(x):
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,32,(3,3),strides=(2,2),padding='valid',name="block_1_1")
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,64,(3,3),strides=(2,2),padding='valid',name="block_2_1")

    x,_ = tiny_block(x,name="block_3")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x,_ = tiny_block(x,name="block_4")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x,x1 = tiny_block(x,name="block_5")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv2d_bn_leaky(x, x.shape[-1], (3, 3), strides=(1, 1), padding='same',name="block_5_5")
    output1 = conv2d_bn_leaky(x, x.shape[-1]//2, (1, 1), strides=(1, 1), padding='same',name="block_5_6")
    x = conv2d_bn_leaky(output1, output1.shape[-1] // 2, (1, 1), strides=(1, 1), padding='same',name="block_5_7")
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    output2 = tf.keras.layers.Concatenate()([x, x1])

    return [output2, output1]

#def last_conv_layer(inputs, args):
#    output_layers = []
#    head_conv_filters = [256, 512]
#
#    for index, x in enumerate(inputs):
#
#        x = conv2d_bn_leaky(x, head_conv_filters[index], (3, 3), name='yolov3_head_%d_1' % (index+1))
#        x = tf.keras.layers.Conv2D(x.shape[-1]//4, (1, 1), use_bias=True,
#        name='yolov3_head_%d_2_conv2d' % (index+1))(x)
								   
#        output_layers.append(x)
		
#    return output_layers
	
def head_2layers(x, args):
	x = tf.reshape(x, [-1, 32*32*(int(args.head_size)*2+int(args.head_size))])
	x = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)(x)
	if args.head_type=='2l':
		x_coord = tf.keras.layers.Dense(8, activation=None)(x)
		x_detection = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)
		x = tf.keras.layers.Concatenate()([x_coord, x_detection])
	if args.head_type=='2l_s':
		x = tf.keras.layers.Dense(9, activation=tf.keras.activations.sigmoid)(x)
	return x
	
def head_5layers(x, args):
	x = tf.reshape(x, [-1, 16*16*args.head_size])
	x = tf.keras.layers.Dense(4*16*args.head_size, activation=tf.keras.activations.relu)(x)
	x = tf.keras.layers.Dense(16*args.head_size, activation=tf.keras.activations.relu)(x)
	x = tf.keras.layers.Dense(4*args.head_size, activation=tf.keras.activations.relu)(x)
	x = tf.keras.layers.Dense(args.head_size, activation=tf.keras.activations.relu)(x)
	x = tf.keras.layers.Dense(9, activation=tf.keras.activations.sigmoid)(x)
	return x

def Yolov4_tiny(args):
	input = tf.keras.layers.Input((None, None, 3))
	outputs = backbone(input)
#    outputs = last_conv_layer(outputs,args)
	if '2l' in args.head_type:
		outputs = head_2layers(outputs[0], args)
	if '5l_s' in args.head_type:
		outputs = head_5layers(outputs[0], args)

	model = tf.keras.Model(inputs=input, outputs=outputs)
	return model