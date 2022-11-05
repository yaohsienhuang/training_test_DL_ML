#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, AveragePooling2D, Activation, multiply, Reshape, add
from tensorflow.keras import backend as K

def SqueezeNet_xclass(x):
    input_img = Input(shape = (224,224,3))
    
    conv2D_1 = Conv2D(filters=96,
                   strides = 2,
                   kernel_size=(7,7),
                   padding='same',
                   activation='relu',
                   name='Conv2D_1')(input_img)
    maxPooling2D_1 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_1')(conv2D_1)
    fire2 = fire_model(maxPooling2D_1, 16, 64, 64,'fire2')
    fire3 = fire_model(fire2, 16, 64, 64,'fire3')
    fire4 = fire_model(fire3, 32, 128, 128,'fire4')
    maxPooling2D_2 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_2')(fire4)
    fire5 = fire_model(maxPooling2D_2, 32, 128, 128,'fire5')
    fire6 = fire_model(fire5, 48, 192, 192,'fire6')
    fire7 = fire_model(fire6, 48, 192, 192,'fire7')
    fire8 = fire_model(fire7, 64, 256, 256,'fire8')
    maxPooling2D_3 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_3')(fire8)
    fire9 = fire_model(maxPooling2D_3, 64, 256, 256,'fire9')
    dropout1 = Dropout(0.5)(fire9)
    conv2D_10 = Conv2D(kernel_size=(1,1), filters=15, padding='same', activation='relu', name='Conv2D_10')(dropout1)
    gap = GlobalAveragePooling2D()(conv2D_10)
    fc = Dense(x,activation='softmax')(gap)
    
    squeeze_net_model = Model(input_img, fc, name='SqueezeNet')
    
    return squeeze_net_model

def SqueezeNet(x):
    input_img = Input(shape = (224,224,3))
    
    conv2D_1 = Conv2D(filters=96,
                   strides = 2,
                   kernel_size=(7,7),
                   padding='same',
                   activation='relu',
                   name='Conv2D_1')(input_img)
    maxPooling2D_1 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_1')(conv2D_1)
    fire2 = fire_model(maxPooling2D_1, 16, 64, 64,'fire2')
    fire3 = fire_model(fire2, 16, 64, 64,'fire3')
    fire4 = fire_model(fire3, 32, 128, 128,'fire4')
    maxPooling2D_2 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_2')(fire4)
    fire5 = fire_model(maxPooling2D_2, 32, 128, 128,'fire5')
    fire6 = fire_model(fire5, 48, 192, 192,'fire6')
    fire7 = fire_model(fire6, 48, 192, 192,'fire7')
    fire8 = fire_model(fire7, 64, 256, 256,'fire8')
    maxPooling2D_3 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_3')(fire8)
    fire9 = fire_model(maxPooling2D_3, 64, 256, 256,'fire9')
    dropout1 = Dropout(0.5)(fire9)
    conv2D_10 = Conv2D(kernel_size=(1,1), filters=15, padding='same', activation='relu', name='Conv2D_10')(dropout1)
    gap = GlobalAveragePooling2D()(conv2D_10)
    fc = Dense(x,activation='sigmoid')(gap)
    
    squeeze_net_model = Model(input_img, fc, name='SqueezeNet')
    
    return squeeze_net_model

def SqueezeNet_gray(x):
    input_img = Input(shape = (224,224,1))
    
    conv2D_1 = Conv2D(filters=96,
                   strides = 2,
                   kernel_size=(7,7),
                   padding='same',
                   activation='relu',
                   name='Conv2D_1')(input_img)
    maxPooling2D_1 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_1')(conv2D_1)
    fire2 = fire_model(maxPooling2D_1, 16, 64, 64,'fire2')
    fire3 = fire_model(fire2, 16, 64, 64,'fire3')
    fire4 = fire_model(fire3, 32, 128, 128,'fire4')
    maxPooling2D_2 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_2')(fire4)
    fire5 = fire_model(maxPooling2D_2, 32, 128, 128,'fire5')
    fire6 = fire_model(fire5, 48, 192, 192,'fire6')
    fire7 = fire_model(fire6, 48, 192, 192,'fire7')
    fire8 = fire_model(fire7, 64, 256, 256,'fire8')
    maxPooling2D_3 = MaxPooling2D(pool_size=(3, 3),
                        strides=(2,2),
                        name='MaxPooling2D_3')(fire8)
    fire9 = fire_model(maxPooling2D_3, 64, 256, 256,'fire9')
    dropout1 = Dropout(0.5)(fire9)
    conv2D_10 = Conv2D(kernel_size=(1,1), filters=15, padding='same', activation='relu', name='Conv2D_10')(dropout1)
    gap = GlobalAveragePooling2D()(conv2D_10)
    fc = Dense(x,activation='softmax')(gap)
    
    squeeze_net_model = Model(input_img, fc, name='SqueezeNet')
    
    return squeeze_net_model

def SEResnet(ver='18'):
    input_img = Input(shape = (224,224,3))
    # SEResnet18
    if ver == '18':
        N = [2, 2, 2, 2]
        bottleneck = False
    # SEResnet34
    elif ver == '34':
        N = [3, 4, 6, 3]
        bottleneck = False
    # SEResnet50
    elif ver == '50':
        N = [3, 4, 6, 3]
        bottleneck = True
    # SEResnet101
    elif ver == '101':
        N = [3, 6, 23, 3]
        bottleneck = True
    # SEResnet154
    elif ver == '154':
        N = [3, 8, 36, 3]
        bottleneck = True
    else:
        print('ver err, no match ver.')
        return
    
    filters = [64, 128, 256, 512]
    
    x = Conv2D(64, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_img)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0])
        else:
            x = _resnet_block(x, filters[0])

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], strides=(2, 2))
        

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k])
            else:
                x = _resnet_block(x, filters[k])
            

    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, use_bias=False, kernel_regularizer=l2(1e-4),
              activation='sigmoid')(x)

    model = Model(input_img, x, name='SEResnet' + ver)
    return model
# In[ ]:


def fire_model(x, s_1x1, e_1x1, e_3x3, fire_name):
    # squeeze part
    squeeze_x = Conv2D(kernel_size=(1,1),filters=s_1x1,padding='same',activation='relu',name=fire_name+'_s1')(x)
    # expand part
#     squeeze_x_bn = BatchNormalization()((squeeze_x))
    expand_x_1 = Conv2D(kernel_size=(1,1),filters=e_1x1,padding='same',activation='relu',name=fire_name+'_e1')(squeeze_x)
    expand_x_3 = Conv2D(kernel_size=(3,3),filters=e_3x3,padding='same',activation='relu',name=fire_name+'_e3')(squeeze_x)
    merge = Concatenate(axis=3, name=fire_name+'_merge')([expand_x_1, expand_x_3])
#     merge_bn = BatchNormalization()((merge))
    return merge



def _resnet_bottleneck_block(input_tensor, filters, strides=(1, 1)):
    bottleneck_expand = 4

    x = BatchNormalization(axis=-1)(input_tensor)
    x = Activation('relu')(x)
    
    if strides != (1, 1) or getattr(input_tensor, 'shape')[-1] != bottleneck_expand * filters:
        input_tensor = Conv2D(bottleneck_expand * filters, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, input_tensor])
    return m

def _resnet_block(input_tensor, filters, strides=(1, 1)):

    x = BatchNormalization(axis=-1)(input_tensor)
    x = Activation('relu')(x)
    
    if strides != (1, 1) or getattr(input_tensor, 'shape')[-1] != filters:
        input_tensor = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, input_tensor])
    return m

def squeeze_excite_block(input_tensor, ratio=16):
    filters = getattr(input_tensor, 'shape')[-1]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([input_tensor, se])
    return x