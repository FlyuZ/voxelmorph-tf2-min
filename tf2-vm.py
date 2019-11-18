# run
import os, sys

# 引入第三方库
import numpy as np
import tensorflow as tf

# 本地引入  自行修改路径
sys.path.append('/home/xxx/xxx/tf2-vm-cvpr/')
import voxelmorph as vxm

# 读取数据  使用kaggle上提供的mri-2d数据
core_path = '/home/xxx/xxx/mri-2d/'
x_train = np.load(os.path.join(core_path, 'train_vols.npy'))
x_val = np.load(os.path.join(core_path, 'validate_vols.npy'))

ndims = 2  # 数据是二维
vol_shape = x_train.shape[1:]  # 输入数据shape
nb_enc_features = [32, 32, 32, 32]  # unet 通道数
nb_dec_features = [32, 32, 32, 32, 32, 16]

# unet层
unet = vxm.networks.unet_core(vol_shape, nb_enc_features, nb_dec_features)

# 卷积层
disp_tensor = tf.keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)

# SpatialTransformer层
moved_image_tensor = vxm.layers.SpatialTransformer(name='spatial_transformer')([unet.inputs[0], disp_tensor])

vxm_model = tf.keras.models.Model(unet.inputs, [moved_image_tensor, disp_tensor])

# losses. 一个是mse 一个是Grad-l2 ，两个相加，权重为1000:10 （个人行为）
losses = ['mse', vxm.losses.Grad('l2').loss]

loss_weights = [1000, 10]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# 输出看一下model信息
vxm_model.summary()

# 数据生成器，输入为两张图，一个是moving，一个是fixed。输出初始化为fixed和空
def vxm_data_generator(x_data, batch_size=32):
    vol_shape = x_data.shape[1:]
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]
        yield inputs, outputs      
        
train_generator = vxm_data_generator(x_train)
# 测试一下数据生成器
# 画图
input_sample, output_sample = next(train_generator)
slices_2d = [f[0,...,0] for f in input_sample + output_sample]
titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'zero']
vxm.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True)

# 训练
nb_epochs = 10
steps_per_epoch = 10
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=1)

# 保存模型
# vxm_model.save_weights('cvpr-mri.h5')

# 推理
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)

# 显示结果
slices_2d = [f[0,...,0] for f in val_input + val_pred]
titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
vxm.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True)
flow = val_pred[1].squeeze()[::3,::3]
vxm.plot.flow([flow], width=10)