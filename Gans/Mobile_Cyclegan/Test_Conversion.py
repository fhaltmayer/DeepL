import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import torch
from torch import nn
import onnx
from onnx2keras import onnx_to_keras
import numpy as np
import tensorflow_addons as tfa
import time
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted


randin = torch.randn(1, 3, 256, 256)
randin_numpy = randin.numpy()
randin_numpy = randin_numpy.transpose(0, 2, 3, 1)
print(randin_numpy.shape)

class ReflectPad(tf.keras.layers.Layer):
    def __init__(self, pads, test_c_order = False):
        super(ReflectPad, self).__init__()
        self.pads = pads
        if test_c_order == False:
            self.op = [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]] 
        else:
            self.op = [[0, 0], [pads[2], pads[6]], [pads[3], pads[7]], [0, 0]]

    def call(self, inputs):
        x = tf.pad(inputs, self.op, 'REFLECT')
        return x 
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pads': self.pads
        })
        return config


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.t0 = torch.nn.ReflectionPad2d((3,3,3,3))
        self.t1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1))
        self.t2 = nn.InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t3 = nn.ReLU(inplace=True)
        self.t4 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.t5 = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t6 = nn.ReLU(inplace=True)
        self.t7 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.t8 = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t9 = nn.ReLU(inplace=True)
        self.t10 = torch.nn.ReflectionPad2d((1,1,1,1))
        self.t11 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), groups=64)
        self.t12 = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t13 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1))
        self.t14 = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t15 = nn.ReLU(inplace=True)
        self.t16 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1))
        self.t17 = nn.InstanceNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.t18= nn.ReLU(inplace=True)
        self.t19 = torch.nn.ReflectionPad2d((3,3,3,3))
        self.t20 = nn.Conv2d(16, 3, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0))

    def forward(self, input):
        x = self.t0(input)
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)
        x = self.t7(x)
        x = self.t8(x)
        x_add = self.t9(x)
        x = self.t10(x_add)
        x = self.t11(x)
        x = self.t12(x)
        x = x + x_add
        x = self.t13(x)
        return x


def get_torch():
    x = SimpleModel().cuda()
    x.eval()
    return x

def get_onnx(torch_model):
    input_name = ["input"]
    output_name = ["output"]
    torch.onnx.export(torch_model, randin.cuda(), "test_convert.onnx", verbose = True, input_names = input_name, output_names = output_name)

def get_keras(dynamic_input = False):
    onnx_model = onnx.load("/home/fico/DeepL/Gans/Mobile_Cyclegan/vangogh2photo_fixed.onnx")
    # Call the converter (input - is the main model input name, can be different for your model)
    k_model = onnx_to_keras(onnx_model, ['input'], test_c_order = True, dynamic_input = dynamic_input)

    # if dynamic_input == True:
    #     model_config = k_model.get_config()
    #     print(model_config['layers'][0])
        

    print(k_model.summary())
    
    return k_model

def get_keras_model_weights():
    model = Sequential()
    model.add(tf.keras.Input(shape=(256, 256, 3), batch_size = 1))
    model.add(ReflectPad([0, 0, 3, 3, 0, 0, 3, 3], test_c_order = True))
    model.add(layers.Conv2D(16, (7, 7), strides = (1,1)))
    model.add(tfa.layers.InstanceNormalization(axis=3, 
                                            epsilon=.00009, 
                                            center=False, 
                                            scale=False))
    for layer in model.layers:
        if "conv" in layer.get_config()["name"]:
            print(layer.get_config()["name"], np.shape(layer.get_weights()[0]))
            return model



def data_loader():
    total_imgs = natsorted(os.listdir("/media/fico/Data/Kaggle/monet2photo/testB"))
    trans = transforms.Compose([
                               transforms.Resize(256, 256),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    count = 0
    for x in total_imgs:
        print(count)
        count += 1
        img_loc = os.path.join("/media/fico/Data/Kaggle/monet2photo/testB", x)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = trans(image)
        image = tensor_image.numpy()
        image = np.expand_dims(image, axis=0)
        image = image.transpose(0,2,3,1)
        yield [image]
        if count == 10:
            break

        

    # for x, y in enumerate(data_load):
    #     y = y.numpy()
    #     y = y.transpose(0,2,3, 1)
    #     yield [y]


def tflite_conversion(keras_model):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.experimental_new_converter = True

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.representative_dataset = data_loader

        tflite_model = converter.convert()
    except:
        print("Dead")

    # Save the model.
    with open('vangogh_float32.tflite', 'wb') as f:
      f.write(tflite_model)

def test_tflite():
    interpreter = tf.lite.Interpreter(model_path="test_convert.tflite", num_threads = 4)
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]['index'], (1, 300, 300, 3))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    im = Image.open("/media/fico/Data/Kaggle/k.jpg")
    im = transforms.Resize((300, 300))(im)
    # im = transforms.RandomCrop(256)(im)
    im.show()

    im = np.array(im, dtype = np.float32)
    im = im.transpose(2,0,1)
    im = (im  - 127.5) / 127.5  
    im = np.expand_dims(im, axis=0)
    im = im.transpose(0, 2, 3, 1)
    input_shape = input_details[0]['shape']

    interpreter.set_tensor(input_details[0]['index'], im)
    start = time.time()
    interpreter.invoke()
    print(time.time()-start)
    interpreter.set_tensor(input_details[0]['index'], im)
    start = time.time()
    interpreter.invoke()
    print(time.time()-start)

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    # x = output_data.numpy()
    x = output_data.transpose(0,3,1,2)

    x = torch.from_numpy(x)

    # x = x.cpu()
    old_min = -1
    old_max = 1
    old_range = old_max - old_min
    new_range = 1 - 0
    x = (((x - old_min)*new_range)/ old_range) + 0
    # x = x.permute(0,2,3,1)
    x = x.squeeze()
    im = transforms.ToPILImage()(x)
    print(im.size)
    im.show()


def main():
    # torch_model = get_torch()
    # get_onnx(torch_model)

    keras_model = get_keras(dynamic_input = True)
# 
    # torch_out = torch_model(randin.cuda())
    # torch_out = torch_out.cpu().detach().numpy()
    # keras_out = keras_model(randin_numpy)
    # torch_out = torch_out.transpose(0, 2, 3, 1)
    # print(np.max(torch_out - keras_out))
    
    # keras_premade = get_keras_model_weights()

    tflite_conversion(keras_model)
    test_tflite()
    # data_loader()





if __name__ == "__main__":
    main()
