# This code is the simplest example of image compression based on neural networks
# Comparison with JPEG is provided as well
# It is a demonstation for Information Theory course
# Written by Evgeny Belyaev, July 2024.

import os
import math
import numpy
from copy import deepcopy
from matplotlib import pyplot as plt
from PIL import Image
import imghdr
import tensorflow
from tensorflow import cast
import keras
from keras import layers
from keras import Model
from keras import backend as K

from params import Params
from skimage.metrics import structural_similarity as ssim

from EntropyCodec import *

# If 0, then the training will be started, otherwise the model will be readed from a file
load_pretrained = False

# Number of images to be compressed and shown from the test folder
num_images_to_show = 5
test_dir = './test/'
train_dir = './train/'
w, h = 128, 128

# Compute PSNR in RGB domain
def PSNR_RGB(image1, image2):
    width, height = image1.size
    I1 = numpy.array(image1.getdata()).reshape(image1.size[0], image1.size[1], 3)
    I2 = numpy.array(image2.getdata()).reshape(image2.size[0], image2.size[1], 3)
    I1 = numpy.reshape(I1, width * height * 3)
    I2 = numpy.reshape(I2, width * height * 3)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    mse = numpy.mean((I1 - I2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    # print("PSNR = %5.2f dB" % psnr)
    return psnr


# Compute PSNR between two vectors
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


# reads all images from folder and puts them into x array
def load_images_from_folder(folder_name):
    dir_list = os.listdir(folder_name)
    Nmax = 0
    for name in dir_list:
        fullname = folder_name + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('')
        else:
            Nmax = Nmax + 1

    x = numpy.zeros([Nmax, w, h, 3])
    N = 0
    for name in dir_list:
        fullname = folder_name + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('Unknown image format for file: ', name)
        else:
            print('Progress: N = %i' % N)
            image = Image.open(fullname)
            I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
            x[N, :, :, :] = I1
            N = N + 1
    return x


# Model training function
def get_model(train_dir, model_name='', add_noise=False, load_pretrained=False, params=Params()):
    input = layers.Input(shape=(w, h, 3))
    # Encoder
    for i in range(params.n_layers):
        e = layers.Conv2D(params.in_channels[i], (params.kernel_sizes[i], params.kernel_sizes[i]),
                           activation=params.activation, padding="same")(input if i == 0 else e)
        e = layers.AveragePooling2D((2, 2), padding="same")(e)
    # e1 = layers.Conv2D(params.in_channels[0], (7, 7), activation="relu", padding="same")(input)
    # e1 = layers.AveragePooling2D((2, 2), padding="same")(e1)
    # e2 = layers.Conv2D(params.in_channels[1], (5, 5), activation="relu", padding="same")(e1)
    # e2 = layers.AveragePooling2D((2, 2), padding="same")(e2)
    # e3 = layers.Conv2D(params.in_channels[2], (3, 3), activation="relu", padding="same")(e2)
    # e3 = layers.AveragePooling2D((2, 2), padding="same")(e3)
    layers.BatchNormalization()
    if add_noise:
        maxt = keras.ops.max(e)
        e = e + maxt * keras.random.uniform(shape=(h // (2**params.n_layers), h // (2**params.n_layers), 16), minval=-1.0 / pow(2, params.bt + 1),
                                              maxval=1.0 / pow(2, params.bt + 1), dtype=None, seed=None)

    # Decoder
    for i in range(params.n_layers - 1, -1, -1):
        x = layers.Conv2DTranspose(params.in_channels[i],
                                   (params.kernel_sizes[i], params.kernel_sizes[i]), strides=2,
                                   activation=params.activation, padding="same")(e if i == params.n_layers - 1 else x)
    # x = layers.Conv2DTranspose(params.in_channels[2], (3, 3), strides=2, activation="relu", padding="same")(e3)
    # x = layers.Conv2DTranspose(params.in_channels[1], (5, 5), strides=2, activation="relu", padding="same")(x)
    # x = layers.Conv2DTranspose(params.in_channels[0], (7, 7), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    encoder = Model(input, e)
    decoder = Model(e, x)
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss='mean_squared_error')
    autoencoder.summary()

    if not load_pretrained:
        print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
        xtrain = load_images_from_folder(train_dir)
        xtrain = xtrain / 255

        with tensorflow.device('gpu'):
            autoencoder.fit(xtrain, xtrain, epochs=params.n_epoch, batch_size=params.batch_size, shuffle=True)

        if model_name == '':
            if add_noise:
                encoder.save('encoder2.keras')
                decoder.save('decoder2.keras')
            else:
                encoder.save('encoder.keras')
                decoder.save('decoder.keras')
        else:
            encoder.save(model_name + '_encoder.keras')
            decoder.save(model_name + '_decoder.keras')
    else:
        if model_name == '':
            if add_noise:
                encoder = keras.models.load_model('encoder2.keras', safe_mode=False)
                decoder = keras.models.load_model('decoder2.keras', safe_mode=False)
            else:
                encoder = keras.models.load_model('encoder.keras', safe_mode=False)
                decoder = keras.models.load_model('decoder.keras', safe_mode=False)
        else:
            encoder = keras.models.load_model(model_name + '_encoder.keras', safe_mode=False)
            decoder = keras.models.load_model(model_name + '_decoder.keras', safe_mode=False)
    return encoder, decoder


# Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
def entropy_encoder(filename, encoder_layers, size_z, size_h, size_w):
    temp = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    for z in range(size_z):
        for h in range(size_h):
            for w in range(size_w):
                temp[z][h][w] = encoder_layers[z][h][w]
    temp = deepcopy(encoder_layers)
    maxbinsize = (size_h * size_w * size_z)
    bitstream = numpy.zeros(maxbinsize, numpy.uint8, 'C')
    stream_size = numpy.zeros(1, numpy.int32, 'C')
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, stream_size)
    name = filename
    path = './'
    fp = open(os.path.join(path, name), 'wb')
    out = bitstream[0:stream_size[0]]
    out.astype('uint8').tofile(fp)
    fp.close()


# Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
def entropy_decoder(filename, size_z, size_h, size_w):
    fp = open(filename, 'rb')
    bitstream = fp.read()
    fp.close()
    bitstream = numpy.frombuffer(bitstream, dtype=numpy.uint8)
    declayers = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    frame_offset = numpy.zeros(1, numpy.int32, 'C')
    frame_offset[0] = 0
    HiddenLayersDecoder(declayers, size_w, size_h, size_z, bitstream, frame_offset)
    return declayers


# This function is searching for the JPEG quality factor (QF)
# which provides neares compression to TargetBPP
def JPEGRDSingleImage(X, TargetBPP, i):
    X = X * 255
    image = Image.fromarray(X.astype('uint8'), 'RGB')
    width, height = image.size
    realbpp = 0
    real_q = 0
    for Q in range(101):
        image.save('test.jpeg', "JPEG", quality=Q)
        image_dec = Image.open('test.jpeg')
        bytesize = os.path.getsize('test.jpeg')
        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(image, image_dec)
        if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realpsnr = psnr
            real_q = Q
    image.save('test.jpeg', "JPEG", quality=real_q)
    image_dec = Image.open('test.jpeg')
    I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    I2 = numpy.array(image_dec.getdata()).reshape(image_dec.size[0], image_dec.size[1], 3)

    # print('\n\n Size = ',numpy.shape(I1))
    psnr = ssim(I1[:, :, 0], I2[:, :, 0], data_range=255.0)
    psnr = psnr + ssim(I1[:, :, 1], I2[:, :, 1], data_range=255.0)
    psnr = psnr + ssim(I1[:, :, 2], I2[:, :, 2], data_range=255.0)
    realpsnr = psnr / 3.0
    JPEGfilename = 'image%i.jpeg' % i
    image.save(JPEGfilename, "JPEG", quality=real_q)
    return real_q, realbpp, realpsnr


def neural_compressor(enc, dec):
    # Run the model for first NumImagesToShow images from the test set
    encoded_layers = enc.predict(xtest, batch_size=num_images_to_show)
    max_encoded_layers = numpy.zeros(num_images_to_show, numpy.float16, 'C')

    # normalization the layer to interval [0,1)
    for i in range(num_images_to_show):
        max_encoded_layers[i] = numpy.max(encoded_layers[i])
        encoded_layers[i] = encoded_layers[i] / max_encoded_layers[i]

    # Quantization of layer to b bits
    encoded_layers1 = numpy.clip(encoded_layers, 0, 0.9999999)
    encoded_layers1 = cast(encoded_layers1 * pow(2, params.b), "int32")

    # Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(num_images_to_show, numpy.float16, 'C')
    declayers = numpy.zeros((num_images_to_show, 16, 16, 16), numpy.uint8, 'C')
    for i in range(num_images_to_show):
        binfilename = 'image%i.bin' % i
        entropy_encoder(binfilename, encoded_layers1[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = entropy_decoder(binfilename, 16, 16, 16)

    # Dequantization and denormalization of each layer
    print(bpp)
    shift = 1.0 / pow(2, params.b + 1)
    declayers = cast(declayers, "float32") / pow(2, params.b)
    declayers = declayers + shift
    encoded_layers_quantized = numpy.zeros((num_images_to_show, 16, 16, 16), numpy.double, 'C')
    for i in range(num_images_to_show):
        encoded_layers_quantized[i] = cast(declayers[i] * max_encoded_layers[i], "float32")
        encoded_layers[i] = cast(encoded_layers[i] * max_encoded_layers[i], "float32")
    decoded_imgs = dec.predict(encoded_layers, batch_size=num_images_to_show)
    decoded_imgsQ = dec.predict(encoded_layers_quantized, batch_size=num_images_to_show)
    return bpp, decoded_imgs, decoded_imgsQ


# Main function
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    # Load test images
    xtest = load_images_from_folder(test_dir)
    xtest = xtest / 255

    # Train/load the model
    params = Params(w=128, h=128, bt=2, n_epoch=100, batch_size=32, in_channels=[256, 128, 16], kernel_sizes=[7, 5, 3], n_layers=3, b=2, activation='gelu')
    encoder, decoder = get_model(train_dir, model_name='my_model_large', add_noise=True, load_pretrained=False, params=params)

    params = Params() # defaults for model loaded
    encoder2, decoder2 = get_model(train_dir, add_noise=True, load_pretrained=True, params=params)

    bpp, decoded_imgs, decoded_imgsQ = neural_compressor(encoder, decoder)
    bpp2, decoded_imgs2, decoded_imgsQ2 = neural_compressor(encoder2, decoder2)

    # Shows NumImagesToShow images from the test set
    # For each image the following results are presented
    # Original image (RAW)
    # Image, represented by the model (without noise additing during training)
    # Image, represented by the model (with noise additing during training)
    # Corresponding JPEG image at the same compression level
    # Q is the quality metric measured as SSIM
    # bpp is bit per pixel after compression (bpp for RAW data is 24 bpp)
    for i in range(num_images_to_show):
        title = ''
        plt.subplot(4, num_images_to_show, i + 1).set_title(title, fontsize=10)
        if i == 0:
            plt.subplot(4, num_images_to_show, i + 1).text(-50, 64, 'RAW')
        plt.imshow(xtest[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(num_images_to_show):
        # psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ[i, :, :, :])
        psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ[i, :, :, 0], data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ[i, :, :, 1], data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ[i, :, :, 2], data_range=1.0)
        psnr = psnr / 3.0

        # title = '%2.2f %2.2f' % (psnr, bpp[i])
        title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp[i])
        plt.subplot(4, num_images_to_show, num_images_to_show + i + 1).set_title(title, fontsize=10)
        if i == 0:
            plt.subplot(4, num_images_to_show, num_images_to_show + i + 1).text(-50, 64, 'AE1')
        plt.imshow(decoded_imgsQ[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(num_images_to_show):
        # psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ2[i, :, :, :])
        psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ2[i, :, :, 0], data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ2[i, :, :, 1], data_range=1.0)
        psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ2[i, :, :, 2], data_range=1.0)
        psnr = psnr / 3.0
        # title = '%2.2f %2.2f' % (psnr, bpp2[i])
        title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp2[i])
        plt.subplot(4, num_images_to_show, 2 * num_images_to_show + i + 1).set_title(title, fontsize=10)
        if i == 0:
            plt.subplot(4, num_images_to_show, 2 * num_images_to_show + i + 1).text(-50, 64, 'AE2')
        plt.imshow(decoded_imgsQ2[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(num_images_to_show):
        JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(xtest[i, :, :, :], bpp[i], i)
        JPEGfilename = 'image%i.jpeg' % i
        JPEGimage = Image.open(JPEGfilename)
        # title = '%2.2f %2.2f' % (JPEGrealpsnr,JPEGrealbpp)
        title = 'Q=%2.2f bpp=%2.2f' % (JPEGrealpsnr, JPEGrealbpp)
        plt.subplot(4, num_images_to_show, 3 * num_images_to_show + i + 1).set_title(title, fontsize=10)
        if i == 0:
            plt.subplot(4, num_images_to_show, 3 * num_images_to_show + i + 1).text(-50, 64, 'JPEG')
        plt.imshow(JPEGimage, interpolation='nearest')
        plt.axis(False)
    plt.show()


    # Готовим контейнеры для данных
    b_values = [2, 3, 4, 5]
    ae2_results = []  # (bpp, SSIM)
    jpeg_results = []  # (bpp, SSIM)
    custom_codec_results = []  # (bpp, SSIM)

    for b in b_values:
        params.b = b

        # Считаем bpp и декодированные изображения для AE2
        bpp2, decoded_imgs2, decoded_imgsQ2 = neural_compressor(encoder2, decoder2)
        avg_ssim_ae2 = 0
        for i in range(num_images_to_show):
            ssim_val = ssim(xtest[i, :, :, 0], decoded_imgsQ2[i, :, :, 0], data_range=1.0)
            ssim_val += ssim(xtest[i, :, :, 1], decoded_imgsQ2[i, :, :, 1], data_range=1.0)
            ssim_val += ssim(xtest[i, :, :, 2], decoded_imgsQ2[i, :, :, 2], data_range=1.0)
            avg_ssim_ae2 += ssim_val / 3.0
        avg_ssim_ae2 /= num_images_to_show
        avg_bpp_ae2 = numpy.mean(bpp2)
        ae2_results.append((avg_bpp_ae2, avg_ssim_ae2))

        # Считаем bpp и SSIM для JPEG
        avg_ssim_jpeg = 0
        avg_bpp_jpeg = 0
        for i in range(num_images_to_show):
            JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(xtest[i, :, :, :], bpp2[i], i)
            avg_ssim_jpeg += JPEGrealpsnr
            avg_bpp_jpeg += JPEGrealbpp
        avg_ssim_jpeg /= num_images_to_show
        avg_bpp_jpeg /= num_images_to_show
        jpeg_results.append((avg_bpp_jpeg, avg_ssim_jpeg))

        # Считаем bpp и SSIM для предложенного кодека
        bpp, decoded_imgs, decoded_imgsQ = neural_compressor(encoder, decoder)
        avg_ssim_custom = 0
        for i in range(num_images_to_show):
            ssim_val = ssim(xtest[i, :, :, 0], decoded_imgsQ[i, :, :, 0], data_range=1.0)
            ssim_val += ssim(xtest[i, :, :, 1], decoded_imgsQ[i, :, :, 1], data_range=1.0)
            ssim_val += ssim(xtest[i, :, :, 2], decoded_imgsQ[i, :, :, 2], data_range=1.0)
            avg_ssim_custom += ssim_val / 3.0
        avg_ssim_custom /= num_images_to_show
        avg_bpp_custom = numpy.mean(bpp)
        custom_codec_results.append((avg_bpp_custom, avg_ssim_custom))

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in ae2_results], [x[1] for x in ae2_results], label="AE2", marker='o')
    plt.plot([x[0] for x in jpeg_results], [x[1] for x in jpeg_results], label="JPEG", marker='s')
    plt.plot([x[0] for x in custom_codec_results], [x[1] for x in custom_codec_results], label="Proposed Codec", marker='^')

    plt.xlabel("bpp (Bits per Pixel)")
    plt.ylabel("Average SSIM")
    plt.title("Average SSIM vs bpp")
    plt.legend()
    plt.grid(True)
    plt.show()