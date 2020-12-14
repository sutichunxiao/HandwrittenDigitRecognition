import os
from flask import Flask, request, redirect, url_for,render_template,make_response, jsonify
from werkzeug import secure_filename
import json
import base64
import numpy as np
import os
from PIL import Image,ImageOps # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid

import tensorflow as tf
from tensorflow.keras import datasets, layers, models



USE_PADDLE = 0

class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        # print(image_path)
        # print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))
        return np.argmax(y[0])

        
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_image(file):
    # 读取图片文件，并将它转成灰度图
    im = Image.open(file).convert('L')
    # 将输入图片调整为 28*28 的高质量图
    im = ImageOps.invert(im)
    im = im.resize((28, 28), Image.ANTIALIAS)
    # 将图片转换为numpy
    im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
    # 对数据作归一化处理
    im = im / 255.0 * 2.0 - 1.0
    return im

def  cut(filename):
    im = Image.open(filename)
    im_array = np.array(im)
    image_array =im_array.sum(axis=2)
    (row,col) = image_array.shape
    tempr0 = 0
    tempr1 = 0
    tempc0 = 0
    tempc1 = 0
    
    for x in range(0,row):
        if image_array.sum(axis=1)[x] != 765*col:
            tempr0 = x
            break
    
    for x in range(row-1,0,-1):
        if image_array.sum(axis=1)[x] != 765*col:
            tempr1 = x
            break

    for x in range(0,col):
        if image_array.sum(axis=0)[x] != 765*row:
            tempc0 = x
            break

    for x in range(col-1,0,-1):
        if image_array.sum(axis=0)[x] != 765*row:
            tempc1 = x
            break
    
    height = tempr1 - tempr0
    width = tempc1 - tempc0
    print(width,height)
    if height < width :
        gap = width -height
        if tempr0 - gap/2 > 0 : 
            if tempr1 + gap/2 <= row:
                tempr0 = tempr0 - gap/2
                tempr1 = tempr1 + gap/2
            else:
                tempr1 = row
                tempr0 = row-width
        else:
            tempr0 = 0
            tempr1 = width
    elif height > width :
        gap = height - width
        if tempc0-gap/2 > 0 :
            if tempc1 + gap/2 < col:
                tempc0 = tempc0 - gap/2
                tempc1 = tempc1 + gap/2
            else:
                tempc1 = col
                tempc0 = col-height
        else:
            tempc0 = 0
            tempc1 = height

    print(tempr0,tempc0,tempr1,tempc1)
    box = (tempc0,tempr0,tempc1,tempr1)
    region = im.crop(box)
    region.save('w2.png')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_success', filename=filename))
    # fp = open('index.html')
    return render_template('index.html', name="Alex")


@app.route('/uploadcanvas', methods=["POST"])
def uploadFromCanvas():
    print('uploadFromCanvas')
    print('request.method =',request.method)
    if request.method == "POST":
        #通过get_data方式
        #recv_data = request.get_data()
        # 通过get_json方式
        recv_data = request.get_json()

        if recv_data is None:
            print("request.get_json() is None")
            recv_data = request.get_data()

        # print("recv_data=", recv_data)
        json_re = json.loads(recv_data)
        # print("json_re=", json_re)
        imgRes = json_re['uploadImg']
        # print("imgRes=",imgRes)
        imgdata = base64.b64decode(imgRes)
        # print("imgdata=",imgdata)
        file = open('1.png', "wb")
        file.write(imgdata)
        file.close()
        cut("1.png")
        number = 0

        if USE_PADDLE:
            tensor_img = load_image('w2.png')
            save_dirname = "recognize_digits.inference.model"
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            inference_scope = fluid.core.Scope()
            with fluid.scope_guard(inference_scope):
                # 使用 fluid.io.load_inference_model 获取 inference program desc,
                # feed_target_names 用于指定需要传入网络的变量名
                # fetch_targets 指定希望从网络中fetch出的变量名
                [inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(save_dirname, exe, None, None)

                # 将feed构建成字典 {feed_target_name: feed_target_data}
                # 结果将包含一个与fetch_targets对应的数据列表
                results = exe.run(inference_program,
                                        feed={feed_target_names[0]: tensor_img},
                                        fetch_list=fetch_targets)
                lab = numpy.argsort(results)

                # 打印 infer_3.png 这张图片的预测结果
                # img=Image.open('image/infer_3.png')
                # plt.imshow(img)
                print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])
                number = lab[0][0][-1]
        else:
            app = Predict()
            number = app.predict('w2.png')

        data = {
        "name":"python",
        "age":str(number)
        }
    return jsonify(data)


@app.route('/upload_success')
def upload_success():
    return '''
    <!doctype html>
    <title>上传成功</title>
    <h1>上传成功</h1>
    <a href="/upload/">继续上传</a>
    '''

if __name__ == '__main__':
    app.run(host='192.168.1.131', port=8080, debug=True)