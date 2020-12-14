import os
from flask import Flask, Response, request,make_response,jsonify
app = Flask(__name__)
print(type(app), app)
# <class 'flask.app.Flask'> <Flask 'run_flask'>

print(app.root_path)  # 返回的是当前运行文件run_flask.py所在的目录
# D:\ZF\1_ZF_proj\2_YOLO\YOLO数据集相关

@app.route('/')
def index():
    return 'Hello world!'


@app.route('/img_flip', methods=["POST","GET"])
def process_img():
    # 接收前端传来的图片  image定义上传图片的key
    
    data = {
        "name":"python",
        "age":"18"
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
