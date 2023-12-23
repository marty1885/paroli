from rknn.api import RKNN
import onnxruntime as ort

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='onnx model path')
parser.add_argument('output', type=str, help='rknn output path')

args = parser.parse_args()

# Load ONNX model and see if the 'g' input exists
sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
input_names = [i.name for i in sess.get_inputs()]
input_size_list = [[1, 192, 55], [1, 1, 55]]
inputs = ['z', 'y_mask']
if 'g' in input_names:
    input_size_list.append([1, 512, 1])
    inputs.append('g')


rknn = RKNN()
rknn.config(target_platform="RK3588")
ret = rknn.load_onnx(args.model,
    input_size_list=input_size_list,
    inputs=inputs,
)
if ret != 0:
    print('load onnx failed')
    exit(ret)
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('build failed')
    exit(ret)
ret = rknn.export_rknn(args.output)
if ret != 0:
    print('export failed')
    exit(ret)

