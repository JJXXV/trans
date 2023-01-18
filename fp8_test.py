from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import popart as prt
import onnx
import torchvision.transforms.functional as F
import argparse
import time
import torch
import torchvision
import numpy as np
np.random.seed(1)
import os
from datetime import datetime
from tqdm import tqdm
import ctypes

input_index=1
output_index=1
class PerfIntervalTimer:
    # Define a simple timer object:
    def __init__(self):
        self.time = None

    def not_set(self):
        return self.time is None

    def last(self):
        return self.time

    def reset(self):
        self.time = time.perf_counter()

    def interval(self):
        now = time.perf_counter()
        interval = now - self.time
        return interval

def get_dataflow(outputs, args):
    anchors = {o: prt.AnchorReturnType("All") for o in outputs}
    return prt.DataFlow(args.bps, anchors)


def get_device(num_ipus):
    device = prt.DeviceManager().acquireAvailableDevice(numIpus=num_ipus)
    return device


def get_synthetic_data(inputs_meta, bps):
    feed_dicts = {}
    for input in inputs_meta:
        dtype = np.float32
        if input.type.tensor_type.elem_type == 10:
            dtype = np.float16
        elif input.type.tensor_type.elem_type == 2:
            dtype = np.uint8
        else:
            print(f"synthetic data input type is FLOAT32. elem_type is {input.type.tensor_type.elem_type}")
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        if not args.callback:
            shape[0] = bps * shape[0]
        data = 1+np.random.rand(*shape).astype(dtype)
        feed_dicts[input.name] = data
    return feed_dicts


def change_onnx_input_dim(model, args):
    inputs = model.graph.input
    for input in inputs:
        input.type.tensor_type.shape.dim[0].dim_value = args.batch_size
    onnx.save(model, "./temp_batch_size_model.onnx")


def get_sess_and_data(args):
    onnx_model = onnx.load(args.onnx_path)

    # change_onnx_input_dim(onnx_model, args)
    inputs = onnx_model.graph.input

    outputs = [o.name for o in onnx_model.graph.output]
    if args.op_name != "None":
        for node in onnx_model.graph.node:
            if node.name == args.op_name:
                extra_output = node.output[0]
                outputs.append(extra_output)
    dataflow = get_dataflow(outputs, args)
    device = get_device(1)

    #opts for best tput
    opts = prt.SessionOptions()
    opts.enableEngineCaching = False
    # opts.enableEngineCaching = True
    opts.groupHostSync = False
    opts.enablePrefetchDatastreams = True
    opts.defaultPrefetchBufferingDepth = 2
    opts.rearrangeAnchorsOnHost = False
    # opts.setAvailableMemoryProportion({"IPU0": availableMemoryProportion})

    if args.model_type != "fp32":
        opts.partialsTypeMatMuls = 'half'
        opts.convolutionOptions = {"partialsType": "half"}

    if args.synthetic:
        opts.syntheticDataMode = prt.SyntheticDataMode.RandomNormal
    opts.enableOutlining = True if args.outline else False
    print(f"outline is: {args.outline}")

    # sess = prt.InferenceSession("./temp_batch_size_model.onnx", dataflow, device, userOptions=opts)
    sess = prt.InferenceSession(args.onnx_path, dataflow, device, userOptions=opts)
    feed_dicts = get_synthetic_data(onnx_model.graph.input, args.bps)
    return sess, feed_dicts


def preprocess_image(args, path):
    image = Image.open(path)
    image = F.resize(image, 256)
    image_crop = F.center_crop(image, 224)
    image_crop = np.array(image_crop)
    if len(image_crop.shape) != 3:
        if len(image_crop.shape) == 2:
            image_crop = np.array([image_crop, image_crop, image_crop])
            image_crop = np.transpose(image_crop, (1, 2, 0))
        else:
            return None
    image_norm = np.transpose(image_crop, (2, 0, 1))
    inp = image_norm[np.newaxis, :]
    inp = inp[:, 0:3, :, :]
    if "VIT" not in args.onnx_path:
        mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
        std = np.array([[[0.229]], [[0.224]], [[0.225]]])
    else:
        mean = np.array([[[0.5]], [[0.5]], [[0.5]]])
        std = np.array([[[0.5]], [[0.5]], [[0.5]]])
    mean = mean[np.newaxis, :]
    std = std[np.newaxis, :]
    mul = 1.0 / (std * 255.0)
    sub = (mean / std)
    inp = inp * mul - sub
    return inp.astype(np.float32)


def inference(args):
    sess, feed_dicts = get_sess_and_data(args)
    sess.prepareDevice()
    anchors = sess.initAnchorArrays()
    stepio = prt.PyStepIO(feed_dicts, anchors)
    timer = PerfIntervalTimer()

    # Input callback is called when the data is needed:
    def input_callback(id, is_prefetch: bool):
        global input_index
        if is_prefetch:
            return
        if timer.not_set():
            timer.reset()

        curtime = datetime.now().strftime('%H%M%S%f')[:-3]
        curtime_tail = datetime.now().strftime('%H%M%S%f')[-3:-1]
        # print("INPUT_CALLBACK:",input_index,curtime,"TAIL",curtime_tail)
        print("INPUT_CALLBACK:  INDEX:",input_index,"TIMESTEP:",curtime,"TIMESTEP_TAIL",curtime_tail)
        input_index+=1
        return feed_dicts[id]

    # Called after the input buffer has been consumed:
    def input_complete_callback(id):
        return

    # Output callback is called when a buffer is needed for the result:
    def output_callback(id):
        return anchors[id]

    # Complete callback is called when the output buffer has
    # been filled (result is ready to be consumed by the host):
    def output_complete_callback(id):
        global output_index
        curtime = datetime.now().strftime('%H%M%S%f')[:-3]
        curtime_tail = datetime.now().strftime('%H%M%S%f')[-3:-1]
        print("OUTPUT_COMPLETE_CALLBACK:  INDEX:",output_index,"TIMESTEP:",curtime,"TIMESTEP_TAIL",curtime_tail)
        output_index+=1

    if args.callback:
        #Create the callback IO system:
        stepio = prt.PyStepIOCallback(
            input_callback,
            input_complete_callback,
            output_callback,
            output_complete_callback,
        )

    for i in range(10): ##warm up 10 runs
        sess.run(stepio)

    start = time.time()
    for i in range(args.iteration):
        global index
        index=0
        a=time.time()
        sess.run(stepio)
        b=time.time()
        print("Current session time",b-a,"for each batch:  ",(b-a)/args.bps)
    end = time.time()

    mean_sess_time = (end - start) / args.iteration
    throughput = args.batch_size * args.bps / mean_sess_time
    print(f'Batch Size : {args.batch_size}, Mean_sess_time : {mean_sess_time}, Throughput : {throughput}')


def validation(args):
    # assert args.batch_size==1 and args.bps==1
    print(args.onnx_path)
    onnx_model = onnx.load(args.onnx_path)
    onnx_input_name = onnx_model.graph.input[0].name
    onnx_output_name = onnx_model.graph.output[0].name
    sess, feed_dicts = get_sess_and_data(args)
    sess.prepareDevice()
    sess.weightsFromHost()
    anchors = sess.initAnchorArrays()
    # data_root = '../../../data/validation_data_for_Res50/imagenet1k-raw_clean/validation/'
    data_root = './data/'
    pred_num = 0
    total = 0
    invalid = 0

    img_count = 0
    local_bs = 0
    with open(data_root+'./val_gt.txt') as f:
        for line in tqdm(f):
            img_count += 1
            line_list = line.split()
            image_file = line_list[0]
            image_label = int(line_list[1])
            image_path = data_root + image_file
            image = preprocess_image(args, image_path)
            if image is None:
                invalid += 1
                continue

            dtype = np.float32 if "fp32" in args.onnx_path else np.float16
            if args.model_type == "fp32":
                dtype = np.float32
            else:
                dtype = np.float16
            # img = np.ascontiguousarray(image).astype(dtype)
            # stepio = prt.PyStepIO({"input": img}, anchors)
            # sess.run(stepio)
            # pred = anchors['class'][0].argmax(0)
            # if pred == image_label:
            #     pred_num += 1
            # total += 1
            if local_bs == 0:
                image_batch = image
                batch_image_label = [image_label]
            else:
                image_batch = np.concatenate((image_batch, image), axis=0)
                batch_image_label.append(image_label)
            local_bs += 1

            if local_bs == args.batch_size * args.bps or img_count == 50000:
                img = np.ascontiguousarray(image_batch).astype(dtype)
                stepio = prt.PyStepIO({onnx_input_name: img}, anchors)
                sess.run(stepio)
                total += local_bs
                res = anchors[onnx_output_name]
                # print(res)
                if args.op_name != "None":
                    extra_res = anchors[args.op_output]
                for i in range(len(batch_image_label)):
                    pred = np.argmax(res[i])
                    if pred == batch_image_label[i]:
                        pred_num += 1
                    # print(pred, batch_image_label[i])
                    if args.op_name != "None":
                        print(extra_res)
                        print(extra_res[0].shape)
                # break
                local_bs = 0
    print(f"acc: {pred_num/total}, pred_num: {pred_num}, total: {total}, invalid pic: {invalid}, valid pic: {total}")


if __name__ == '__main__':
    ctypes.cdll.LoadLibrary('./custom_op/custom_ops.so')
    ctypes.cdll.LoadLibrary('./custom_op/libpopconverter.so')
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='batch size of inference')
    parser.add_argument(
        '--onnx_path',
        type=str,
        default='./resnet50_pop/resnet50_fp8_wo_conv0_w_lsq_-4.onnx',
        help='model version, only support resnet50 for now')

    parser.add_argument(
        '--model_type',
        type=str,
        default='fp8')

    parser.add_argument(
        '--op_name',
        type=str,
        default='None')

    parser.add_argument(
        '--op_output',
        type=str,
        default='None')

    parser.add_argument(
        '--iteration',
        type=int,
        default=100,
        help='number of iteration')

    parser.add_argument(
        '--bps',
        type=int,
        default=1,
        help="number of batches per step")

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='enable inference with synthetic data')

    parser.add_argument(
        '--callback',
        action='store_true',
        help='enable inference with callback')

    parser.add_argument(
        '--validation',
        action='store_true',
        help='Test acc1 on validation data')

    parser.add_argument(
        '--outline',
        action='store_true')

    args = parser.parse_args()

    if args.validation:
        args.synthetic = False
        validation(args)
    else:
        args.synthetic = True
        inference(args)
