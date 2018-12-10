#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer

import shutil
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None) and (args.standalone_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None) and (args.standalone_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path, standalone_output_path, phase):
    try:
        sess = tf.InteractiveSession()
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if data_output_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path is not None:
            print_stderr('Saving source...')
            with open(code_output_path, 'w') as src_out:
                src_out.write(transformer.transform_source())

        if standalone_output_path:
            filename, _ = os.path.splitext(os.path.basename(standalone_output_path))
            temp_folder = os.path.join(os.path.dirname(standalone_output_path), '.tmp')
            os.makedirs(temp_folder)

            if data_output_path is None:
                data = transformer.transform_data()
                print_stderr('Saving data...')
                data_output_path = os.path.join(temp_folder, filename) + '.npy'
                with open(data_output_path, 'wb') as data_out:
                    np.save(data_out, data)

            if code_output_path is None:
                print_stderr('Saving source...')
                code_output_path = os.path.join(temp_folder, filename) + '.py'
                with open(code_output_path, 'wb') as src_out:
                    src_out.write(transformer.transform_source())

            checkpoint_path = os.path.join(temp_folder, filename + '.ckpt')
            graph_name = os.path.basename(standalone_output_path)
            graph_folder = os.path.dirname(standalone_output_path)
            input_node = transformer.graph.nodes[0].name
            output_node = transformer.graph.nodes[-1].name
            tensor_shape = transformer.graph.get_node(input_node).output_shape
            tensor_shape_list = [tensor_shape.batch_size, tensor_shape.height, tensor_shape.width, tensor_shape.channels]

            sys.path.append(os.path.dirname(code_output_path))
            module = os.path.splitext(os.path.basename(code_output_path))[0]
            class_name = transformer.graph.name
            KaffeNet = getattr(__import__(module), class_name)

            data_placeholder = tf.placeholder(tf.float32, tensor_shape_list, name=input_node)
            net = KaffeNet({input_node: data_placeholder})

            # load weights stored in numpy format
            net.load(data_output_path, sess)

            print_stderr('Saving checkpoint...')
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_path)

            print_stderr('Saving graph definition as protobuf...')
            tf.train.write_graph(sess.graph.as_graph_def(), graph_folder, graph_name, False)

            input_graph_path = standalone_output_path
            input_saver_def_path = ""
            input_binary = True
            input_checkpoint_path = checkpoint_path
            output_node_names = output_node
            restore_op_name = 'save/restore_all'
            filename_tensor_name = 'save/Const:0'
            output_graph_path = standalone_output_path
            clear_devices = True

            print_stderr('Saving standalone model...')
            freeze_graph(input_graph_path, input_saver_def_path,
                         input_binary, input_checkpoint_path,
                         output_node_names, restore_op_name,
                         filename_tensor_name, output_graph_path,
                         clear_devices, '')

            shutil.rmtree(temp_folder)

        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('--standalone-output-path', help='Save generated standalone tensorflow model to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.standalone_output_path, args.phase)


if __name__ == '__main__':
    main()
