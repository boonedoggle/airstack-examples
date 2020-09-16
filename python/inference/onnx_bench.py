#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility script to benchmark the data rate that a neural network will support.
"""

import numpy as np
import time
import cusignal
import onnxruntime as rt

# Default inference settings.
ONNX_FILE_NAME = 'pytorch/avg_pow_net.onnx'
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
BATCH_SIZE = 128  # Must be less than or equal to max_batch_size when creating plan file
NUM_BATCHES = 128  # Number of batches to run. Set to float('Inf') to run continuously
INPUT_DTYPE = np.float32


def plan_bench(onnx_file_name=ONNX_FILE_NAME, cplx_samples=CPLX_SAMPLES_PER_INFER,
         batch_size=BATCH_SIZE, num_batches=NUM_BATCHES, input_dtype=INPUT_DTYPE):

    # Use cuSignal to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network. ONNXRUNTIME does not currently support
    # complex valued input, so we treat the data stream from the receiver as interleaved
    # float32 data types.
    input_len = 2 * cplx_samples
    buff = cusignal.get_shared_mem((batch_size, input_len), dtype=input_dtype)

    # Setup ONNX model for inference and get the name of the input node
    dnn = rt.InferenceSession(onnx_file_name)
    input_name = dnn.get_inputs()[0].name

    # Populate input buffer with test data
    buff[:] = np.random.randn(batch_size, input_len).astype(input_dtype)

    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(num_batches):
        # ONNXRUNTIME requires the reshape below
        _ = dnn.run(None, {input_name: buff})[0]
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = cplx_samples * batch_size * num_batches

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * buff.itemsize * 8 / 1e3
    print('Result:')
    print('  ONNX File         : {}'.format(onnx_file_name))
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))


if __name__ == '__main__':
    plan_bench()
