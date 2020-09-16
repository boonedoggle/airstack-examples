#!/usr/bin/env python3
#
# Copyright 2020, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import onnxruntime as rt
import cusignal
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW

# Default inference settings.
ONNX_FILE_NAME = 'pytorch/avg_pow_net.onnx'
CPLX_SAMPLES_PER_INFER = 2048  # This should be half input_len from the neural network
BATCH_SIZE = 128

# Top-level SDR settings.
SAMPLE_RATE = 125e6  # AIR-T sample rate
CENTER_FREQ = 2400e6  # AIR-T Receiver center frequency
CHANNEL = 0  # AIR-T receiver channel

# Detector settings
THRESHOLD_DB = -55.0


def main():
    # Use cuSignal to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network. ONNXRUNTIME does not currently support
    # complex valued input, so we treat the data stream from the receiver as interleaved
    # float32 data types.
    input_len = 2 * CPLX_SAMPLES_PER_INFER
    buff = cusignal.get_shared_mem((BATCH_SIZE, input_len), dtype=np.float32)
    buff_len = input_len * BATCH_SIZE

    # Setup ONNX model for inference and get the name of the input node
    dnn = rt.InferenceSession(ONNX_FILE_NAME)
    input_name = dnn.get_inputs()[0].name

    # Create, configure, and activate AIR-T's radio hardware
    sdr = Device()
    sdr.setSampleRate(SOAPY_SDR_RX, CHANNEL, SAMPLE_RATE)
    sdr.setGainMode(SOAPY_SDR_RX, CHANNEL, True)
    sdr.setFrequency(SOAPY_SDR_RX, CHANNEL, CENTER_FREQ)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [CHANNEL])
    sdr.activateStream(rx_stream)
    thresh = 10.0 ** (THRESHOLD_DB / 10.0)  # Convert to linear units

    # Start receiving signals and performing inference
    print('Receiving Data')
    while True:
        try:
            sdr.readStream(rx_stream, [buff], buff_len)  # Receive samples to buffer
            result = dnn.run(None, {input_name: buff})  # Run buffer through DNN
            if np.any(result[0] > thresh):
                p = 10 * np.log10(result[0].max())
                print('Detection: {:0.2f} dB > {:0.1f} dB'.format(p, THRESHOLD_DB))
        except KeyboardInterrupt:
            break
    sdr.closeStream(rx_stream)


if __name__ == '__main__':
    main()
