# 测精度需要加 --validation, 测TPUT不需要加
# 如果需要测试FP32，加入 --model_type fp32
python3 fp8_test.py \
--batch_size 1  \
--outline  \
--validation  \
--onnx_path ./onnx_model/test_fp8_25_28_32_64_39_89_73.onnx