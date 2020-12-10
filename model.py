import os
import numpy as np

from paddle.fluid.core import AnalysisConfig, create_paddle_predictor

__all__ = ['Model']

class Model():
    # 初始化函数
    def __init__(self, modelpath, use_gpu=False, use_mkldnn=True, combined=True):
        # 加载模型预测器
        self.predictor = self.load_model(modelpath, use_gpu, use_mkldnn, combined)

        # 获取模型的输入输出
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.input_tensor = self.predictor.get_input_tensor(self.input_names[0])
        self.output_tensor = self.predictor.get_output_tensor(self.output_names[0])

    # 模型加载函数
    def load_model(self, modelpath, use_gpu, use_mkldnn, combined):
        # 对运行位置进行配置
        if use_gpu:
            try:
                int(os.environ.get('CUDA_VISIBLE_DEVICES'))
            except Exception:
                print('Error! Unable to use GPU. Please set the environment variables "CUDA_VISIBLE_DEVICES=GPU_id" to use GPU.')
                use_gpu = False
                
        # 加载模型参数
        if combined:
            model = os.path.join(modelpath, "__model__")
            params = os.path.join(modelpath, "__params__")
            config = AnalysisConfig(model, params)
        else:
            config = AnalysisConfig(modelpath)

        # 设置参数
        if use_gpu:   
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            if use_mkldnn:
                config.enable_mkldnn()
        config.disable_glog_info()
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)
        
        # 通过参数加载模型预测器
        predictor = create_paddle_predictor(config)
        
        # 返回预测器
        return predictor

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []

        # 遍历输入数据进行预测
        for input_data in input_datas:
            inputs = input_data.copy()
            self.input_tensor.copy_from_cpu(inputs)
            self.predictor.zero_copy_run()
            output = self.output_tensor.copy_to_cpu()
            outputs.append(output)
        
        # 预测结果合并
        outputs = np.concatenate(outputs, 0)

        # 返回预测结果
        return outputs