## CTR 快速入门

本项目主要使用 DeepCTR 进行CTR模型训练。

开源本项目，方便初学者快速进行CTR模型训练。

## 【项目说明】

- 配置化运行，无需修改代码。
- 仅需提供训练数据，并通过简单的配置，即可实现CTR模型的训练。
- 支持模型 DeepFM、xDeepFM、DCNMix、DeepFEFM、DIFM 等。
- 最优模型自动保存，输出详细评估结果。
- 项目依赖见 requirements.txt
 
## 【运行指令及参数】

使用了两份数据集进行演示，详细执行指令如下（查看代码了解更多的执行参数）：

#### 1、movie

    python train.py --model "DeepFM" --topic "movie" \
        --params_file "./params/movie_params.json" \
        --train_data "./train_data/movie_sample.csv"
        
#### 2、criteo

    python train.py --model "xDeepFM" --topic "criteo" \
        --params_file "./params/criteo_params.json" \
        --train_data "./train_data/criteo_sample.txt" \
        
