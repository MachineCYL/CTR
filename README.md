## CTR 快速入门

本项目主要使用 DeepCTR 进行CTR模型训练。

开源本项目，方便初学者快速进行CTR模型训练。

## 【项目说明】

- 配置化运行，无需修改代码。
- 仅需提供训练数据，并通过简单的配置，即可实现CTR模型的训练。
- 支持模型 DeepFM、xDeepFM、DCNMix、DeepFEFM、DIFM 等。
- 最优模型自动保存，输出详细评估结果。
- 支持导出 SavedModel 格式模型，方便 TF Serving 部署。
- 项目依赖见 requirements.txt
 
## 【运行指令及参数】

### 1. 模型训练
使用了两份数据集进行演示，详细执行指令如下（查看代码了解更多的执行参数）：

#### 1.1 movie
```
python train.py --model "DeepFM" --topic "movie" \
    --params_file "./params/movie_params.json" \
    --train_data "./train_data/movie_sample.csv"
```

#### 1.2 criteo

```
python train.py --model "xDeepFM" --topic "criteo" \
    --params_file "./params/criteo_params.json" \
    --train_data "./train_data/criteo_sample.txt"
```
     
### 2. docker 部署指令   

默认导出的 SavedModel 格式模型会保存在 ./export_model/ 目录下面，TF Serving 部署指令如下：
```
docker pull tensorflow/serving:2.0.0

docker run -d \
  -p 8500:8500 \
  -p 8501:8501 \
  --name "tf_serving" \
  -v "/data/MachineCYL/CTR/export_model/movie-DeepFM/:/models/model/" \
  tensorflow/serving:2.0.0
  
```

服务请求示例如下：
```
curl -d '{"inputs": {
    "userId": [[1]],
    "movieId": [[1]],
    "releaseYear": [[1]],
    "userAvgRating": [[1]],
    "movieAvgRating": [[1]],
    "userRatingCount": [[1]],
    "userAvgReleaseYear": [[1]],
    "userReleaseYearStddev": [[1]],
    "userRatingStddev": [[1]],
    "userRatedMovie1": [[1]],
    "userRatedMovie2": [[1]],
    "userRatedMovie3": [[1]],
    "userRatedMovie4": [[1]],
    "userRatedMovie5": [[1]],
    "userGenre1": [[1]],
    "userGenre2": [[1]],
    "userGenre3": [[1]],
    "userGenre4": [[1]],
    "userGenre5": [[1]],
    "movieGenre1": [[1]],
    "movieGenre2": [[1]],
    "movieGenre3": [[1]],
    "movieRatingCount": [[1]],
    "movieRatingStddev": [[1]]
    }
}' -X POST http://127.0.0.1:8501/v1/models/model:predict
```

请求返回的结果如下：
```
{
    "outputs": [
        [
            0.936053038
        ]
    ]
}
```