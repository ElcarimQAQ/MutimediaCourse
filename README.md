# 文献检索期末作业实验报告

**杨丽冰-51255901139**



本次期末作业主要参考阿里巴巴达摩院的开源项目https://github.com/OFA-Sys/Chinese-CLIP，实现了一个基于Chinese-CLIP的以文搜图项目，Chinese-CLIP是CLIP模型的中文版，使用大规模中文数据进行训练（~2亿图文对），可以实现中文的以文搜图。

数据的来源或者形式是多种多样的，每一种都可以称为一种模态。例如：图像、视频、声音、文字等都是不同模态的数据。现有的检索技术可以分为单模态检索和多模态检索。单模态检索要求查询与被查询的数据属于同一种模态类型，比如文本; 多模态检索则融合了不同模态进行检索，比如以图搜文，以文搜图。

本次作业使用 CLIP 模型实现一个自然语言图像检索的任务。接下来，会首先介绍CLIP模型原理，然后是CLIP具体代码实现，最终以“以文搜图”为例介绍如何使用CLIP模型完成下游任务。

# CLIP模型解读

CLIP（[Contrastive Language–Image Pre-training](https://arxiv.org/abs/2103.00020) ）是由OpenAI开源的基于对比学习的大规模图文预训练模型。 对比学习着重于学习同类实例之间的共同特征，区分非同类实例之间的不同之处。CILP可以缓解有监督深度学习模型的三个问题：

- 标注数据获取成本高（Costly datasets）：之前大部分模型用的数据集都是人工标注，而CLIP通过从互联网上搜索图像-文本对的方式和利用各种公开数据集构建了一个包含4亿对图像-文本的新的数据集。
- 适用范围窄（Narrow）：根据有标注数据集训练的话输出是有限的，即输出限制在了标签范围，而CLIP在常见图像上就不受限制。
- 实用性不佳（Poor real-world performance）：CLIP可以不用在数据集上训练，直接在基准上评估，而不是仅优化其在基准性上的性能，因此实用范围更广。

更多详情请参考[CLIP官网](https://openai.com/blog/clip/)。

CLIP模型架构图如下：

![img](http://picbed.elcarimqaq.top/img/b7f7e93399464f39b874716448a86124c36c9726b3384aaaa7fde084f8e9453d)

从上图可知，CLIP对比预训练模块模型架构是一个图文匹配的双流分支，主要包括图像编码器和文本编码器。CLIP联合训练图像编码器和文本编码器来预测图像-文本是否匹配。在测试时，学习得到的文本编码器通过嵌入目标数据集类别的名称或描述来合成零样本线性分类器。

图像编码器可以是resnet50或vision transformer（ViT）等，文本编码器可以是transformer。Create dataset classifier from label text和Use for zero-shot prediction是测试过程。

在传统的分类模型中，为了解决多分类问题（例如三个类别：猫、狗和猪），就需要提供大量的猫、狗和猪的图片用以模型训练，然后给定一张新的图片，就能判定属于猫、狗或猪的其中哪一类。但是对于之前训练图片未出现的类别（例如牛），这个模型便无法将牛识别出来，而ZSL(zero-shot learning)就是为了解决这种问题。在ZSL中，某一类别在训练样本中未出现，但是我们知道这个类别的特征，然后通过语料知识库，便可以将这个类别识别出来。

CLIP的训练数据是从网络社交媒体上搜集的4亿个图像文本对。在训练阶段，对于一个batch内的数据，首先通过图像编码器和文本编码器的到图像和文本的特征，然后将所有的图像和文本特征分别计算内积，这样可以得到一个矩阵。从图像的角度看，行方向就是一个分类器，从文本角度看，列方向也是一个分类器。内积最大对应的标签就是分类结果。

已知一个batch中的文本和图像的匹配关系，目标函数就是最大化同一对图像和文本特征的内积，也就是矩阵对角线上的元素(上图蓝色背景)，而最小化与不相关特征的内积。

下图是一个例子：

![img](http://picbed.elcarimqaq.top/img/2583ee0682164b688ec7bb47a3fc00500c491352e9864f0d8b46d4166af00f9f)

在图文矩阵中，概率值越大说明图文匹配度越高，对角线为配对的图文，因此概率值最高。

用通俗的话讲，这个模型包含三步，1）文本和图片两路分别进行特征编码；2）将文本、图片的特征从各自的单模态特征空间投射到一个多模态的特征空间；3）在多模态的特征空间中，原本成对的图像文本（正样本）的特征向量之间的距离应该越近越好，互相不成对的图像文本（负样本）的特征向量之间的距离应该越远越好。

## 举个例子

用一张皮卡丘，去匹配"杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "杨丽冰的作业"等文本，可以看到结果最高的是"皮卡丘"，为9.448e-01，和"杨丽冰的作业"则没有关系。

根据关联程度的高低，就可以实现以文推图。

![image-20230102203922995](http://picbed.elcarimqaq.top/img/image-20230102203922995.png)

![image-20230102203716199](http://picbed.elcarimqaq.top/img/image-20230102203716199.png)



# 方案设计

以文搜图顾名思义，所需要的基础组件涉及**文，搜，图**三部分。

**图**构成的是底库的部分，主要包含三套东西，一个是原始图片库，另一个是与原始图片相对应的向量库。中间是负责对原始图片进行语义向量化编码的模型推理服务。

**文**对应的是请求侧的内容。请求侧主要需要的是对文本进行语义向量化编码的模型推理服务。

**搜**对应的是连接请求、向量库、图片库的搜索过程。一条请求的文本，经过模型编码后，我们可以获得与请求对应的向量。我们拿这个向量到向量数据库进行近似搜索，获得 topK 个最近的图片向量。最后，通过向量-图片的ID关联，从图片库获得对应的原始图片，组织请求的返回结果。

具体流程如下：

1. 构建图像文本多模态模型；
2. 利用图像和文本联合训练多模态模型，该模型可以对文本和图像进行编码；
3. 将待搜索的图像集合进行编码制作成图像数据库以备检索；
4. 输入检索文本，CLIP模型计算图像与文本之间的关联程度，从图像库中检索图像。

# 详细实现及实验结果

## 准备图片库的数据

本次使用的数据集是[MUGE](https://tianchi.aliyun.com/muge).

MUGE是业界首个大规模中文多模态评测基准，由达摩院联合浙江大学、阿里云天池平台联合发布，中国计算机学会计算机视觉专委会（CCF-CV专委）协助推出。目前包括：多模态理解与生成任务在内的多模态评测基准，其中包括图像描述、图文检索以及基于文本的图像生成。

数据集的组织如下：

![image-20230102194358613](http://picbed.elcarimqaq.top/img/image-20230102194358613.png)

其中，为保证文件处理效率，图片不是以大量的小文件方式存放，而是将训练/验证/测试图片以base64形式分别存放在`×××_imgs.tsv`文件中。文件每行表示一张图片，包含图片id（int型）与图片base64，以tab隔开。

```python
from PIL import Image
from io import BytesIO
import base64

img = Image.open(file_name) # 访问图片路径
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
base64_str = base64_str.decode("utf-8") # str
```

![image-20230102212334520](http://picbed.elcarimqaq.top/img/image-20230102212334520.png)

文本信息及图文对匹配关系则保存在`xxx_texts.jsonl`文件，文件每行是一行json。

![image-20230102211150444](http://picbed.elcarimqaq.top/img/image-20230102211150444.png)

最后，还需要将tsv和jsonl文件一起序列化，转换为内存索引的LMDB数据库文件，方便读取。

## 预训练模型准备

图像特征部分的提取可以选用ModifiedResNet或VisualTransformer。

ModifiedResNet中使用的核心优化包括：

- Zhang,2019中的[anti-alias](https://arxiv.org/pdf/1904.11486.pdf)改进方法；
- 使用[weight norm](https://arxiv.org/abs/1602.07868)替代了[batch norm](https://arxiv.org/abs/1502.03167)；
- [GELU](https://arxiv.org/abs/1606.08415)激活函数。

VisualTransformer结构图下：

![img](http://picbed.elcarimqaq.top/img/3231e33e6d684df1be7df95033d61d5c29d5e17e425d41b2940feb8000fbd098)

文本特征提取选用Transformer。

Chinese-CLIP目前开源5个不同规模，其模型信息和下载方式见下表：

![image-20230102195731280](http://picbed.elcarimqaq.top/img/image-20230102195731280.png)

我选择使用了其中的B/16 作为本次作业的模型。模型放在data目录下的pretrained_weights 文件夹中。

![image-20230102200907784](http://picbed.elcarimqaq.top/img/image-20230102200907784.png)

## 图文特征提取

代码使用GPU单卡进行图文特征提取：

在代码中使用ViT模型提取图像特征，并且加载预训练模型，调用模型的encode_image函数即可得到图像编码特征。

```python
 print('Make inference for images...')
    args.image_feat_output_path = "{}.img_feat.jsonl".format(args.text_data.replace("_texts.jsonl", "_imgs"))
    write_cnt = 0
    with open(args.image_feat_output_path, "w") as fout:
        model.eval()
        dataloader = img_data.dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader):
                image_ids, images = batchimage-20230102203021192
                images = images.cuda(args.gpu, non_blocking=True)
                image_features = model(images, None)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                    write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))
```

同样，使用Transformer模型提取图像特征，并且加载预训练模型，调用模型的encode_text函数即可得到文本编码特征。

```python
 write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.eval()
            dataloader = text_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    text_ids, texts = batch
                    texts = texts.cuda(args.gpu, non_blocking=True)
                    text_features = model(None, texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} text features are stored in {}'.format(write_cnt, args.text_feat_output_path))
```

![image-20230102201324655](http://picbed.elcarimqaq.top/img/image-20230102201324655.png)

可以看到成功完成了特征提取，包括5008行文本特征以及29806 行图片特征。

将产生的图文特征保存起来，放在/home/lbyang/data/datasets/MUGE目录下。

图片特征保存在`xxx_imgs.img_feat.jsonl`下，格式为：

{"image_id": 1000002, "feature": [0.0198, ..., -0.017, 0.0248]}

![image-20230102202818173](http://picbed.elcarimqaq.top/img/image-20230102202818173.png)

文本特征则保存于`xxx_texts.txt_feat.jsonl`，格式如下：

{"text_id": 248816, "feature": [0.1314, ..., 0.0018, -0.0002]}

![image-20230102203021192](http://picbed.elcarimqaq.top/img/image-20230102203021192.png)

其中的image_id和text_id负责索引图片和文本。

## KNN检索

可以使用一个简单的KNN实现检索功能，检索出每一个text对应的前10相关的图片，返回的image_id就是对应的结果。

```python
print("Begin to compute top-{} predictions for texts...".format(args.top_k))
with open(args.output, "w") as fout:
    with open(args.text_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_id = obj['text_id']
            text_feat = obj['feature']
            score_tuples = []
            text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda() # [1, feature_dim]
            idx = 0
            while idx < len(image_ids):
                img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
                batch_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
                for image_id, score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], batch_scores.squeeze(0).tolist()):
                    score_tuples.append((image_id, score))
                    idx += args.eval_batch_size
                    top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                    fout.write("{}\n".format(json.dumps({"text_id": text_id, "image_ids": [entry[0] for entry in top_k_predictions]})))

                    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
                    print("Done!")
```

![image-20230102213425303](http://picbed.elcarimqaq.top/img/image-20230102213425303.png)

结果：

![image-20230102213445921](http://picbed.elcarimqaq.top/img/image-20230102213445921.png)

## 可视化界面搭建

已经实现了检索的功能，最后需要的就是可视化一下，把检索出来的图片展示出来。

于是使用Gradio 来搭建一个可视化的demo，Gradio是一个 轻量级的机器学习web  Demo 构建工具。更多细节可以参考Gradio 的[官网](https://gradio.app/)。

这一部分代码在chinese_clip_applications目录下，主要代码：

![image-20230102222044872](http://picbed.elcarimqaq.top/img/image-20230102222044872.png)

```python
from functools import partial
import json
from multiprocessing.pool import ThreadPool as Pool
import gradio as gr
from utils import *


def text2image_gr():
    def clip_api(query_text='', return_n=8, model_name=clip_base, thumbnail="是"):
        data = {"num_images": int(return_n)}
        if query_text:
            data["inputs"] = [query_text]
        else:
            return None
        payload = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", clip_service_url_d[model_name], headers=headers, data=payload)

        ret_json = json.loads(response.text)
        pool = Pool()
        new_url2image = partial(url2img, thumbnail=thumbnail)
        ret_imgs = pool.map(new_url2image, ret_json['recalls'])
        pool.close()
        pool.join()
        return ret_imgs

    title = "<h1 align='center'>中文CLIP文到图搜索应用</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    text = gr.Textbox(value="戴口罩的人", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=50, step=1, value=8, label="返回图片数（可能被过滤部分）", elem_id=2)
                model = gr.components.Radio(label="模型选择", choices=[clip_base, clip_large, clip_large_336],
                                            value=clip_base, elem_id=3)
                thumbnail = gr.components.Radio(label="是否返回缩略图", choices=[yes, no],
                                                value=yes, elem_id=4)
                btn = gr.Button("搜索", )
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：").style(grid=4, height=200)
        inputs = [text, num, model, thumbnail]
        btn.click(fn=clip_api, inputs=inputs, outputs=out)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        demo.launch(
            enable_queue=True,
        )
```

运行Gradio，网页成功运行：

![image-20230102221017995](http://picbed.elcarimqaq.top/img/image-20230102221017995.png)

![image-20230102221215740](http://picbed.elcarimqaq.top/img/image-20230102221215740.png)

检索结果：

![image-20230102221400943](http://picbed.elcarimqaq.top/img/image-20230102221400943.png)

![image-20230102222208770](http://picbed.elcarimqaq.top/img/image-20230102222208770.png)

# 参考资料

1. https://github.com/openai/CLIP
2. 基于CLIP实现的以文搜图https://aistudio.baidu.com/aistudio/projectdetail/4458949?channelType=0&channel=0
3. 中文CLIP Tutorialhttps://modelscope.cn/docs/%E4%B8%AD%E6%96%87CLIP%20Tutorial
4. Gradio https://gradio.app/
5. https://zhuanlan.zhihu.com/p/537123858
6. 图像&文本的跨模态相似性比对检索 https://github.com/mymagicpower/AIAS/blob/main/7_engine_hub/image_text_search/README.md