[**中文**](./README.md)

以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。在LLM大行其道的背景下，几乎所有的NLP任务都转化为了基于提示的语言生成任务。然而，在中文医学NLP社区中，尚未有一个统一任务形式的评测基准。

----

[Text2DT](https://github.com/michael-wzhu/Text2DT_Baseline) | [中文医疗在线问诊数据集ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | [中文问诊大模型ChatMed-Consult](https://huggingface.co/michaelwzhu/ChatMed-Consult) | [中医药指令数据集ChatMed_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_TCM_Dataset) |  [中医药大模型ChatMed-TCM](https://huggingface.co/michaelwzhu/ChatMed-TCM) | [Candidate-Soups: 提升非自回归翻译模型的有效trick](https://github.com/boom-R123/Candidate_Soups)


## 更新

2023/07/18 添加了基于LlaMA的LoRA微调代码；并且使用vllm对模型推理加速(相比于huggingface的生成加速2.5倍左右)。

2023/07/02 开源PromptCBLUE的各个prompt模板；同时，对模板采用ChatGPT进行扩充，将会把提示模板扩展到500个左右。

2023/06/25 测试ChatGPT在四千tokens长度以内，采用In-context learning模式，完成PromptCBLUE评测表现！

2023/05/12 更新ChatGLM-6B + Lora方法在dev集表现(在相同训练步数，相同最大长度限制下，比p-tuning表现较好)。同时添加baseline代码的[requirements.txt](./requirements.txt)

2023/5/09 上传了基于ChatGLM-B + Lora方法的参数高效微调代码，作为baseline,代码见[ChatGLM+lora code](./src/ft_chatglm_lora)

2023/5/05 上传了基于ChatGLM + P-tuning的参数高效微调代码，作为baseline,代码见[ChatGLM+ptuning code](./src/ft_chatglm_ptuning)。快速上手，请参看[ChatGLM+ptuning方法的README](./src/README.md)。

2023/4/25 PromptCBLUE(v0.1)上线了，将持续更新！ 🎉🎉🎉




## 数据集详情

### PromptCBLUE总体统计


| PromptCBLUE      | -      |
|-------------|--------|
| 版本号         | v0.2   |
| prompt 模板数量 | 94     |
| 训练集         | 68900  |
| 验证集         | 10360  |
| 测试集A        | 10320 |
| 测试集B        | 10320  |


### CBLUE任务改造

我们采用94个指令微调模板，对CBLUE基准中的各个任务进行。经过改造后，医疗文本NLP数据集都将转化为如下格式。input字段是模型的输入，target字段是模型的输出，type是原任务类型(不作为模型输入)，answer_choices字段是选项，只有分类、术语标准化、推理类任务上该字段才会有意义。

```bash
{
  "input": str,
  "target": str,
  "type": str,
  "answer_choices": str,
  "sample_id": str,
}
```

为了将CBLUE中的各种不同任务适配为符合LLM的输入输出格式，我们对CBLUE各个数据集进行了相应的改造。详见[CBLUE任务改造](https://github.com/michael-wzhu/PromptCBLUE/blob/main/src/data/CBLUE任务改造说明与举例.md)


### 评价指标

本评测任务只有一个测试集，但是其包含多个任务的测试样本，我们采用在各个任务上分别计分的方式进行评测。各个任务上的评测指标如下：

- 对于CMeEE-V2和IMCS-V2-NER任务，采用基于实体实例层面的严格的(strict)，micro的Precision, Recall, F1分数。这里的实体实例包含mention（即实体名称的所有组成字符）和类型这两个组成字段。这里"严格的"指模型必须在指定的样本sample_id上，完全正确预测出ground truth中的实体实例的mention和类型，才能算是成功预测出这个实体实例，则true positive (TP) 加1。而如果模型预测的实体实例不在ground truth中，则false positive (FP)加1。如果ground truth中的实体实例未被模型预测到，则false negative(FN)加1。最终根据整个测试集上的TP，FP，FN计算Precision, Recall, F1分数。
- 对于CMeIE任务，采用基于三元组实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的三元组实例包含头实体mention, 尾实体mention，和关系类型字段。
- 对于CHIP-CDEE任务，采用基于临床事件实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的临床事件实例包含主体词，发生状态，描述词和解剖部位字段。
- 对于IMCS-V2-SR和CHIP-MDCFNPC任务，采用基于临床发现或者症状实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的临床发现或者症状实例包含mention和阴阳性判断标签字段。
- 对CHIP-CDN任务，采用基于ICD-10标准词实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的ICD-10标准词实例包含mention和阴阳性判断标签字段。
- 对CHIP-STS， KUAKE-QQR, KUAKE-IR，KUAKE-QTR任务，我们采用Micro的precision, recall, F1分数作为评估指标。对CHIP-CTC，IMCS-V2-DAC，KUAKE-QIC, 采用Macro的precision, recall, F1分数作为评估指标。
- 对于MedDG和IMCS-V2-MRG数据集，我们采用Rouge-1，Rouge-2，Rouge-L分数作为评估指标。为避免分词影响，计算rouge分数前，会将句子中的汉字拆开，用空格分隔。IMCS-V2-MRG任务中，需要将模型生成的诊断报告拆分为主诉, 现病史, 辅助检查, 既往史, 诊断, 建议这六个章节，分别计算rouge得分后取平均分。

上述任务中，F1(micro/macro)或者Rouge-L将作为每个任务的主要指标。

**总体打分**的计算：我们将对每个任务上的F1(micro/macro)或者Rouge-L分数进行平均，得到总体分数，作为榜单排名的及评奖的依据。


## baseline模型

我们基于[ChatGLM-6B模型](https://github.com/THUDM/ChatGLM-6B)构建PromptCBLUE的baseline模型。代码和运行操作详见[PromptCBLUE-baseline模型](https://github.com/michael-wzhu/PromptCBLUE/blob/main/src/)。我们考虑以下baseline方法:

- 基于[ChatGLM-6B模型](https://github.com/THUDM/ChatGLM-6B)模型，在PromptCBLUE的训练集(68900个样本)上采用p-tuning的参数高效微调方法进行微调(bsz=8,gradient accumulation=8, steps=3000)；
- 基于ChatGLM-6B模型，采用Lora的参数高效微调方法进行微调(bsz=4,lora_rank=8, lora作用在query_key_value,dense,dense_h_to_4h,dense_4h_to_h模块，gradient_accumulation=16, steps=3000)；
- 基于ChatGLM-6B + AdaLora的微调（实验设置与上述LoRA方法一致，steps=5100）；结果来自[boom-R123](https://github.com/boom-R123)

另外，大家都知道ChatGPT作为强大的大模型，其in-context learning(ICL)能力非常强，所以我们也评测了ChatGPT（截止2023年6月25日）在PromptCBLUE的dev集表现。在预测每个dev样本时，采用训练样本中的同任务下固定的3-20个样例（根据样例长度，尽量塞满ChatGPT的最大允许长度）作为demonstrations，供ChatGPT学习并相应的给出dev样本的预测结果。


在dev集上实验结果如下：

| task         | metric    | ChatGLM-6B + ptuning | ChatGLM-6B + LoRA | ChatGLM-6B + AdaLoRA | ChatGPT + ICL |
|--------------|-----------|----------------------|-------------------|----------------------|---------------|
| CMeEE-V2     | micro-F1  | 0.6359               | 0.6725            | 0.6634               | 0.4698        |
| CMeIE        | micro-F1  | 0.3765               | 0.4555            | 0.4290               | 0.3058        |
| CHIP-CDN     | micro-F1  | 0.7805               | 0.8461            | 0.8465               | 0.6069        |
| CHIP-CDEE    | micro-F1  | 0.4914               | 0.5456            | 0.5131               | 0.2838        |
| CHIP-STS     | micro-F1  | 0.7696               | 0.8081            | 0.7618               | 0.7108        |
| CHIP-CTC     | macro-F1  | 0.8046               | 0.8086            | 0.7398               | 0.5253        |
| KUAKE-IR     | micro-F1  | 0.6154               | 0.6835            | 0.7657               | 0.5183        |
| KUAKE-QIC    | macro-F1  | 0.8113               | 0.7390            | 0.8400               | 0.4851        |
| KUAKE-QQR    | micro-F1  | 0.5537               | 0.6348            | 0.6738               | 0.3040        |
| KUAKE-QTR    | micro-F1  | 0.4701               | 0.5428            | 0.5076               | 0.2318        |
| CHIP-MDCFNPC | micro-F1  | 0.6865               | 0.7366            | 0.7531               | 0.5854        |
| IMCS-V2-DAC  | macro-F1  | 0.7147               | 0.7639            | 0.7168               | 0.3455        |
| IMCS-V2-NER  | micro-F1  | 0.8508               | 0.8709            | 0.8779               | 0.5684        |
| IMCS-V2-SR   | micro-F1  | 0.6168               | 0.6330            | 0.6458               | 0.3305        |
| IMCS-V2-MRG  | Rouge-L   | 0.4707               | 0.4663            | 0.4811               | 0.3253        |
| MedDG        | Rouge-L   | 0.1035               | 0.1117            | 0.1298               | 0.1361        |
| Overall      | avg score | 0.6095               | 0.6448            | 0.6466               | 0.4208        |


我们将会持续不断地输出各种不同的baseline模型与代码给大家，希望大家持续关注本repo：
- ⏳ TODO: 更多微调方法(如Parallel-Adapter, BitFit等)；
- ⏳ TODO: 针对每个任务采用高效微调的方法，在预测时对不同任务调用不同的高效微调模块；


## References

- [PromptCBLUE基准论文: PromptCBLUE: A Chinese Prompt Tuning Benchmark for the Medical Domain](https://arxiv.org/abs/2310.14151)
- [CHIP-PromptCBLUE评测任务综述： Overview of the PromptCBLUE Shared Task in CHIP2023](https://arxiv.org/abs/2312.17522)
- [ChatGLM-6b模型](https://github.com/THUDM/ChatGLM-6B)
- [CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark](https://aclanthology.org/2022.acl-long.544) (Zhang et al., ACL 2022)
- [Text2DT论文: Text2MDT: Extracting Medical Decision Trees from Medical Texts](https://arxiv.org/pdf/2401.02034.pdf)
- Zan, Hongying, Wenxin Li, Kunli Zhang, Yajuan Ye, Baobao Chang and Zhifang Sui. “Building a Pediatric Medical Corpus: Word Segmentation and Named Entity Annotation.” Chinese Lexical Semantics (2020).
- Guan, Tongfeng, Hongying Zan, Xiabing Zhou, Hongfei Xu and Kunli Zhang. “CMeIE: Construction and Evaluation of Chinese Medical Information Extraction Dataset.” Natural Language Processing and Chinese Computing (2020).
- Zong, Hui, Jinxuan Yang, Zeyu Zhang, Zuofeng Li and Xiaoyan Zhang. “Semantic categorization of Chinese eligibility criteria in clinical trials using machine learning methods.” BMC Medical Informatics and Decision Making 21 (2021): n. pag.
- Liu, Wenge, Jianheng Tang, Jinghui Qin, Lin Xu, Zhuguo Li and Xiaodan Liang. “MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System.” ArXiv abs/2010.07497 (2020): n. pag.
- Chen, W., Zhiwei Li, Hongyi Fang, Qian-Qian Yao, Cheng Zhong, Jianye Hao, Qi Zhang, Xuanjing Huang, Jianjun Peng and Zhongyu Wei. “A benchmark for automatic medical consultation system: frameworks, tasks and datasets.” Bioinformatics 39 (2022): n. pag.
- Chen, W., Cheng Zhong, Jiajie Peng and Zhongyu Wei. “DxFormer: a decoupled automatic diagnostic system based on decoder–encoder transformer with dense symptom representations.” Bioinformatics 39 (2022): n. pag.
- Wei, Zhongyu, Qianlong Liu, Baolin Peng, Huaixiao Tou, Ting Chen, Xuanjing Huang, Kam-Fai Wong and Xiangying Dai. “Task-oriented Dialogue System for Automatic Diagnosis.” Annual Meeting of the Association for Computational Linguistics (2018).
- Lin, Xinzhu, Xiahui He, Qin Chen, Huaixiao Tou, Zhongyu Wei and Ting Chen. “Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph.” Conference on Empirical Methods in Natural Language Processing (2019).
- Liao, Kangenbei, Qianlong Liu, Zhongyu Wei, Baolin Peng, Qin Chen, Weijian Sun and Xuanjing Huang. “Task-oriented Dialogue System for Automatic Disease Diagnosis via Hierarchical Reinforcement Learning.” ArXiv abs/2004.14254 (2020): n. pag.
- Long, Dingkun, Qiong Gao, Kuan-sheng Zou, Guangwei Xu, Pengjun Xie, Rui Guo, Jianfeng Xu, Guanjun Jiang, Luxi Xing and P. Yang. “Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval.” Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (2022): n. pag.
- 熊英,陈漠沙,陈清财,汤步洲.CHIP-2021评测任务1概述:医学对话临床发现阴阳性判别任务[J].医学信息学杂志,2023,44(3):46~51
- 骆迅,倪渊,汤步洲,雷健波. 基于竞赛视角探讨文本语义匹配技术在中文医学文本领域中的应用 [J]. 中国数字医学. 2021 (11)
- 李文锋，朱威，王晓玲，等.Text2DT:面向临床针对文本的决策规则抽取技术[J].医学信息学杂志，2022，43（12）：16-22.

