#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys

sys.path.append("./")

from src.ft_chatglm_ptuning.tokenization_chatglm import ChatGLMTokenizer
from src.ft_chatglm_ptuning.configuration_chatglm import ChatGLMConfig
from src.ft_chatglm_ptuning.modeling_chatglm import ChatGLMForConditionalGeneration
from src.ft_chatglm_ptuning.trainer_seq2seq import Seq2SeqTrainer
from src.ft_chatglm_ptuning.arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


def main():

    # 参数解析
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # 解析 json 文件参数 或 命令行参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 日志配置：日志格式和处理范围
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 设置随机种子：保证实验结果的可重现
    set_seed(training_args.seed)

    # 加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(                                        # 加载并缓存数据集
        "json",                                                         # 根据指定格式解析文件
        data_files=data_files,                                          # 指定数据集文件的路径
        cache_dir=model_args.cache_dir,                                 # 指定缓存数据集的目录,以便将来重新使用时加快加载速度 - None
        # use_auth_token=True if model_args.use_auth_token else None,     # 用于需要认证的私有数据集或需要访问受限的数据集 - None
        token=True if model_args.use_auth_token else None,              # use_auth_token 将弃用, token 代替
    )
    print("chatGLM_pTuning raw_datasets: ", raw_datasets)

    # 加载预训练模型的参数
    # config = AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     trust_remote_code=True
    # )
    config = ChatGLMConfig.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    # 配置预训练模型的 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     trust_remote_code=True
    # )
    tokenizer = ChatGLMTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )

    # 加载指定模型的所有权重和配置
    # model = AutoModel.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     trust_remote_code=True
    # )
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,  # 加载 ChatGLMForConditionalGeneration 模型
        config=config,  # 使用提供的配置 config
    )
    # 如果有一个指定的 P-Tuning 检查点路径, 需要加载额外的状态字典
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder

        # 加载 pTuning 模型的状态字典
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))

        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            # 对于包含 transformer.prefix_encoder. 前缀的键，去掉该段，重新保存
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        # 将 new_prefix_state_dict 加载到模型的前缀编码器中 - 更新权重, 使其包含从检查点加载的权重
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    # 量化处理
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if model_args.pre_seq_len is not None:
        # P-tuning v2 微调
        # 将模型的权重转换为半精度浮点数（16位）。这可以减少内存占用，提高计算效率，特别是在使用 GPU 时。
        model = model.half()
        # 将前缀编码器的权重转换回单精度浮点数（32位）：前缀编码器的权重在半精度浮点数下可能会导致数值不稳定或精度损失，所以需要保持为单精度。
        model.transformer.prefix_encoder.float()
    else:
        # Finetune - 常规微调
        # 单精度浮点数（32位）: 深度学习模型训练的默认精度，可以提供更高的数值稳定性和精度。
        model = model.float()

    # 源文件前缀
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # 预处理数据集
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.  -  train.sh 中显式配置
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # 验证集/测试集的数据预处理方法实现
    def preprocess_function_eval(examples):
        # print("examples: ", examples)

        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            # 添加 target 数据
            if not examples[response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[response_column][i])

            # 添加 input 数据 - 对于是否存在历史对话，区别处理
            if examples[prompt_column][i]:
                query = examples[prompt_column][i]
                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:                           # 添加历史对话
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        # 对每个输入字段，添加 前缀 - 参数：source_prefix
        inputs = [prefix + inp for inp in inputs]
        # print("inputs: ", inputs)
        # tokenizer 分词 - 截断/填充
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        # 计算损失时是否忽略  填充 token
        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # 训练数据的预处理方法实现
    def preprocess_function_train(examples):
        # 最大序列长度
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            # 一条数据同时包含 prompt 和 target
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                # 根据是否存在历史数据 history_column， 构建 prompt
                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                # 对 prompt 添加前缀, 编码 prompt 和 answer, 不添加特殊标记
                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                # 截断
                if len(a_ids) > data_args.max_source_length - 1:        # -1: a_ids + gmask_id
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:        # -2: bos_id + b_ids + eos_id
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                # 构建标签：labels - 保留 target 序列
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]    # [-100]：特殊值，用于忽略这些位置的损失计算

                # 填充 pad
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                # print("input_ids: ", input_ids)
                # print("labels: ", labels)

                # 将 labels 中的 pad_token_id 替换为 -100, 以便在计算损失时忽略这些填充部分 - 默认 True
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    # 训练集数据预处理
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        # 若设置了最大训练样本数, 则截断超过部分 - 默认 None
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # 在主进程中首先进行训练集预处理
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,   # 默认 None
                remove_columns=column_names,
                load_from_cache_file=False,                     # 不从缓存文件加载，强制重新处理数据
                desc="Running tokenizer on train dataset",
            )
        # 打印训练数据集中的前两个示例
        print_dataset_example(train_dataset[0])
        print_dataset_example(train_dataset[1])

    # 验证集数据预处理
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])
        print_dataset_example(eval_dataset[1])

    # 测试集数据预处理
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])
        print_dataset_example(predict_dataset[1])

    # 创建数据整理器：Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False                           # False：数据集预处理时，已经实现了数据填充和对齐
    )

    # 计算评估预测结果的指标 - ROUGE 和 BLEU 分数
    def compute_metrics(eval_preds):
        preds, labels = eval_preds      # 预测结果 + 实际标签
        if isinstance(preds, tuple):
            preds = preds[0]

        # 解码 preds - 跳过特殊标签
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # 将标签中的 -100 替换为 pad_token_id, 再解码 - (-100 不能解码)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 存储各个评估指标的得分
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        # 遍历解码后的预测结果和标签
        for pred, label in zip(decoded_preds, decoded_labels):
            # jieba 分词
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            # 初始化 Rouge 对象
            rouge = Rouge()

            # 将分词结果 hypothesis 连接成字符串, 若为空, 则设置为 -
            hypothesis = ' '.join(hypothesis)
            if not hypothesis:
                hypothesis = "-"

            # 计算 rouge 分数
            """ rouge 结果示例 - 这里因为只计算一对假设和参考, 故 scores 列表中只有一项
            scores = [
                {
                    'rouge-1': {'r': 0.8, 'p': 0.75, 'f': 0.77},
                    'rouge-2': {'r': 0.6, 'p': 0.55, 'f': 0.57},
                    'rouge-l': {'r': 0.7, 'p': 0.65, 'f': 0.67}
                }
            ]
            """
            scores = rouge.get_scores(hypothesis, ' '.join(reference))
            result = scores[0]
            # 获取 F1 值 - *100 - round: 四舍五入 - 4 位有效数字
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            # 计算 bleu-4 分数
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))   # 计算列表元素的平均值
        return score_dict

    # 覆盖 Seq2SeqTrainer 的解码参数
    # generation_max_length: 控制生成任务中生成文本的最大长度
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    # generation_num_beams: 束搜索的束宽
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # 初始化 Seq2SeqTrainer 训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )

    # 设置和执行 模型训练
    if training_args.do_train:
        # resume_from_checkpoint：支持从指定检查点恢复训练
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint

        model.gradient_checkpointing_enable()       # 启用模型的梯度检查点：节省显存，在反向传播过程中逐层保存计算图，但会增加计算时间
        model.enable_input_require_grads()          # 启用模型的输入需要梯度：确保输入张量在反向传播过程中正确计算梯度 - 对于某些优化技术或模型调整是必要的

        # 训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        # 训练指标
        metrics = train_result.metrics

        # 确定最大训练样本数
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        # 将实际训练的样本数记录到指标中，确保不超过最大训练样本数
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)       # 记录训练期间的各项指标, 有助于跟踪训练过程中的性能和进展
        trainer.save_metrics("train", metrics)      # 保存训练期间的各项指标到文件或其他存储介质
        trainer.save_state()                        # 保存当前训练状态，包括模型权重、优化器状态和学习率调度器状态, 这使得训练可以在中断后从保存状态继续

    # 评估
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512,
                                   temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 预测
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            # max_tokens=512,
            max_new_tokens=data_args.max_target_length,
            do_sample=True,
            top_p=0.7,
            temperature=0.95,
            # repetition_penalty=1.1
        )
        metrics = predict_results.metrics
        print(metrics)
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                assert len(labels) == len(list_test_samples)

                output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.json")

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, (p, l) in enumerate(zip(predictions, labels)):
                        samp = list_test_samples[idx]
                        samp["target"] = p
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
