# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors, the HuggingFace Datasets and SuperGLUE Authors.
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

# Lint as: python3
"""The M3CoT benchmark."""


import json
import os

import datasets

_DESCRIPTION = """\
Multi-modal Chain-of-Thought (MCoT) requires models to leverage knowledge from both textual and visual modalities for step-by-step reasoning, which gains increasing attention. 
Nevertheless, the current MCoT benchmark still faces some challenges: (1) **absence of visual modal reasoning**, (2) **single-step visual modal reasoning**, and (3) **Domain missing**, thereby hindering the development of MCoT.	 
Motivated by this, we introduce a novel benchmark (M<sup>3</sup>CoT) to address the above challenges, advancing the multi-domain, multi-step, and multi-modal CoT.
Additionally, we conduct a thorough evaluation involving abundant MCoT approaches on Vision Large Language Models (VLLMs). 
In addition, we highlight that the current VLLMs still struggle to correctly reason in M<sup>3</sup>CoT and there remains a large gap between existing VLLMs and human performance in M<sup>3</sup>CoT, despite their superior results on previous MCoT benchmarks. 
To our knowledge, we take the first meaningful step toward the multi-domain, multi-step, and multi-modal scenario in MCoT.
We hope that M<sup>3</sup>CoT can serve as a valuable
resource, providing a pioneering foundation in multi-domain, multi-step, multi-modal chain-of-thought research.
"""

_CITATION = """\
@inproceedings{chen-etal-2024-m3cot,
    title = "M$^3$CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought",
    author = "Chen, Qiguang  and
      Qin, Libo  and
      Zhang, Jin  and
      Chen, Zhi  and
      Xu, Xiao  and
      Che, Wanxiang",
    booktitle = "Proc. of ACL",
    year = "2024",
}
"""

_HOME_PAGE_URL = "xxx"

_LICENSE = "CC BY 4.0"

_DATA_DIR = "data"

def read_jsonl(load_path):
    if not os.path.exists(load_path):
        print("Missing PATH: ", load_path)
        return []
    with open(load_path, "r", encoding="utf8") as f:
        res_list = []
        for i, line in enumerate(f):
            try:
                res_list.append(json.loads(line.strip()))
            except:
                print("Error in line :", i)
    return res_list

class M3CoTConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, **kwargs):
        """BuilderConfig for SuperGLUE.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict.
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 0.0.2: Initial version.
        super(M3CoTConfig, self).__init__(version=datasets.Version("0.0.3"), **kwargs)
        self.data_url = ""


class M3CoT(datasets.GeneratorBasedBuilder):
    """The SuperGLUE benchmark."""

    BUILDER_CONFIGS = [
        M3CoTConfig()
    ]

    def _info(self):
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
            "id": datasets.Value("string"),
            "category": datasets.Value("string"),
            "image_id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "choices": datasets.features.Sequence(datasets.Value("string")),
            "context": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "rationale": datasets.Value("string"),
            "split": datasets.Value("string"),
            "image": datasets.Image(),
            "domain": datasets.Value("string"),
            "topic": datasets.Value("string")
        }),
            homepage=_HOME_PAGE_URL,
            citation=_CITATION,
            license=_LICENSE
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_list": os.path.join(_DATA_DIR, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_list": os.path.join(_DATA_DIR, "dev.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_list": os.path.join(_DATA_DIR, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_list, split):
        for data in read_jsonl(data_list):
            yield data["id"], data
    
    
