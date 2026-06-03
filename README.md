# TRIO

This repository is the official implementation of [TRIO]().

**[TRIO: Token Reduction via Inference-Objective Guidance for Efficient Vision-Language Models]()**
<br/>
[Haokui Zhang](https://scholar.google.com/citations?hl=en&user=m3gPwCoAAAAJ), 
[Congyang Ou](https://github.com/ocy1), 
[Dawei Yan](https://scholar.google.com/citations?user=U8KJSfcAAAAJ&hl=zh-CN&oi=ao), 
[Peng Wang](), 
[Qingsen Yan](https://scholar.google.com/citations?user=BSGy3foAAAAJ&hl=zh-CN&oi=ao), 
[Yu Zhang](), 
[Ying Li](), 
[Rong Xiao](), 
<br/>



## Overall
<p align="center">
  <a href="https://raw.githubusercontent.com/ocy1/TRIO/main/Images/overall_new.pdf">
    <img src="https://raw.githubusercontent.com/ocy1/TRIO/main/Images/overall_new.png" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>
   
  </em>
</p>

<blockquote>
  <p>
    </strong> We propose TRIO, a training-free method that selects vision tokens via gradient saliency and a feature-space NMS strategy, improving efficiency while preserving performance and maintaining compatibility with efficient attention operators.
  </p>
</blockquote>

## Example
<p align="center">
  <a href="https://raw.githubusercontent.com/ocy1/TRIO/main/Images/4%20methods%20new.pdf">
    <img src="https://raw.githubusercontent.com/ocy1/TRIO/main/Images/4%20methods.jpg"
         alt="Comparison of token selection strategies"
         width="92%" />
  </a>
  <br/>
  <em>
    Comparison of different token selection strategies.
  </em>
</p>

<blockquote>
  <p>
   </strong> Comparison of different token selection strategies.
    (a) Ours; (b) text-to-image attention based; (c) diversity-oriented selection;
    (d) CLS-token attention based.
  </p>
</blockquote>

## Performance Results

The following table presents a detailed comparison of our method (TRIO) against various state-of-the-art baselines across multiple benchmarks under different token retention budgets.

<table>
<thead>
<tr>
<th align="left">Methods</th>
<th align="left">Venue</th>
<th align="center">GQA</th>
<th align="center">MMB</th>
<th align="center">MMB-cn</th>
<th align="center">MME</th>
<th align="center">POPE</th>
<th align="center">SQA</th>
<th align="center">VQAv2</th>
<th align="center">TextVQA</th>
<th align="center">Average</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11" align="center"><strong>Upper Bound (576 Tokens)</strong></td>
</tr>
<tr>
<td align="left">Vanilla</td>
<td align="left">–</td>
<td align="center">61.9</td>
<td align="center">64.7</td>
<td align="center">58.1</td>
<td align="center">1862</td>
<td align="center">85.9</td>
<td align="center">69.5</td>
<td align="center">78.5</td>
<td align="center">58.2</td>
<td align="center">100%</td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/o VE (192 Tokens, 33.3%)</strong></td>
</tr>
<tr>
<td align="left">FastV</td>
<td align="left">ECCV’24</td>
<td align="center">52.7</td>
<td align="center">61.2</td>
<td align="center">57.0</td>
<td align="center">1612</td>
<td align="center">64.8</td>
<td align="center">67.3</td>
<td align="center">67.1</td>
<td align="center">52.5</td>
<td align="center">89.0%</td>
</tr>
<tr>
<td align="left">PDrop</td>
<td align="left">CVPR’25</td>
<td align="center">57.1</td>
<td align="center">63.2</td>
<td align="center">56.8</td>
<td align="center">1766</td>
<td align="center">82.3</td>
<td align="center">68.8</td>
<td align="center">75.1</td>
<td align="center">56.1</td>
<td align="center">96.2%</td>
</tr>
<tr>
<td align="left">SparseVLM</td>
<td align="left">ICML’25</td>
<td align="center">59.5</td>
<td align="center">64.1</td>
<td align="center">53.7</td>
<td align="center">1787</td>
<td align="center">85.3</td>
<td align="center">68.7</td>
<td align="center">75.6</td>
<td align="center">57.8</td>
<td align="center">97.2%</td>
</tr>
<tr>
<td align="left">DART</td>
<td align="left">EMNLP’25</td>
<td align="center">60.0</td>
<td align="center">63.6</td>
<td align="center">57.0</td>
<td align="center">1856</td>
<td align="center">82.8</td>
<td align="center">69.8</td>
<td align="center">76.7</td>
<td align="center">57.4</td>
<td align="center">98.3%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>61.0</strong></td>
<td align="center"><strong>64.4</strong></td>
<td align="center"><strong>57.6</strong></td>
<td align="center"><strong>1789</strong></td>
<td align="center"><strong>86.5</strong></td>
<td align="center"><strong>69.0</strong></td>
<td align="center"><strong>77.7</strong></td>
<td align="center"><strong>57.2</strong></td>
<td align="center"><strong>98.8%</strong></td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/ VE (192 Tokens, 33.3%)</strong></td>
</tr>
<tr>
<td align="left">VisionZip</td>
<td align="left">CVPR’25</td>
<td align="center">59.3</td>
<td align="center">63.0</td>
<td align="center">57.3</td>
<td align="center">1782</td>
<td align="center">85.3</td>
<td align="center">68.9</td>
<td align="center">76.8</td>
<td align="center">57.3</td>
<td align="center">97.8%</td>
</tr>
<tr>
<td align="left">HoloV</td>
<td align="left">NeurIPS’25</td>
<td align="center">59.0</td>
<td align="center">65.4</td>
<td align="center">58.0</td>
<td align="center">1820</td>
<td align="center">85.6</td>
<td align="center">69.8</td>
<td align="center">76.7</td>
<td align="center">57.4</td>
<td align="center">98.7%</td>
</tr>
<tr>
<td align="left">SCOPE</td>
<td align="left">NeurIPS’25</td>
<td align="center">60.1</td>
<td align="center">63.6</td>
<td align="center">56.8</td>
<td align="center">1804</td>
<td align="center">86.4</td>
<td align="center">68.8</td>
<td align="center">77.2</td>
<td align="center">57.7</td>
<td align="center">98.3%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>61.1</strong></td>
<td align="center"><strong>64.2</strong></td>
<td align="center"><strong>57.9</strong></td>
<td align="center"><strong>1808</strong></td>
<td align="center"><strong>86.4</strong></td>
<td align="center"><strong>68.2</strong></td>
<td align="center"><strong>77.9</strong></td>
<td align="center"><strong>57.4</strong></td>
<td align="center"><strong>98.9%</strong></td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/o VE (128 Tokens, 22.2%)</strong></td>
</tr>
<tr>
<td align="left">FastV</td>
<td align="left">ECCV’24</td>
<td align="center">49.6</td>
<td align="center">56.1</td>
<td align="center">56.4</td>
<td align="center">1490</td>
<td align="center">59.6</td>
<td align="center">60.2</td>
<td align="center">61.8</td>
<td align="center">50.6</td>
<td align="center">83.2%</td>
</tr>
<tr>
<td align="left">PDrop</td>
<td align="left">CVPR’25</td>
<td align="center">56.0</td>
<td align="center">61.1</td>
<td align="center">56.6</td>
<td align="center">1644</td>
<td align="center">82.3</td>
<td align="center">68.3</td>
<td align="center">72.9</td>
<td align="center">55.1</td>
<td align="center">94.0%</td>
</tr>
<tr>
<td align="left">SparseVLM</td>
<td align="left">ICML’25</td>
<td align="center">58.4</td>
<td align="center">64.5</td>
<td align="center">51.1</td>
<td align="center">1746</td>
<td align="center">85.0</td>
<td align="center">68.6</td>
<td align="center">73.8</td>
<td align="center">56.7</td>
<td align="center">95.6%</td>
</tr>
<tr>
<td align="left">DART</td>
<td align="left">EMNLP’25</td>
<td align="center">58.7</td>
<td align="center">63.2</td>
<td align="center">57.5</td>
<td align="center">1840</td>
<td align="center">80.1</td>
<td align="center">69.1</td>
<td align="center">75.9</td>
<td align="center">56.4</td>
<td align="center">97.0%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>60.0</strong></td>
<td align="center"><strong>62.9</strong></td>
<td align="center"><strong>57.1</strong></td>
<td align="center"><strong>1807</strong></td>
<td align="center"><strong>86.7</strong></td>
<td align="center"><strong>68.5</strong></td>
<td align="center"><strong>76.5</strong></td>
<td align="center"><strong>57.2</strong></td>
<td align="center"><strong>98.1%</strong></td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/ VE (128 Tokens, 22.2%)</strong></td>
</tr>
<tr>
<td align="left">VisionZip</td>
<td align="left">CVPR’25</td>
<td align="center">57.6</td>
<td align="center">62.0</td>
<td align="center">56.7</td>
<td align="center">1761.7</td>
<td align="center">83.2</td>
<td align="center">68.9</td>
<td align="center">75.6</td>
<td align="center">56.8</td>
<td align="center">96.4%</td>
</tr>
<tr>
<td align="left">HoloV</td>
<td align="left">NeurIPS’25</td>
<td align="center">57.7</td>
<td align="center">63.9</td>
<td align="center">56.5</td>
<td align="center">1802</td>
<td align="center">84.0</td>
<td align="center">69.8</td>
<td align="center">75.5</td>
<td align="center">56.8</td>
<td align="center">97.2%</td>
</tr>
<tr>
<td align="left">SCOPE</td>
<td align="left">NeurIPS’25</td>
<td align="center">59.7</td>
<td align="center">62.5</td>
<td align="center">56.9</td>
<td align="center">1776</td>
<td align="center">86.1</td>
<td align="center">68.4</td>
<td align="center">76.5</td>
<td align="center">57.2</td>
<td align="center">97.5%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>60.0</strong></td>
<td align="center"><strong>62.9</strong></td>
<td align="center"><strong>56.7</strong></td>
<td align="center"><strong>1799</strong></td>
<td align="center"><strong>86.4</strong></td>
<td align="center"><strong>69.2</strong></td>
<td align="center"><strong>77.1</strong></td>
<td align="center"><strong>57.0</strong></td>
<td align="center"><strong>98.1%</strong></td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/o VE (64 Tokens, 11.1%)</strong></td>
</tr>
<tr>
<td align="left">FastV</td>
<td align="left">ECCV’24</td>
<td align="center">46.1</td>
<td align="center">48.0</td>
<td align="center">52.7</td>
<td align="center">1256</td>
<td align="center">48.0</td>
<td align="center">51.1</td>
<td align="center">55.0</td>
<td align="center">47.8</td>
<td align="center">74.0%</td>
</tr>
<tr>
<td align="left">PDrop</td>
<td align="left">CVPR’25</td>
<td align="center">41.9</td>
<td align="center">33.3</td>
<td align="center">50.5</td>
<td align="center">1092</td>
<td align="center">55.9</td>
<td align="center">68.6</td>
<td align="center">69.2</td>
<td align="center">45.9</td>
<td align="center">74.4%</td>
</tr>
<tr>
<td align="left">SparseVLM</td>
<td align="left">ICML’25</td>
<td align="center">53.8</td>
<td align="center">60.1</td>
<td align="center">52.7</td>
<td align="center">1589</td>
<td align="center">77.5</td>
<td align="center">69.8</td>
<td align="center">68.2</td>
<td align="center">53.4</td>
<td align="center">90.6%</td>
</tr>
<tr>
<td align="left">DART</td>
<td align="left">EMNLP’25</td>
<td align="center">55.9</td>
<td align="center">60.6</td>
<td align="center">53.2</td>
<td align="center">1765</td>
<td align="center">73.9</td>
<td align="center">69.8</td>
<td align="center">72.4</td>
<td align="center">54.4</td>
<td align="center">92.8%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>58.0</strong></td>
<td align="center"><strong>61.6</strong></td>
<td align="center"><strong>53.7</strong></td>
<td align="center"><strong>1681</strong></td>
<td align="center"><strong>84.3</strong></td>
<td align="center"><strong>68.5</strong></td>
<td align="center"><strong>74.8</strong></td>
<td align="center"><strong>54.9</strong></td>
<td align="center"><strong>94.7%</strong></td>
</tr>
<tr>
<td colspan="11" align="center"><strong>w/ VE (64 Tokens, 11.1%)</strong></td>
</tr>
<tr>
<td align="left">VisionZip</td>
<td align="left">CVPR’25</td>
<td align="center">55.1</td>
<td align="center">60.1</td>
<td align="center">55.4</td>
<td align="center">1690</td>
<td align="center">77.0</td>
<td align="center">69.0</td>
<td align="center">72.4</td>
<td align="center">55.5</td>
<td align="center">93.0%</td>
</tr>
<tr>
<td align="left">HoloV</td>
<td align="left">NeurIPS’25</td>
<td align="center">55.3</td>
<td align="center">63.3</td>
<td align="center">55.1</td>
<td align="center">1715</td>
<td align="center">80.3</td>
<td align="center">69.5</td>
<td align="center">72.8</td>
<td align="center">55.4</td>
<td align="center">94.4%</td>
</tr>
<tr>
<td align="left">SCOPE</td>
<td align="left">NeurIPS’25</td>
<td align="center">58.3</td>
<td align="center">61.7</td>
<td align="center">54.4</td>
<td align="center">1698</td>
<td align="center">83.9</td>
<td align="center">68.6</td>
<td align="center">75.3</td>
<td align="center">56.6</td>
<td align="center">95.4%</td>
</tr>
<tr>
<td align="left"><strong>TRIO (Ours)</strong></td>
<td align="left"><strong>Ours</strong></td>
<td align="center"><strong>58.3</strong></td>
<td align="center"><strong>61.6</strong></td>
<td align="center"><strong>56.5</strong></td>
<td align="center"><strong>1744</strong></td>
<td align="center"><strong>86.4</strong></td>
<td align="center"><strong>68.6</strong></td>
<td align="center"><strong>75.9</strong></td>
<td align="center"><strong>56.2</strong></td>
<td align="center"><strong>96.6%</strong></td>
</tr>
</tbody>
</table>

> **Note:** Base model used is LLaVA-1.5-7B. The baseline "Upper Bound" utilizes all 576 tokens.
## Performance Results

Here is the performance comparison of our method against the baseline LLaVA models. Our approach significantly reduces computation overhead (Prefill Time, Total Time, FLOPs, and KV Cache) while maintaining competitive performance on the POPE benchmark.

| Methods | Prefill Time↓ (s) | Total Time↓ (s) | Avg FLOPs↓ (T) | KV Cache↓ (MB) | POPE (Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLaVA-1.5-7B** | 1401 (1.00×) | 2234 (1.00×) | 2.98 (1.00×) | 318 (1.00×) | 85.9 |
| +Ours(%11.1) | **1106 (1.27×)** | **1978 (1.13×)** | **0.45 (6.62×)** | **62 (5.13×)** | 84.3 |
| **LLaVA-NEXT-7B** | 4934 (1.00×) | 5921 (1.00×) | 16.67 (1.00×) | 1156 (1.00×) | 86.5 |
| +Ours(%11.1) | **1844 (2.67×)** | **2810 (2.11×)** | **2.68 (6.22×)** | **191 (6.05×)** | 84.5 |

> **Note:** > * `↓` indicates that lower is better. 
> * Values in parentheses represent the reduction ratio/speedup compared to the respective baseline.

## Set Up
## LLaVA

1. Clone this repository.

```bash
git clone https://github.com/ocy1/TRIO
cd TRIO
```

2.Environment Setup and Preparation

```conda create -n TRIO python=3.10 -y
conda activate TRIO
pip install -e .
pip install flash-attn --no-build-isolation
```
3.Download Multimodal Benchmark
Please follow the detailed instruction in https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

## Qwen2.5-VL
```
conda create -n TRIO_Qwen25VL python=3.10 -y
conda activate TRIO_Qwen25VL

pip install -U transformers==4.55.4
pip install flash-attn --no-build-isolation
pip install -e .
```







## Usage

### LLaVA

For LLaVA-1.5 and LLaVA-NeXT, you can run evaluation with the scripts under `scripts/`. A typical example is:

```bash
python -m llava.eval.model_vqa_loader \
    --model-path ../data/model/llava-v1.5-7b \
    --question-file ../data/eval/MME/llava_mme.jsonl \
    --image-folder ../data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ../data/eval/MME/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[32]' \
    --visual_token_num 96 \
    --conv-mode vicuna_v1
```

Please replace `--model-path` with your own checkpoint path. `--question-file` specifies the evaluation file, `--image-folder` should point to the corresponding image directory, and `--answers-file` is the path used to save model predictions.

The pruning behavior is mainly controlled by `--layer_list`, `--image_token_list`, and `--visual_token_num`. By default, we use:

* `--layer_list '[16]'`
* `--image_token_list '[32]'`
* `--visual_token_num 96`

This means 96 visual tokens are kept after the visual encoder, and then reduced to 32 at the 16th LLM layer. Under this setting, the model maintains an average of 64 visual tokens across all 32 layers.

Both `--layer_list` and `--image_token_list` support multiple entries. For example, you may set:

```bash
--layer_list '[8,16]' --image_token_list '[188,96]'
```

to perform progressive pruning at multiple layers, similar to PyramidDrop-style scheduling.

### Qwen2.5-VL

For Qwen2.5-VL, the evaluation command is:

```bash
python -m qwen.eval.model_vqa_loader \
    --model-path ../data/model/Qwen2.5-VL-7B-Instruct \
    --question-file ../data/eval/MME/llava_mme.jsonl \
    --image-folder ../data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ../data/eval/MME/answers/Qwen2.5-VL-7B-Instruct.jsonl \
    --temperature 0 \
    --layer-list '[14]' \
    --image-token-ratio-list '[0.333]' \
    --image-token-ratio 0.167
```

Since Qwen2.5-VL uses dynamic-resolution visual encoding, the number of visual tokens varies across images. Therefore, ratio-based settings are used instead of fixed token counts.

By default:

* `--image-token-ratio 0.167` keeps 16.7% of visual tokens after visual encoding
* `--image-token-ratio-list '[0.333]'` further reduces them to 33.3% at the 14th LLM layer

This results in an average visual token retention ratio of 11.1% across the 28 LLM layers.




## Citation
If you find our work helpful or use our code in your research, please consider citing our paper:


```bibtex
@misc{zhang2026triotokenreductioninferenceobjective,
      title={TRIO: Token Reduction via Inference-Objective Guidance for Efficient Vision-Language Models}, 
      author={Haokui Zhang and Congyang Ou and Dawei Yan and Peng Wang and Qingsen Yan and Yu Zhang and Ying Li and Rong Xiao},
      year={2026},
      eprint={2602.04657},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.04657}, 
}
```
