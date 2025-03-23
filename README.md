# A Multi-Sample Speculative-DEcoding-Based LLM Reasoning System 

An LLM reasoning system combining EMS-SD for multi-sample speculative decoding and LLMonk for multi-sample reasoning.

## Usage
The EMS-SD method necessitates the alteration of theCUDA kernel, which is implemented on the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) framework.

So you can to configure the environments following [FasterTransformer GPT Guide](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md).

The main steps involved are as follows:
1. Install dependencies and compile C++ libraries.
2. Download huggingface opt model and convert, including opt-350m, opt-13b, etc.

```bash
./env_setup.sh
```

**Run LLMonk on GSM8k:**

```bash
./llmonk_run.sh
```

**Analyze the Speed of Model Inference:**\
Change the root directory and the experiment name in the `parse+results.py` script, and run
```bash
python parse_results.py
```

**Check the Correctness of the results:**\
Change the root directory and the experiment name in the `correctness_check.py` script, and run
```bash
python correctness_check.py
```



## Acknowledgements

1. [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
2. [LLMA](https://arxiv.org/abs/2304.04487)
3. [Draft Model Predication](https://arxiv.org/abs/2211.17192)
4. [Large Language Model Monkey](https://arxiv.org/abs/2407.21787)
5. [EMS-SD](https://github.com/niyunsheng/EMS-SD)
