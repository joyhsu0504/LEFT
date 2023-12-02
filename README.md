# Logic-Enhanced Foundation Model


![figure](figure.png)
<br />
<br />
**What’s *Left*? Concept Grounding with Logic-Enhanced Foundation Models**
<br />
[Joy Hsu](http://web.stanford.edu/~joycj/)\*,
[Jiayuan Mao](http://jiayuanm.com/)\*,
[Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html),
[Jiajun Wu](https://jiajunwu.com/)
<br />
In Conference on Neural Information Processing Systems (NeurIPS) 2023
<br />

[[paper](https://arxiv.org/abs/2310.16035)]  [[project page](https://web.stanford.edu/~joycj/projects/left_neurips_2023.html)] [[colab](https://colab.research.google.com/drive/1PHHvjImk7IK2m48j1TkCywxAYr7MatUT?usp=sharing)]

## Colab
Run this [colab](https://colab.research.google.com/drive/1PHHvjImk7IK2m48j1TkCywxAYr7MatUT?usp=sharing) to train and evaluate LEFT on a new dataset in ~100 lines of code! Additionally, try out this [colab](https://colab.research.google.com/drive/1b0Bzlyr05GS01phCDfnggVB54Zr-y9Sy?usp=sharing) to evaluate LEFT on one of our trained domains.

## Setup
Run the following commands to setup LEFT.

Make a new conda environment.
```bash
  conda create -n left python=3.10
  conda activate left
```

Install [Jacinle](https://github.com/vacancy/Jacinle).
```bash
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<PATH_TO_JACINLE>/bin:$PATH
```

Install [Concepts](https://github.com/concepts-ai/concepts).
```bash
  git clone https://github.com/concepts-ai/Concepts.git
  cd Concepts
  pip install -e .
```

Install the following libraries.
```bash
  pip install PyYAML
  pip install peewee
```

Note: you may need to install a PyTorch version that has been compiled with your version of the CUDA driver. For example, `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`.

## Demo
Check out this [demo notebook](starter-simple-shapes.ipynb) to apply LEFT on a new dataset in ~100 lines of code! See the same notebook in [colab](https://colab.research.google.com/drive/1PHHvjImk7IK2m48j1TkCywxAYr7MatUT?usp=sharing) form as well. 

Install the following libraries to run the demo.
```bash
  conda install -c conda-forge notebook
  pip install opencv-python
```

You may need to add the below import in the demo notebook.
```bash
  import sys
  sys.path.append("<PATH_TO_JACINLE>")
```

## Train & evaluation
Please see the individual READMEs to train and evaluate models. 

Install the following library to train models.
```bash
  conda install tensorflow
  pip install chardet
```

Note: Before running train & eval commands, set `export PATH=<PATH_TO_JACINLE>/bin:$PATH`.

## Warning
LEFT leverages a pre-trained large language model as its language interpreter, and hence, even though our prompts are general examples of first-order logic, we do not have direct control over the LLM's generation. The LLM may output harmful biases.

