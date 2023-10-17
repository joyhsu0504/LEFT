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

## Setup
Run the following commands to install necessary dependencies.
```
pip install git+https://github.com/concepts-ai/concepts
```

## Train & evaluation
Please see the individual READMEs. 

## Demo
Coming soon!

## Warning
LEFT leverages a pre-trained large language model as its language interpreter, and hence, even though our prompts are general examples of first-order logic, we do not have direct control over the LLM's generation. The LLM may output harmful biase.

