# Practical Machine Learning and Deep Learning
## Assignment 1
### Report 1
### By Mohamed Nguira

### Contents:
 1. Task Description
 2. Baseline Idea
 3. Hypothesis 1
 4. Hypothesis 2
 5. Hypothesis 3
 6. Results

### 1) Task Description:
Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

Your assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 

### 2) Baseline Idea:
The proposed solution contains a constructive algorithm that uses preprocessing, data analysis and various types of pretrained and untrained models.
Our problem is a NLP problem so we will focus on the idea of using unigrams and bigrams as they are considered strong tools for any NLP based algorithm.
We recall that Unigrams are the simplest and most basic form of text representation. They are essentially individual words or tokens in a given text. Each word in a text is considered independently as a unigram. On the other hand, Bigrams are a step beyond unigrams and involve grouping consecutive pairs of words in a text. In other words, bigrams consist of two adjacent words in a text, considered together.
We will here use a combination of both.
First of all, the data was arranged in such a way that the 1st pair always has a greater toxicity score than the 2nd pair. 

### 3) Hypothesis 1:
#### We will need to train a model on detecting the level of toxicity for a given text.
Given the dataset, It is possible to construct and train a model to compute the toxicity level of a given text. This will help us evaluate the output of our main algorithm and help to reduce the toxicity level of the source text since it's the main task of the problem alongside keeping the same meaning as the original phrase.

My approach is to use the ROBERTA model for this task.  Roberta (A Robustly Optimized BERT Pretraining Approach) is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model, developed by Facebook AI. 
RoBERTa is pre-trained on a massive amount of text data from the internet, similar to BERT. However, RoBERTa uses more extensive training data and employs additional techniques like sentence-level and document-level masked language modeling, making it more robust.

The RoBERTa model demonstrates improved performance on a wide range of NLP benchmarks, including tasks like text classification, named entity recognition, sentence pair classification, and more. Its increased training data, longer training time, and dynamic masking all contribute to its robustness and effectiveness.

### 4) Hypothesis 2:
#### Summarizing the source text and identifying the number of stop word may correlate to the level of toxicity

After further analysis of the dataset, it is apparent that text with higher toxicity rate usually have complicated structure which can be reduced by summarizing it. Furthermore, we will train a t5-model to be able to summarize a given text.
T5-model is a versatile and powerful natural language processing (NLP) model introduced by Google AI in a research paper in 2019. T5 builds on the transformer architecture, which has been highly influential in NLP, and it's designed with a text-to-text framework, where all NLP tasks are cast as a text generation task.

T5 is designed to take both input and output as text, meaning that all NLP tasks, such as text classification, translation, summarization, and question-answering, are converted into a text generation problem. This approach simplifies the modeling process and allows for a unified architecture to handle various tasks.

For stop words, We will consider the following statistics for each type of sentences:

1. Number of stop words
2. The portion of stop word-like tokens out of all tokens
3. The average length of a non-stop word

We will further perform a feature importance analysis using RandomForestRegressor to identify which characteristics may affect the toxicity of the text

### 5) Hypothesis 3:
#### It is best to use a seq2seq model with a penalty for toxic sentences and dissimilarity between the 2 sentences
We recall that the main problem is to generate a non toxic phrase conserving the same meaning. So it make sense to train a model to penalize toxic phrases using the pre trained model from hypothesis 1 and penalize phrases that lose the meaning of the original phrase.
For this task, it's best to use the BART model,
which stands for "Bidirectional and Auto-Regressive Transformers,". It is a sequence-to-sequence model based on the transformer architecture, and it was introduced in a research paper in 2019.

It is well suited for our task since BART combines two fundamental principles. It has a bidirectional architecture, which means it can understand context from both the left and the right of a given word or token. BART uses bidirectional encoders for pre-training, which helps it capture more comprehensive context.

Fine-Tuning for Specific Tasks: After pre-training, BART can be fine-tuned for various natural language processing tasks, such our original task of detoxification of text.

BART has demonstrated strong performance on a variety of text generation and text understanding tasks and has become a valuable tool in the field of natural language processing. It's particularly well-regarded for its ability to generate high-quality text summaries and has been used in applications where generating human-like text is crucial.

#### 6) Results:

