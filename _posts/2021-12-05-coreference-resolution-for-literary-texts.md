---
layout: post
title: Coreference Resolution for Literary Texts
tags: [Coreference Resolution, Neural Network, English Literature, Computational Humanities]
authors: Yang, Funing, Wellesley College
---

# Coreference Resolution for Literary Texts

## The Task

The coreference resolution task is a long-standing clustering task in Natural Language Processing (NLP) that links mentions that refer to the same entities together. A mention is a noun or a noun phrase and normally falls into categories such as a person, a location, or an organization, etc. Given this definition, the coreference resolution task is often closely related to the Named Entity Recognition (NER) task, where it identifies all named entities (such as person, location, organization, date, etc.). And indeed, NER is always incorporated in coreference resolution models or inference algorithms.

For example, consider the following sentence:

"(Harry Potter) is the (boy-who-lived), and (he) made {Voldemort} lose {his} power when (he) was a baby." As humans, we can easily perceive that the mentions in parenthesis are referring to Harry, and in curly brackets are the ones referring to Voldemort. However, more complex sentence structures will give rise to subtle challenges for this task.

Coreference resolution has lots of applications, and is often an important foundational task in many NLP tasks, such as question answering, text summarization, reading comprehension, and in interdisciplinary areas such as computational social sciences and humanities, which we'll elaborate on in this post.

## Challenges for the Literary domain

There are several challenges for the literary domain, some notable ones as follows:

* Lengthy sentences: according to [[Bamman et al.,2019]](#Bamman), literary texts in the Litbank dataset, a representative selection of 100 works from classic English literary texts across 18th to 20th century, are four times longer than the OntoNotes dataset for traditional named entity recognition tasks. The length of the sentence poses a significant challenge for the coreference models as they tend to have limited maximal distances for preceding and subsequent spans to make inferences on.

* Nested mentions: unlike straightforward news texts and day-to-day English usage, literary texts often contain ambiguous nested structures in their figurative language usages. For example, "the smart and wise Hobbit Frodo" has two layers of mentions: the entire phrase, or just "Frodo", the proper name. Given this challenge, coreference resolution models for the English literature domain often needs to be trained on either the longest or shortest spans of mentions.

* Ambiguity: unlike newswires or reports, literary entities can change over the course of a novel. For instance, is the seven-year-old Pip at the beginning of Dicken's *Great Expectations* the same identity as the thiry-year-old Pip in the end? Such questions are philosophical and computer programs cannot answer in full. Most coreference systems treat the entities as static throughout a piece of text for simplicity.

## Evaluation metrics

Having briefly elaborated on the task itself as well as specific domain challenges, let us turn to the evaluation metrics before diving into the specific model architectures. To evaluate the effectiveness of a coreference resolution model, the research community has developed several metrics employed in the well-known CONLL shared task:

* B-CUBED. This metric takes the weighted sum of the precision or recall for each mention, e.g. 
$ Recall = \sum_{r \in R} w_r \cdot \sum_{t \in T} \frac{|t \cap r|^2}{|t|} $. $R$ is the predicted result set of entity clusters, $T$ is the true set of entity clusters, $w_r$ is the weight for entity $r$, and the right-most factor is the number of true positives divided by the number of all positives [[Moosavi et al., 2016]](#Moosavi).

* MUC. If we consider a coreferent cluster as a set of linked mentions, the MUC metric measures the minimum number of link modifications needed to make $R$ the same as $T$. The partition function measures the number of clusters in the set of predicted clusters $R$ that intersect with the given true cluster $$t$$.

* CEAF. This metric first maps predicted result entity clusters $r \in R$ to true entity clusters $t \in T$ via a similarity measure denoted $\phi(T, R)$. We then use the mapped true cluster $m(r)$ with maximum similarity to $r$ to compute the precision or recall.

## Coreference Models

### Popular Approaches at a Glance

As coreference resolution is a long-standing traditional task, there has been a surge of rule-based, linguistic-driven, and neural-network-based approaches. [[Lee et al., 2017]](#Lee) has summarized the popular model approaches as follows:

1. Mention-pair classifiers

Being the most intuitive yet inefficient model type, mention-pair classifiers iterate over all possible pairs of mentions (where a `mention' is defined as a reference to an entity) and train a binary classifier on each pair to predict whether or not the mentions in that pair are coreferent. Then define some thresholding criterion to turn these yes-no coreference pairs into a cluster of mentions for each entity. {https://web.stanford.edu/class/archive/cs/cs224n/cs224n. 1162/handouts/cs224n-lecture11-coreference.pdf}

2. Entity-level models

Entity-level models, also called cluster-based models, incorporate features defined over clusters of entities as opposed to just mention pairs when making decisions regarding coreference scores. In [[Clark and Manning, 2016]](#ClarkandManning), the authors proposed a neural network approach with dense vector representations over pairs of coreference clusters for model training. 

3. Mention-ranking models
The mention-pair method only indirectly solves the coreference resolution task. Note that we want clusters for each entity, not coreferent pairs, and it is unclear what is the best way to merge these pairs into clusters. Furthermore, the mention-pair technique allows for incompatible mentions in the same cluster: a female entity could be coreferent with `he' in one pair and coreferent with `she' in another pair, so the gender of the cluster would be ambiguous. The mention-ranking approach helps prevent this problem by ranking the possible antecedents and choosing just one antecedent for a given mention {https://galhever.medium.com/a-review-to-co-reference-resolution-models-f44b4360a00}


### End-to-end Neural Coreference Resolution

[[Lee et al., 2017]](#Lee) has proposed the first end-to-end neural coreference resolution system that becomes the foundation for many subsequent neural-network based approaches for this task in the literary domain. Then, what does their architecture look like? 

#### The Key Idea
The approache considers all possible spans within a given document as potential mentions, and learns distributions over possible antecedents for each. The model computes contextualized span embeddings with the attention mechanism, and maximizes the marginal likelihood of gold antecedent spans from coreference clusters. 

#### The Input
The input for the model contains a document, D, containing T words, with optional metadata such as speakers or genre information.

#### The Model
[[Lee et al., 2017]](#Lee) utilizes LSTM and attention mechanisms for their architecture, and defined pair-wise coreference scores to apply in their span representations.

1. Input to Span Representation

(Insert architecture image! - Figure 1)

* The first step of the end-to-end pipeline takes in the document input (as simple as a sentence), tokenize and calculate the word and character embedding (x), fixed pretrained word embeddings from 300-dimensonal GloVe embeddings, and 1-dimensional convolution neural networks (CNN) over characters.

* The word embeddings then are feeded as inputs into the Bidirectional LSTM network (x*), where it encodes each word in its context, as the surrounding context for each mention span plays an integral role on the scoring of the coreference pairs. 

(insert formulas on pg.190 in Span Representations)

As we can observe from the equations above, x*t is the concatenated output of the bidirectional LSTM.

It is interesting to note that Lee et al. only uses independent LSTMs for every sentence and decides that cross-sentence context is not helpful, this is not the case for literary texts as the mentions can span across chapters.

* The Bidirectional LSTM outputs then gets passed to the next layer of the network to calculate the span head $$\hat{x}$$, where it is a weighted sum of word vectors in a given span. Attention mechanism is applied using this step over words within each span.

(Insert: formula block on top of p.191)

* The final span representation is then calculated as the accumulation of all the above information with:

(Insert: formula block for gi on p191)



2. Coreference Scoring Calculation
* Once we have the span representation calculated from the first step, we then move on to the coreference scoring calculation step to obtain the final coreference score for a pair of spans.

The architecture is as follows:

(insert Figure 2. on p.190)

Recall from the last step that each possible span i in the document will have a corresponding vector representation $g(i)$. Given the span representations, the scoring functions are then calculated via feed-forward neural networks:

(insert equation on p. 190 left)

* With the coreference score (s) being calculated, we are finally able to learn the conditional probability distribution $P(y1,...,yn|D)$ with the final softmax layer to represent the most likely configuration that produces the correct clustering. 

The equation is as follows:

(insert equation (top) on p. 189)

$s(i,j)$ is the pairwise score for a coreference link between span i and span j in document D. Lee et al. define it as follows:

(insert equation (bottom) on p.189)

The $\epsilon$ represents the dummy antecedent, where span j is either not an entity mention, or it is an entity mention but it is not coreferent with any previous span. By setting the coreference score to 0, the model is able to aggressively prune away the pairs less likely to belong in the same cluster to save computational costs. 


#### Challenges

(1) As mentioned in the paper, the biggest challenge of the model is the computational size, with the full model reaching $O(T^4)$ with document length T. To address this challenge, the authors proposes an aggressive pruning approach where they greedily prune potential spans with the following constraints:
They only consider spans up to L words for unary mention scores; further, they only keep the top $\lambda*T$ spans with the highest mention scores and consider only up to K antecedents for each.

(2) From a qualitative analysis, we can observe that the usage of word embeddings for word similarity makes the model prone to predicting false positive links with related/similar words. Consider the following example listed in the paper: "(The flight attendants) have until 6:00 today to ratify labor concessions. (The pilots’) union and ground crew did so yesterday." The model mistakenly identifies "the flight attendants" and "the pilots'" as coreferent, possibly due to the shared contextual embeddings they have. 

(3) We can also observe the texts for experiments are relatively simple without too complex syntactic structures or figurative languages. While this is sufficient for most daily text sources, literature does requires more finetuning and domain-specific adaptations.

#### References

(Double check citation format!)
Nafise   Sadat   Moosavi   and   Michael   Strube.   2016.Which coreference evaluation metric do you trust?a proposal for a link-based entity aware metric.As-sociation for Computational Linguistics
### Code

{% highlight js %}
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
{% endhighlight %}


### Lists

<dl>
  <dt>HyperText Markup Language (HTML)</dt>
  <dd>The language used to describe and define the content of a Web page</dd>

  <dt>Cascading Style Sheets (CSS)</dt>
  <dd>Used to describe the appearance of Web content</dd>

  <dt>JavaScript (JS)</dt>
  <dd>The programming language used to build advanced Web sites and applications</dd>
</dl>


### Tables
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Upvotes</th>
      <th>Downvotes</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Totals</td>
      <td>21</td>
      <td>23</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Alice</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Charlie</td>
      <td>7</td>
      <td>9</td>
    </tr>
  </tbody>
</table>

-----

# Images, gifs, and assets

If you include an hosted elsewhere on the web, the process is trivial. Simply use the standard GitHub-flavoured-MarkDown syntax.

`![Example Image](https://iclr.cc/static/core/img/ICLR-logo.svg)` becomes:

![Example Image](https://iclr.cc/static/core/img/ICLR-logo.svg)
(be wary of copyrights).

However, if your image must be hosted locally, it's a bit more touchy. You must add the site's URL (use the `{{ site.url }}` syntax).

`![ICLR LOGO](\{\{ site.url \}\}/public/images/example_content_jdoe/ICLR-logo.png)` (without the `\` character before `{` and `}`) becomes:
![ICLR LOGO]({{ site.url }}/public/images/example_content_jdoe/ICLR-logo.png)

In order to ensure there is no namespace conflict between submissions, we ask you to add your images
inside a separate folder inside `/public/images`. For example, this blog post is called "example-content" and the first
author is John Doe, so the images go in `/public/images/example_content_jdoe`. Try to pick a unique name.

To add HTML figures, you must add them to the `_includes` folder. Again, add them under a unique name.

Here, `\{\% include example_content_jdoe/plotly_demo_1.html \%\}` (without the `\` character before `}`, `{` and `%`) becomes

{% include example_content_jdoe/plotly_demo_1.html %}



## How to add $\LaTeX$ commands to your posts:

### Inline

To add inline math, you can use `$ <math> $`. Here is an example:


`$ \sum_{i=0}^j \frac{1}{2^n} \times i $` becomes
$ \sum_{i=0}^j \frac{1}{2^n} \times i $

### Block

To add block math, you *must* use `$$<math>$$`. Here are some examples:

```
$$\begin{equation}
a \times b \times c = 0 \\
j=1 \\
k=2 \\
\end{equation}$$
```

...becomes...

$$\begin{equation}
a \times b \times c = 0 \\
j=1 \\
k=2 \\
\end{equation}$$

```
$$\begin{align}
i2 \times b \times c =0 \\
j=1 \\
k=2 \\
\end{align}$$
```

...becomes...

$$\begin{align}
i2 \times b \times c =0 \\
j=1 \\
k=2 \\
\end{align}$$

Don't forget the enclosing `$$`! Otherwise, your newlines won't work:

```
\begin{equation}
i2=0 \\
j=1 \\
k=2 \\
\end{equation}
```

...becomes...

\begin{equation}
i2=0 \\
j=1 \\
k=2 \\
\end{equation}