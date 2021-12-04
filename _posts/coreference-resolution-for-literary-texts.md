---
layout: post
title: Coreference Resolution for Literary Texts
tags: [test, tutorial, markdown]
authors: Yang, Funing, Wellesley College
---


<div class="message">
  Hello world!
</div>


## Inline HTML elements

HTML defines a long list of available inline tags, a complete list of which can be found on the [Mozilla Developer Network](https://developer.mozilla.org/en-US/docs/Web/HTML/Element).

- **To bold text**, use `<strong>`.
- *To italicize text*, use `<em>`.
- Abbreviations, like <abbr title="HyperText Markup Langage">HTML</abbr> should use `<abbr>`, with an optional `title` attribute for the full phrase.
- Citations, like <cite>&mdash; Mark otto</cite>, should use `<cite>`.
- <del>Deleted</del> text should use `<del>` and <ins>inserted</ins> text should use `<ins>`.
- Superscript <sup>text</sup> uses `<sup>` and subscript <sub>text</sub> uses `<sub>`.

Most of these elements are styled by browsers with few modifications on our part.

## Coreference Resolution for Literary Texts

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


* Praesent commodo cursus magna, vel scelerisque nisl consectetur et.
* Donec id elit non mi porta gravida at eget metus.
* Nulla vitae elit libero, a pharetra augue.

1. Vestibulum id ligula porta felis euismod semper.
2. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
3. Maecenas sed diam eget risus varius blandit sit amet non magna.

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