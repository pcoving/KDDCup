KDDCup 2013
===========
Details of the competition can be found [here][1].

To keep the repo lightweight, the dataset does not ship with the code. The `.csv` data can be downloaded from [Kaggle][2] (requires account) and untarred in the top-level directory.

Some benchmarks require the [scikit-learn][3] package.

Theory
------
Semi-supervised learning review:

- [Semi-Supervised Learning Literature Survey][4]

The competition appears to be an instance of bipartite ranking:

- [An Efficient Boosting Algorithm for Combining Preferences][5]
- [A boosting algorithm for learning bipartite ranking functions with partially labeled data][6]

Personalized PageRank with Monte Carlo looks promising:

- [Monte Carlo methods in PageRank computation][7]

Ideas
-----
- Build features with link analysis on author/paper graph, possibly with [NetworkX][8] library (doesn't seem to scale, looks like we need our own implementation)
- How to use titles, keywords, affliction and other raw text features?

[1]: https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge
[2]: https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge/data
[3]: http://scikit-learn.org/dev/
[4]: http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf
[5]: http://www.ai.mit.edu/projects/jmlr/papers/volume4/freund03a/freund03a.pdf
[6]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.4696&rep=rep1&type=pdf
[7]: http://www-sop.inria.fr/members/Konstantin.Avratchenkov/pubs/mc.pdf
[8]: http://networkx.github.io/