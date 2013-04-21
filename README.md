KDDCup
------
Details of the competition can be found [here][1].

To keep the repo lightweight, the dataset does not ship with the code. The `.csv` data can be downloaded from [Kaggle][2] (requires account) and untarred in the top-level directory.

Some benchmarks require the [scikit-learn][3] package.

Theory
------
The competition appears to be an instance of bipartite ranking:

- [A boosting algorithm for learning bipartite ranking functions with partially labeled data][3]


[1]: https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge
[2]: https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge/data
[3]: http://scikit-learn.org/dev/
[4]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.4696&rep=rep1&type=pdf