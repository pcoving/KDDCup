import numpy as np
import cPickle
import csv
import itertools
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import ShuffleSplit
from rankboost import BipartiteRankBoost
import random
import sys

def shuffleCrossValidation(labels, features, classifier, 
                           n_iter=5, test_size=0.25, random_state=1,
                           verbose=0, pairwise=False):
    
    ss = ShuffleSplit(len(labels), n_iter=n_iter, 
                      test_size=test_size, random_state=random_state)
    
    score = []
    for iteration, (train_authors, test_authors) in enumerate(ss):

        if not pairwise:
            X_train = [val for idx in train_authors for val in features[idx]]
            y_train = [val for idx in train_authors for val in labels[idx]]
        else:
            X_train = [features[idx] for idx in train_authors]
            y_train = [labels[idx] for idx in train_authors]
            y_train, X_train = pairwise_transform(y_train, X_train)
            def classifierComp(x1, x2):
                xtrans = x2 + x1 + [x2 - x1 for (x1, x2) in zip(x1,x2)]
                return int(classifier.predict(xtrans)[0])

        classifier = classifier.fit(X_train, y_train)
        
        if not pairwise:
            # predict everything at once
            X_test = [val for idx in test_authors for val in features[idx]]
            P_test = classifier.predict_proba(X_test)[:,1]
                
        myscore = 0.0
        flatidx = 0
        for idx in test_authors:
            myfeatures, mylabels = features[idx], labels[idx]

            if pairwise:                
                c = zip(myfeatures, mylabels)
                random.shuffle(c)
                myfeatures = [e[0] for e in c]
                mylabels = [e[1] for e in c]
                        
                ranking = sorted(range(len(myfeatures)), 
                                 key=myfeatures.__getitem__, 
                                 cmp=classifierComp)
            else:
                npapers = len(mylabels)
                ranking = P_test[flatidx:flatidx+npapers].argsort()[::-1]
                flatidx += npapers

            ranked_labels = [mylabels[rank] for rank in ranking]

            myscore += scoreAuthor(ranked_labels)
            
        score.append(myscore/len(test_authors)) # MAP
        if verbose > 0:
            print 'iteration', iteration, 'score:', myscore/len(test_authors)

    score = np.array(score)
    print 'score mean, std:', score.mean(), score.std()
            
def trainAndPredict(trainlabels, trainfeatures, 
                    testlabels, testfeatures, classifier, pairwise=False):
        
    if not pairwise:
        X_train = [paperfeature for authorfeatures in trainfeatures 
                   for paperfeature in authorfeatures]
        y_train = [paperlabel for authorlabels in trainlabels 
                   for paperlabel in authorlabels]
    else:
        X_train = [authorfeatures for authorfeatures in trainfeatures]
        y_train = [authorlabels for authorlabels in trainlabels]
        y_train, X_train = pairwise_transform(y_train, X_train)
        def classifierComp(x1, x2):
            xtrans = x2 + x1 + [x2 - x1 for (x1, x2) in zip(x1,x2)]
            return int(classifier.predict(xtrans)[0])

    classifier.fit(X_train, y_train)

    if not pairwise:
        X_test = [paperfeature for authorfeatures in testfeatures 
                  for paperfeature in authorfeatures]
        P_test = classifier.predict_proba(X_test)[:,1]
        
    flatidx = 0
    ranked_papers = []
    for idx, labels in enumerate(testlabels):
        author, papers = labels

        if pairwise:
            ranking = sorted(range(len(testfeatures[idx])), 
                             key=testfeatures[idx].__getitem__, 
                             cmp=classifierComp)
        else:
            npapers = len(papers)
            ranking = P_test[flatidx:flatidx+npapers].argsort()[::-1]
            flatidx += npapers
        
        ranked_papers.append([author, [papers[rank] for rank in ranking]])
        
    with open('submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for author, papers in ranked_papers:
            writer.writerow([author, " ".join([str(pid) for pid in papers])])

def pairwise_transform(labels, features):

    new_features = []
    new_labels = []
    for label, feature in zip(labels, features):
        
        nconfirmed = sum(label)
        for iconf, idel in itertools.product(range(nconfirmed), range(nconfirmed, len(label))):
            assert label[iconf] == 1 and label[idel] == 0
            new_features.append(feature[iconf] + feature[idel] + [feat1 - feat2 for (feat1, feat2) in zip(feature[iconf], feature[idel])])
            new_labels.append(1)
            new_features.append(feature[idel] + feature[iconf] + [feat2 - feat1 for (feat1, feat2) in zip(feature[iconf], feature[idel])])
            new_labels.append(-1)
            
    return new_labels, new_features

def loadFeatures(namelist, mode):
    featurelist = []
    for name in namelist:
        filename = name + '.' + mode
        featurelist.append(cPickle.load(open(filename, 'rb')))
    features = []
    for feats in zip(*featurelist):
        features.append([list(tup) for tup in zip(*feats)])
        
    return features

def scoreAuthor(ranked_labels):
    # computes average precision for ranked labels of an author's papers

    score = 0.0
    confirmedCount = 0
    for idx, label in enumerate(ranked_labels):
        if label is 1:
            confirmedCount += 1
            score += float(confirmedCount)/float(idx+1)            
    score /= ranked_labels.count(1)

    return score

if __name__ == '__main__':
    #classifier = RandomForestClassifier(n_estimators=100, 
    #                                    n_jobs=-1,
    #                                    min_samples_split=10,
    #                                    random_state=1)
    
    #classifier = AdaBoostClassifier(n_estimators=200)
    #classifier = BipartiteRankBoost(n_estimators=50, verbose=1)

    classifier = GradientBoostingClassifier(n_estimators=200, 
    #                                        subsample=0.9, 
    #                                        learning_rate=0.05,
    #                                        min_samples_split=2, 
    #                                        min_samples_leaf=1,
    #                                        max_depth=1,
                                            random_state=1,
                                            verbose=1)  

    feature_list = ['nauthors', 'npapers', 'year', 'nsamevenue', 'nattrib', 'paperrank', 'globalpaperrank', 'ncoauthor']

    trainfeatures = loadFeatures(feature_list, mode='train')
    trainlabels = cPickle.load(open('labels.train', 'rb')) 

    shuffleCrossValidation(trainlabels, trainfeatures, classifier, n_iter=5, verbose=2, pairwise=False)
    
    #trainAndPredict(trainlabels, trainfeatures, testlabels, testfeatures, classifier, pairwise=True)


    
