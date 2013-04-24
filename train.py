import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import ShuffleSplit
import csv
from rankboost import BipartiteRankBoost

def shuffleCrossValidation(labels, features, classifier, 
                           n_iter=5, test_size=0.25, random_state=1,
                           verbose=0):
    
    ss = ShuffleSplit(len(labels), n_iter=n_iter, 
                      test_size=test_size, random_state=random_state)
    
    score = []
    for iteration, (train_authors, test_authors) in enumerate(ss):
        myscore = 0.0
        X_train = [val for idx in train_authors for val in features[idx]]
        Y_train = [val for idx in train_authors for val in labels[idx]]

        classifier = classifier.fit(X_train, Y_train)
        
        # predict everything at once
        X_test = [val for idx in test_authors for val in features[idx]]
        P_test = classifier.predict_proba(X_test)[:,1]

        flatidx = 0
        for idx in test_authors:
            npapers = len(labels[idx])
            ranking = P_test[flatidx:flatidx+npapers].argsort()[::-1]
            flatidx += npapers
            
            ranked_labels = [labels[idx][rank] for rank in ranking]
            myscore += scoreAuthor(ranked_labels)
            
        score.append(myscore/len(test_authors))
        if verbose == 1:
            print 'iteration', iteration, 'score:', myscore/len(test_authors)

    score = np.array(score)
    print 'score mean, std:', score.mean(), score.std()
            
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

def trainAndPredict(trainlabels, trainfeatures, 
                    testlabels, testfeatures, classifier):
        
    X_train = [paperfeature for authorfeatures in trainfeatures 
               for paperfeature in authorfeatures]
    Y_train = [paperlabel for authorlabels in trainlabels 
               for paperlabel in authorlabels]

    classifier.fit(X_train, Y_train)

    '''
    should prune based on importances, seems to improve results
    print classifier.feature_importances_
    for feature in trainfeatures:
        for val in feature:
            del val[3]
    '''
    
    X_test = [paperfeature for authorfeatures in testfeatures 
               for paperfeature in authorfeatures]
    P_test = classifier.predict_proba(X_test)[:,1]
        
    flatidx = 0
    ranked_papers = []
    for author, papers in testlabels:
        npapers = len(papers)
        ranking = P_test[flatidx:flatidx+npapers].argsort()[::-1]
        flatidx += npapers
        
        ranked_papers.append([author, [papers[rank] for rank in ranking]])
        
    with open('DataRev2/submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for author, papers in ranked_papers:
            writer.writerow([author, " ".join([str(pid) for pid in papers])])

if __name__ == '__main__':
    #classifier = RandomForestClassifier(n_estimators=100, 
    #                                    n_jobs=-1,
    #                                    min_samples_split=10,
    #                                    random_state=1)

    #classifier = AdaBoostClassifier(n_estimators=200)
    classifier = BipartiteRankBoost(n_estimators=50, verbose=0)

    #classifier = GradientBoostingClassifier(n_estimators=200, 
    #                                        subsample=0.8, 
    #                                        learning_rate=0.15,
    #                                        random_state=1)
    
    trainlabels, trainfeatures = cPickle.load(open('train_features.p', 'rb'))
    
    shuffleCrossValidation(trainlabels, trainfeatures, classifier, n_iter=5, verbose=1)
    
    #testlabels, testfeatures = cPickle.load(open('test_features.p', 'rb'))
    
    #trainAndPredict(trainlabels, trainfeatures, testlabels, testfeatures, classifier)


    
