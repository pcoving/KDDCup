import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import ShuffleSplit
import csv
    
def shuffleCrossValidation(labels, features, classifier, 
                           n_iter=5, test_size=0.25, random_state=1):
    
    ss = ShuffleSplit(len(labels), n_iter=n_iter, 
                      test_size=test_size, random_state=random_state)
    
    score = []
    for train_authors, test_authors in ss:
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
    score /= sum(ranked_labels)

    return score

def trainAndPredict(trainlabels, trainfeatures, 
                    testlabels, testfeatures, classifier):

    X_train = [val for feature in trainfeatures for val in feature]
    Y_train = [val for label in trainlabels for val in label]

    classifier.fit(X_train, Y_train)

    X_test = [val for feature in testfeatures for val in feature]
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
    #classifier = AdaBoostClassifier(n_estimators=100)
    classifier = GradientBoostingClassifier(n_estimators=200, 
                                            subsample=0.8, 
                                            learning_rate=0.15,
                                            random_state=2)
    
    trainlabels, trainfeatures = cPickle.load(open('train_features.p', 'rb'))
    
    shuffleCrossValidation(trainlabels, trainfeatures, classifier)
    
    #testlabels, testfeatures = cPickle.load(open('test_features.p', 'rb'))
    
    #trainAndPredict(trainlabels, trainfeatures, testlabels, testfeatures, classifier)
