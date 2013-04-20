import numpy as np
import csv
import cPickle

class Author():
    def __init__(self):
        self.papers = []
        
class Paper():
    def __init__(self):
        self.authors = []
        self.year = None
        self.journalid = None
        self.conferenceid = None

def loadAuthorsAndPapers(path='DataRev2/'):
    
    authors = {}
    papers = {}
    with open(path + 'PaperAuthor.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for paperid, authorid, name, affiliation in reader:
            paperid, authorid = int(paperid), int(authorid)
            if paperid not in papers:
                papers[paperid] = Paper()
            papers[paperid].authors.append(int(authorid))
            
            if authorid not in authors:
                authors[authorid] = Author()
            authors[authorid].papers.append(int(paperid))

    notfound = 0
    with open(path + 'Paper.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for paperid, title, year, conferenceid, journalid, keyword in reader:
            paperid = int(paperid)
            try:
                papers[paperid].year = int(year)
                papers[paperid].conferenceid = int(conferenceid)
                papers[paperid].journalid = int(journalid)
            except KeyError:
                notfound += 1

    print 'unable to find ' + str(notfound) + ' papers '

    return authors, papers

def buildFeatures(authors, papers, path='DataRev2/'):
    # recreates features from Kaggle benchmark without SQL    

    labels = []
    features = []
        
    # build features for training set...
    with open('DataRev2/Train.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for authorid, confirmedids, deletedids in reader:
            authorid = int(authorid)
            confirmedids = [int(id) for id in confirmedids.split(' ')]
            deletedids = [int(id) for id in deletedids.split(' ')]

            myfeatures, mylabels = [], []
            for cid in confirmedids:
                mylabels.append(1)  # 1 = confirmed
                myfeatures.append(generateFeatures(cid, authorid, papers, authors))
            for did in deletedids:
                mylabels.append(0)  # 0 = deleted
                myfeatures.append(generateFeatures(did, authorid, papers, authors))
            
            features.append(myfeatures)    
            labels.append(mylabels)
    
    return labels, features

def saveFeatures(labels, features, filename='train_features.p'):
    print 'saving features...'
    cPickle.dump([labels, features], open(filename,'wb'))

    
def generateFeatures(paperid, authorid, papers, authors):
    '''
    so far:
    Number of Authors on Paper
    Year of Paper
    Number of Papers by Author
    Number of Author's Papers in Journal
    Number of Author's Papers in Conference
    Number of Author's Papers with Coauthors
    '''
    nauthors = len(papers[paperid].authors)
    npapers = len(authors[authorid].papers)
    year = papers[paperid].year
    if papers[paperid].conferenceid > 0:
        nconference = 0
        for pid in authors[authorid].papers:
            if papers[pid].conferenceid == papers[paperid].conferenceid:
                nconference += 1
    else:
        nconference = -1  # indicates no conference info in data
    if papers[paperid].journalid > 0:
        njournal = 0
        for pid in authors[authorid].papers:
            if papers[pid].journalid == papers[paperid].journalid:
                njournal += 1
    else:
        njournal = -1  # indicates no journal info in data

    ncoauthor = 0
    for aid in papers[paperid].authors:
        if aid != authorid:
            for pid in authors[aid].papers:
                if pid != paperid:
                    ncoauthor += 1

    features = [npapers, nauthors, year, 
                nconference, njournal, ncoauthor]
    return features

if __name__ == '__main__':
    authors, papers = loadAuthorsAndPapers()
    labels, features = buildFeatures(authors, papers)
    saveFeatures(labels, features)
