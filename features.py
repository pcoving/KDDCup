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

def buildTrainFeatures(authors, papers, path='DataRev2/'):
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

def buildTestFeatures(authors, papers, path='DataRev2/'):
    labels = [] # [authorid, paperid] is label for test data (needed for submission)...
    features = []
        
    # build features for training set...
    with open('DataRev2/Valid.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for authorid, paperids in reader:
            authorid = int(authorid)
            paperids = [int(id) for id in paperids.split(' ')]
            
            myfeatures, mylabels = [], []
            for pid in paperids:
                mylabels.append(pid) 
                myfeatures.append(generateFeatures(pid, authorid, papers, authors))
            
            features.append(myfeatures)    
            labels.append([authorid, mylabels])
    
    return labels, features

def saveFeatures(labels, features, filename):
    print 'saving features to', filename, '...'
    cPickle.dump([labels, features], open(filename,'wb'))
    
def generateFeatures(paperid, authorid, papers, authors):
    '''
    so far:
    nauthors = Number of Authors on Paper
    npapers = Number of Papers by Author
    year = Year of Paper
    njournal = Number of Author's Papers in Journal
    nconference = Number of Author's Papers in Conference
    ncoauthor = Number of Author's Papers with Coauthors
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
    labels, features = buildTrainFeatures(authors, papers)
    saveFeatures(labels, features, 'train_features.p')

    labels, features = buildTestFeatures(authors, papers)
    saveFeatures(labels, features, 'test_features.p')
