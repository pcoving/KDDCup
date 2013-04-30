import csv
import cPickle
import itertools
import random

class Author():
    def __init__(self):
        self.papers = []

class Paper():
    def __init__(self):
        self.authors = []
        self.year = None
        self.journalid = None
        self.conferenceid = None
        self.title = None
        self.affiliation = None
        self.paperrank = None

def loadAuthorsAndPapers(path='dataRev2/'):
    print 'loading authors and papers...'

    authors = {}
    papers = {}
    with open(path + 'PaperAuthor.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for paperid, authorid, name, affiliation in reader:
            paperid, authorid = int(paperid), int(authorid)
            if paperid not in papers:
                papers[paperid] = Paper()
            papers[paperid].authors.append(authorid)
            papers[paperid].affiliation = affiliation # no need for this yet..
            
            if authorid not in authors:
                authors[authorid] = Author()
            authors[authorid].papers.append(paperid)
            
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
                papers[paperid].title = title # no need for this yet...
            except KeyError:
                notfound += 1

    #print 'unable to find ' + str(notfound) + ' papers'
    print 'done.'
    return authors, papers

def calcPersonalizedPaperRank(authorid, authors, papers, beta=0.8, nwalks=50):
    
    for paper in papers.values():
        paper.paperrank = 0
    for pid in authors[authorid].papers:
        for walk in range(nwalks):
            current_pid = pid
            #papers[current_pid].paperrank += 1
            if len(papers[current_pid].authors) > 1:
                while (random.random() < beta):   # will pass with probability beta...
                    random_aid = authorid
                    while (random_aid != authorid):
                        random_aid = random.choice(papers[current_pid].authors)
                    current_pid = random.choice(authors[random_aid].papers)
                    papers[current_pid].paperrank += 1

def buildTrainFeatures(authors, papers, path='dataRev2/'):
    
    labels = []
    features = []

    # build features for training set...
    conferences = {}
    journals = {}
    with open(path + 'Train.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for authorid, confirmedids, deletedids in reader:
            authorid = int(authorid)
            confirmedids = [int(id) for id in confirmedids.split(' ')]
            deletedids = [int(id) for id in deletedids.split(' ')]

            calcPersonalizedPaperRank(authorid, authors, papers)

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

def buildTestFeatures(authors, papers, path='dataRev2/'):
    labels = [] # [authorid, paperid] is label for test data (needed for submission)...
    features = []

    # build features for training set...
    with open(path + 'Valid.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for authorid, paperids in reader:
            authorid = int(authorid)
            paperids = [int(id) for id in paperids.split(' ')]
            
            calcPersonalizedPaperRank(authorid, authors, papers)
            
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
    nsamejournal = Number of Author's Papers in Journal
    nsameconference = Number of Author's Papers in Conference
    nattrib = Number of times Paper has been attributed to Author
    globalpaperrank = Degree of Paper on Paper/Author graph (actually pagerank on undirected graph!)
    ncoauthor = Number of Author's Papers with Coauthors
    
    how to create continuous, graph-based analogs for the various "count" features?
    '''
    nauthors = len(papers[paperid].authors)
    npapers = len(authors[authorid].papers)
    
    if papers[paperid].conferenceid > 0:
        nsameconference = 0
        for pid in authors[authorid].papers:
            if papers[pid].conferenceid == papers[paperid].conferenceid:
                nsameconference += 1
    else:
        nsameconference = -1  # indicates no conference info in data
    if papers[paperid].journalid > 0:
        nsamejournal = 0
        for pid in authors[authorid].papers:
            if papers[pid].journalid == papers[paperid].journalid:
                nsamejournal += 1
    else:
        nsamejournal = -1  # indicates no journal info in data
    
    globalpaperrank = 0
    for aid in papers[paperid].authors:
        if aid != authorid:
            for pid in authors[aid].papers:
                if pid != paperid:
                    globalpaperrank += 1

    ncoauthor = 0
    for coauthorid in papers[paperid].authors:
        if coauthorid != authorid:
            for pid in authors[coauthorid].papers:
                if pid != paperid and authorid in papers[pid].authors:
                    ncoauthor += 1

    nattrib = 0
    for pid in authors[authorid].papers:
        if pid == paperid:
            nattrib += 1

    features = [npapers, nauthors, papers[paperid].year, 
                nsameconference, nsamejournal, ncoauthor,
                nattrib, globalpaperrank, papers[paperid].paperrank]

    return features

if __name__ == '__main__':
    authors, papers = loadAuthorsAndPapers()
    labels, features = buildTrainFeatures(authors, papers)
    saveFeatures(labels, features, 'train_features.p')
    
    #labels, features = buildTestFeatures(authors, papers)
    #saveFeatures(labels, features, 'test_features.p')
