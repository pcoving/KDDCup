import csv
import cPickle
import itertools
import random
import sys

class Author():
    def __init__(self):
        self.papers = []
        self.journals = []
        self.conferences = []

class Paper():
    def __init__(self):
        self.authors = []
        self.year = None
        self.journalid = None
        self.conferenceid = None
        self.title = None
        self.affiliation = None
        self.paperrank = None

class Venue():
    def __init__(self):
        self.papers = []

def loadAuthorsPapers(path='dataRev2/'):
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
            # no need for this yet..
            papers[paperid].affiliation = affiliation 
            
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
                # no need for this yet...
                papers[paperid].title = title 
            except KeyError:
                notfound += 1

    for paper in papers.values():
        # remove duplicate authors
        paper.authors = list(set(paper.authors))

    # some (~500) papers don't have author information? could be an artifact of how the data was generated, but only a tiny fraction...
    #print 'unable to find ' + str(notfound) + ' papers'
    
    print 'done.'
    return authors, papers

def loadVenues(authors, papers):
    
    venues = {}
    assert False # broken...
    '''
    for paper in papers.values():
        jid = paper.journalid
        if jid > 0:
            if jid not in journals:
                journals[jid] = Journal()
            for aid in paper.authors:
                journals[jid].authors.append(aid)
                authors[aid].journals.append(jid)
    '''
    return venues

def csvGenerator(mode, path='dataRev2/'):
    
    if mode == 'train':
        with open(path + 'Train.csv') as csvfile:
            reader = csv.reader(csvfile)
            reader.next() # skip header
            for authorid, confirmedids, deletedids in reader:
                authorid = int(authorid)
                confirmedids = [int(id) for id in confirmedids.split(' ')]
                deletedids = [int(id) for id in deletedids.split(' ')]
                yield authorid, confirmedids + deletedids

    elif mode == 'test':
        with open(path + 'Valid.csv') as csvfile:
            reader = csv.reader(csvfile)
            reader.next() # skip header
            for authorid, paperids in reader:
                authorid = int(authorid)
                paperids = [int(id) for id in paperids.split(' ')]
                yield authorid, paperids

    else:
        print 'mode must be "train" or "test"'
        raise ValueError


def saveFeature(feature, name, mode):
    filename = name + '.' + mode
    print 'saving feature to', filename, '...'
    cPickle.dump(feature, open(filename,'wb'))

def labels(mode='train', path='dataRev2/'):
    
    labels = []
    if mode == 'train':
        with open(path + 'Train.csv') as csvfile:
            reader = csv.reader(csvfile)
            reader.next() # skip header
            for authorid, confirmedids, deletedids in reader:
                mylabels = []
                authorid = int(authorid)
                confirmedids = [int(id) for id in confirmedids.split(' ')]
                deletedids = [int(id) for id in deletedids.split(' ')]
                for cid in confirmedids:
                    mylabels.append(1)  # 1 = confirmed
                for did in deletedids:
                    mylabels.append(0)  # 0 = deleted
                labels.append(mylabels)

    elif mode == 'test':
        for authorid, paperids in csvGenerator(mode=mode, path=path):
            labels.append([authorid, paperids])

    else:
        print 'mode must be "train" or "test"'
        raise ValueError
            
    saveFeature(labels, name='labels', mode=mode) 

def nauthors(papers, authors, mode='train', path='dataRev2/'):
    '''
    Number of authors on paper
    '''

    print 'generating nauthors feature...'

    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        features.append([len(papers[pid].authors) for pid in paperids])

    saveFeature(features, name='nauthors', mode=mode) 

def npapers(papers, authors, mode='train', path='dataRev2/'):
    '''
    Number of papers written by author
    '''

    print 'generating npapers feature...'

    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        features.append([len(authors[authorid].papers) for pid in paperids])

    saveFeature(features, name='npapers', mode=mode) 

def year(papers, authors, mode='train', path='dataRev2/'):
    '''
    Year paper was written
    '''

    print 'generating year feature...'

    features = []
    for authorid, paperids in csvGenerator(mode=mode):
        features.append([papers[pid].year for pid in paperids])

    saveFeature(features, name='year', mode=mode) 

def nsamevenue(papers, authors, mode='train', path='dataRev2/'):
    '''
    Number of times author has published at venue
    '''
    
    print 'generating nsamevenue feature...'
    
    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        myfeatures = []
        for pid in paperids:
            if papers[pid].conferenceid > 0:
                myfeatures.append([papers[pid2].conferenceid for pid2 in authors[authorid].papers].count(papers[pid].conferenceid))
            elif papers[pid].journalid > 0:
                myfeatures.append([papers[pid2].journalid for pid2 in authors[authorid].papers].count(papers[pid].journalid))
            else:
                myfeatures.append(-1)

        features.append(myfeatures)

    saveFeature(features, name='nsamevenue', mode=mode) 

def nattrib(papers, authors, mode='train', path='dataRev2/'):
    '''
    Number of times paper has been attributed to author
    '''
    
    print 'generating nattrib feature...'

    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        myfeatures = []
        for pid in paperids:
            myfeatures.append(authors[authorid].papers.count(pid))
        features.append(myfeatures)
        
    saveFeature(features, name='nattrib', mode=mode) 

def paperrank(papers, authors, mode='train', path='dataRev2/', beta=0.8, nwalks=50):
    '''
    Personalized page rank
    '''
    
    print 'generating paperrank feature...'
    
    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
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
                        
        features.append([papers[pid].paperrank for pid in paperids])
        
    saveFeature(features, name='paperrank', mode=mode) 

    '''
    ranked = [label for pr, label in sorted(zip(myfeatures, mylabels), key=lambda tup: -tup[0])]
            score += scoreAuthor(ranked)
            count += 1
            #print score/count
    '''

def globalpaperrank(papers, authors, mode='train', path='dataRev2/'):
    '''
    On the undirected paper-author graph, the page rank is simply the degree
    '''
    
    print 'generating globalpaperrank feature...'
    
    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        myfeatures = []
        for paperid in paperids:
            globalpaperrank = 0
            for aid in papers[paperid].authors:
                if aid != authorid:
                    for pid in authors[aid].papers:
                        if pid != paperid:
                            globalpaperrank += 1
                            
            myfeatures.append(globalpaperrank)
        features.append(myfeatures)
    
    saveFeature(features, name='globalpaperrank', mode=mode) 


def ncoauthor(papers, authors, mode='train', path='dataRev2/'):
    '''
    Number of times author has published with coauthors on paper
    '''
    
    print 'generating ncoauthor feature...'

    features = []
    for authorid, paperids in csvGenerator(mode=mode, path=path):
        myfeatures = []
        for paperid in paperids:
            ncoauthor = 0
            for coauthorid in papers[paperid].authors:
                if coauthorid != authorid:
                    for pid in authors[coauthorid].papers:
                        if pid != paperid and authorid in papers[pid].authors:
                            ncoauthor += 1
            myfeatures.append(ncoauthor)
        features.append(myfeatures)
    
    saveFeature(features, name='ncoauthor', mode=mode) 
    
if __name__ == '__main__':
        
    authors, papers = loadAuthorsPapers()
    labels(mode='train')
    nauthors(papers, authors, mode='train')
    npapers(papers, authors, mode='train')
    year(papers, authors, mode='train')
    nsamevenue(papers, authors, mode='train')
    nattrib(papers, authors, mode='train')
    globalpaperrank(papers, authors, mode='train')
    paperrank(papers, authors, mode='train')
    ncoauthor(papers, authors, mode='train')
