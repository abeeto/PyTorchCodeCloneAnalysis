class CloneFragment:
    def __init__(self, repoName, userName, fileName, startline, endline, pcid):
        self.repoName = repoName
        self.userName  = userName
        self.fileName = fileName
        self.startline = startline
        self.endline = endline
        self.pcid = pcid

class Bucket:
    def __init__(self, bucketId, nlines, similarity):
        self.bucketId = bucketId
        self.nlines = nlines
        self.similarity = similarity
        self.cloneFrags = []

    def add_cloneFrag(self, cloneFragment):
        self.cloneFrags.append(cloneFragment)

class User:
    def __init__(self, userName, followers, followees, stars, repos, forks):
        self.userName = userName
        self.followers = followers
        self.followees = followees
        self.stars = stars
        self.repos = repos
        self.forks = forks
        