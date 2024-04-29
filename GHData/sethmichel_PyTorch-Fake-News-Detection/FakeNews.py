# Seth Michel

import json
import requests                     # establish an https connection to rapid api for hoaxy
from newspaper import Article       # downloads the article

# uses newspaper to download and parse the article then returns it
# called from main()
def DownloadArticle(url):
    myArticle = Article(url, language = "en")
    myArticle.download()
    myArticle.parse()

    return myArticle


# > hoaxy grabs 100 related articles and gives them scores and other things
#   this is used is calculating my articles score
# > called from main
def GetRelatedArticles(hoaxyHeaders, title):
    queryString = {"sort_by": "relevant", "use_lucene_syntax": "true", "query": title}
    hoaxyUrl = "https://api-hoaxy.p.rapidapi.com/articles"

    response = requests.request("GET", hoaxyUrl, headers = hoaxyHeaders, params = queryString)
    usableResponse = json.loads(response.content)   # deserialized response so I can use it

    return usableResponse


# > hoxy gets 20 latest realted articles and gives them scores and other things
#   used in calculating main article score
# > called from main
def GetLatestArticles(hoaxyHeaders, title):
  queryString = {"sort_by": "relevant", "use_lucene_syntax": "true", "query": title, "past_hours":"48"}
  hoaxyUrl = "https://api-hoaxy.p.rapidapi.com/LatestArticles"

  response = requests.request("GET", hoaxyUrl, headers = hoaxyHeaders, params = queryString)
  usableResponse = json.loads(response.content)   # deserialized response so I can use it

  return usableResponse


# returns fake news status: if article site is known for political bias, fake news, factchecking, or satire
# Returns nothing if site is credible
def SiteCredibilityCheck(title, adverifaiHeader, passedUrl):
    adverifaiUrl = "https://adverifai-api.p.rapidapi.com/source_check"
    queryString = {"url": passedUrl}

    response = requests.request("GET", adverifaiUrl, headers = adverifaiHeader, params = queryString)
    
    # check: internal server error. If sites credible then response is blank
    if (response.status_code == 500):   
        print("Error: internal server error: code 500")
        return "Internal server error: code 500"

    if (response.text == "{}\n"):
        print("credible site")
        return "credible site"
    
    # dict with the result but in a long string of different languages, split grabs the english only
    responseJson = json.loads(response.text)
    usableResponse = (responseJson["fakeDescription"].split("."))[0]

    print(usableResponse)
    
    return usableResponse


def getMean(stats):
    counter = 0
    totalScore = 0

    for item in stats:
        if (item[0] == "claim"):
            counter = counter + 1
            totalScore = totalScore + int(item[1])

    if (counter > 0):
      return totalScore / counter
    else:
      return 0


# likely true, possibly true, possibly false, likely false, very likely false
# scorelist = list of tuples, mean = int, buckets = list
def categorizeScore(scoreList, mean, buckets):
    for score in scoreList:
        if (score[1] < mean - mean * .05):
            buckets[0] += 1   # likely true
        elif (score[1] > mean - (mean * .1) and score[1] < mean + (mean * .05)):
            buckets[1] += 1   # possibly true
        elif (score[1] > mean + (mean) and score[1] < mean + (mean * .05)):
            buckets[2] += 1   # "possibly false"
        elif (score[1] > mean + (mean * .05) and score[1] < mean + (mean * .1)):
            buckets[3] += 1   # "likely false"
        else:
            buckets[4] += 1   # "very likely false"
    
    return buckets


# args is relatedArticleStats then latestArticlesStats. this function does the
# > same code for both though
def printScores(*args):
    buckets = [0,0,0,0,0]
    mean = 0

    for i in range(0, len(args)):
        mean = getMean(args[i])
        buckets = categorizeScore(args[i], mean, buckets)

    return buckets


def GatherData(url, hoaxyHeaders, adverifaiHeaders):  
    relatedArticleStats = []
    latestArticlesStats = []

    article = DownloadArticle(url)
    relatedArticles = GetRelatedArticles(hoaxyHeaders, article.title)
    latestArticles = GetLatestArticles(hoaxyHeaders, article.title)
    siteCredibility = SiteCredibilityCheck(article.title, adverifaiHeaders, url)

    for i in range(0, 100):
        relatedArticleStats.append((relatedArticles["articles"][i]["site_type"], relatedArticles["articles"][i]["score"]))

    if (latestArticlesStats != []):
        for i in range(0, 20):
            latestArticlesStats.append((latestArticles["articles"][i]["site_type"], latestArticles["articles"][i]["score"]))

    return [article, relatedArticleStats, latestArticlesStats, siteCredibility]

# political bias, fake news, factchecking, or satire
# likely true, possibly true, possibly false, likely false, very likely false
def FinalTruthChooser(scoreList, siteCredibility):
    chooser = 0
    result = ""
    scoreList.sort()   # ascending order

    if (siteCredibility == "credible site"):
        chooser = 4
    elif (siteCredibility == "political bias" or siteCredibility == "regularly imprecise"):
        chooser = 3
    elif (siteCredibility == "fake news" or siteCredibility == "pseudo science, conspiracy"):
        chooser = 1
    elif (siteCredibility == "factchecking"):
        chooser = 3
    elif (siteCredibility == "satire"):
        chooser = 0

    # logic
    # if chooser = 4: if bucket 4 is > 34, result stands
    # if chooser = 0: if bucket 2+3+4 is > 69, result += 1
    # if chooser = 1, 2, or 3: sum of buckets above, sum buckets below. above - below. 
    # > perfect dist is 0. Higher = more above, lower = more below. result += 1 if 
    # > above 49, result -= 1 if below -49
    result = chooser
    if (chooser == 4 and scoreList[4] > 34):
        pass

    elif (chooser == 0):
        if (scoreList[2] + scoreList[3] + scoreList[4] > 69):
            result += 1

    else:
        # get sum of buckets below and sum of buckets above chooser bucket
        below = 0
        above = 0
        for i in range(0, chooser):
            below += scoreList[i]
        for i in range(4, chooser - 1, -1):
            above += scoreList[i]

        distribution = above - below
        if (distribution > 49):
            result += 1
        elif (distribution < -49):
            result -= 1

    return result


def main():
    relatedArticleStats = []
    latestArticlesStats = []

    adverifaiHeaders = {"x-rapidapi-host": "adverifai-api.p.rapidapi.com",
                         "x-rapidapi-key": ""}

    hoaxyHeaders = {"x-rapidapi-host": "api-hoaxy.p.rapidapi.com",
                     "x-rapidapi-key": ""}

    urls = ["https://www.cnn.com/2020/11/25/politics/trump-outcasts-biden-public-servants/index.html",
            "https://sports.yahoo.com/us-soccers-new-president-and-ceo-must-deliver-on-these-pressing-challenges-162356307.html",
            "https://nypost.com/2020/03/28/mercenary-nurses-get-big-bucks-to-work-in-nyc-during-coronavirus-outbreak/",
            "https://politics.theonion.com/trump-announces-plan-to-retrain-nation-s-3-million-unem-1842531861",
            "https://www.infowars.com/posts/sunday-live-trump-excoriates-doj-fbi-missing-in-action-over-election-fraud/"]

    articleData = GatherData(urls[3], hoaxyHeaders, adverifaiHeaders)

    article = articleData[0]
    relatedArticlesStats = articleData[1]
    latestArticlesStats = articleData[2]
    siteCredibility = articleData[3]

    scoreList = printScores(relatedArticleStats, latestArticlesStats)

    finalAnswer = FinalTruthChooser(scoreList, siteCredibility)
    printStr = ": \"" + article.title + "\""

    # likely true, possibly true, possibly false, likely false, very likely false
    if (finalAnswer == 4):
        printStr = "likely true" + printStr
    elif (finalAnswer == 3):
        printStr = "possibly true" + printStr
    elif (finalAnswer == 2):
        printStr = "possibly false" + printStr
    elif (finalAnswer == 1):
        printStr = "likey false" + printStr
    else:
        printStr = "very likely false" + printStr

    print(printStr)


main()