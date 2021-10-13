from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import os
import base64
from PIL import Image
import tweepy
import json
import re
from authAPI import *
import datetime
import shutil

app = Flask(__name__)

finwizUrl = 'https://finviz.com/quote.ashx?t='
vader = SentimentIntensityAnalyzer()


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods = ['POST'])
def getVals():

    raw_search_str = request.form['stock']

    search_str = re.sub('[^A-Za-z0-9]+', '', raw_search_str)
    print("Search Term is : " + search_str)
    ngtv_lst = []
    pstv_lst = []

    url = finwizUrl + search_str

    urlCode = validateURL(url)
    print(" URL Connectivity Status : " + str(urlCode))

    if len(search_str) > 0:
        dt = datetime.datetime.now()
        dt_str = dt.strftime('%m%d%Y%H%M%S')
        twtrImgEncd, twtrFlg, twtrInfo, ngtv_lst,pstv_lst = twitterSA(search_str,dt_str)
        newsImgEncd, newsFlg, newsInfo = newsIOSA(search_str,dt_str)
        stockImg, stockFlg, stockInfo = stockSA(search_str,dt_str)
    else:
        dt = datetime.datetime.now()
        dt_str = dt.strftime('%m%d%Y%H%M%S')
        twtrFlg = False
        newsFlg = False
        stockFlg = False

    with open('user_input.txt', 'a') as f:
        print(dt.strftime('%m/%d/%Y %H:%M:%S')+"  :  "+ raw_search_str + "  -> Twitter Flag: " + str(twtrFlg) + " ,News Flag: " + str(newsFlg) + " ,Stock Flag: " + str(stockFlg), file=f)


    if (not twtrFlg) & (not newsFlg) & (not stockFlg) or len(search_str)==0:
        badSearch = encodeImage('Invalid_Search2.jpeg',dt_str)
        return render_template('index.html',
                               badSearch_img_data=badSearch.decode('utf-8'),search_str=search_str)
    else:
        return render_template('index.html',
                               stk_img_data=stockImg.decode('utf-8'),stockInfo=stockInfo,
                               twtr_img_data=twtrImgEncd.decode('utf-8'),twtrInfo=" - "+twtrInfo,ngtv_lst=ngtv_lst,pstv_lst=pstv_lst,
                                   pstvStr="Most Positive Tweets", ngtvStr="Most Negative Tweets",
                               newsio_img_data=newsImgEncd.decode('utf-8'),newsInfo=" - "+newsInfo,
                               search_str= search_str)


def validateURL(url):
    try:
        ureq = Request(url, headers={'user-agent': 'my-app'})
        response = urlopen(ureq)
        return response.code in range(200, 209)
    except Exception:
        return False

"""
stockSA Function does web scrapping from https://finviz.com/quote.ashx?t= and 
 get the News headlines alone and
  does entiment analysis using Vader SentimentIntensityAnalyzer 
    and finally plots on the scores <Bar Plot>
"""
def stockSA (search_str,dt_str):

    url = finwizUrl + search_str
    urlCode = validateURL(url)

    print(" URL Connectivity Status : " + str(urlCode))

    if urlCode:
        parsed_data = []
        news_tables = {}

        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, 'html')
        news_table = html.find(id='news-table')
        news_tables[search_str] = news_table

        for ticker, news_table in news_tables.items():
            for row in news_table.find_all('tr'):
                title = row.a.text
                date_data = row.td.text.split(' ')

                if len(date_data) == 1:
                    time = date_data[0]
                else:
                    time = date_data[1]
                    date = date_data[0]

                parsed_data.append([ticker, date, time, title])

        df = pd.DataFrame(parsed_data, columns=['stock', 'date', 'time', 'title'])


        f = lambda title: vader.polarity_scores(title)['compound']

        df['compound'] = df['title'].apply(f)
        df['date'] = pd.to_datetime(df['date']).dt.date

        # mean_df = df.groupby(['ticker', 'date'])['compound'].mean()
        mean_df = df.groupby(['stock', 'date']).mean()

        mean_df = mean_df.unstack()
        mean_df = mean_df.xs('compound', axis="columns").transpose()
        mean_df = mean_df.sort_values(by= 'date', ascending= False)
        mean_df = mean_df.head(20)

        img_name = dt_str + search_str+"_Stock.jpg"
        print("Image Name is :"+ img_name)

        mean_df.plot(kind='bar')
        plt.savefig(img_name)
        stockImgEncd = encodeImage(img_name,dt_str)
        stockFlg = True if df.shape[0] >0 else False

        return (stockImgEncd, stockFlg,'')

    else:
        return (encodeImage('Invalid_Search.jpeg',dt_str), False, '')

"""
twitterSA() pulls data from twitter feed using Hashtag and 
  does entiment analysis using Vader SentimentIntensityAnalyzer 
    and finally plots on the scores <Scatter Plot>
"""
def twitterSA(search_str,dt_str):

    img_name = dt_str + search_str + "_Twtr.jpg"
    print("Image Name is :" + img_name)

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    number_of_tweets = 500
    tweets = []
    likes = []
    time = []
    ngtv_lst =[]
    pstv_lst =[]

    for i in tweepy.Cursor(api.search, q='#'+search_str, tweet_mode='extended').items(number_of_tweets):
        tweets.append(i.full_text)
        likes.append(i.favorite_count)
        time.append(i.created_at)



    preTwitterDF = pd.DataFrame({"tweets": tweets ,"likes": likes, "time": time})
    tweetPhr = " ".join(tweets)

    sentiment_dict = vader.polarity_scores(tweetPhr)
    cmp_scr = str(round(sentiment_dict['compound'] * 100, 2))
    print(sentiment_dict)

    twitterDF = preTwitterDF[~preTwitterDF["tweets"].str.contains('RT @', na=False)]

    lmd_vs = lambda txt: vader.polarity_scores(txt)['compound']
    twitterDF['compound'] = twitterDF['tweets'].apply(lmd_vs)

    score = ""
    if sentiment_dict.get('compound') >= 0.05:
        score = "Positive"
        print("Positive")
    elif sentiment_dict.get('compound') <= -0.0:
        score = "Negative"
        print("Negative")
    else:
        score = "Nuetral"
        print("Nuetral")
    print("sentiment_dict for Twitter is : "+ str(sentiment_dict))
    plt.figure(figsize=[8, 8])
    plt.scatter(twitterDF['time'], twitterDF['compound'], c=twitterDF['compound'], cmap='RdYlGn')
    #plt.title("Twitter is " + score.upper() + " for "+ search_str.upper(), fontsize=15)
    plt.xticks(rotation=90, ha="right")
    plt.colorbar()
    plt.savefig(img_name,dpi=200)

    twtr_ngtv = twitterDF.sort_values(by='compound').head(5)
    ngtv_lst = twtr_ngtv['tweets'].to_list()
    twtr_pstv = twitterDF.sort_values(by='compound', ascending=False).head(5)
    pstv_lst = twtr_pstv['tweets'].to_list()

    twtrImgEncd = encodeImage(img_name,dt_str)
    twtrFlg = True if twitterDF.shape[0] > 5 else False

    if twtrFlg:
        return(twtrImgEncd,twtrFlg, score,ngtv_lst,pstv_lst)
    else:
        return (encodeImage('No_Data.jpeg',dt_str), False,'Not Enough Data',ngtv_lst,pstv_lst)

"""
newsIOSA() pulls news articles from NewsIO API, Using Free account only get 10 articles are allowed maximum 
  does entiment analysis using Vader SentimentIntensityAnalyzer 
    and finally plots on the scores <Pie Plot>
"""
def newsIOSA(search_str,dt_str):
    img_name = dt_str + search_str + "_newsIO.jpg"
    url = "https://newsdata.io/api/1/news?apikey=" + newsIO_api_key + "&language=en&q=" + search_str
    urlCode = validateURL(url)

    if urlCode:
        resp = urlopen(url)
        data_json = json.loads(resp.read())
        result = data_json['results']

        links = []
        content = ""
        for rec in result:
            links.append(rec['link'])
            # print(rec['description'])
            content += str(rec['description']) + " " + str(rec['content']) + " "

        sentiment_dict = vader.polarity_scores(content)
        print("sentiment_dict for NewsIO-API is : "+ str(sentiment_dict))
        score = ""
        if sentiment_dict.get('compound') >= 0.05:
            score = "Positive"
            print("Positive")
        elif sentiment_dict.get('compound') <= -0.0:
            score = "Negative"
            print("Negative")
        else:
            score = "Nuetral"
            print("Nuetral")

        dictDF = pd.DataFrame(sentiment_dict.items(), columns=['Sentiment', 'Score'])
        sntntDF = dictDF.drop(dictDF[dictDF['Sentiment'] == 'compound'].index)

        plt.figure(figsize=[8, 6])
        my_colors = ['lightblue', 'lightgreen', 'silver']
        my_explode = (0, 0.1, 0)
        #plt.title("News Articles are " + score.upper()  + " for "+ search_str.upper(), fontsize=20)
        plt.rcParams['font.size'] = '16'
        plt.pie(sntntDF['Score'], labels=sntntDF['Sentiment'], shadow=True, autopct='%1.1f%%', startangle=15, colors=my_colors,explode=my_explode)
        plt.savefig(img_name,dpi=200)

        newsImgEncd = encodeImage(img_name,dt_str)
        newsFlg = True if sentiment_dict.get('compound') > 0.0 else False

        if newsFlg:
            return (newsImgEncd, newsFlg,score)
        else:
            return (encodeImage('Missing_Data.jpeg',dt_str), False,'Not Enough Data')

    else:
        return (encodeImage('Missing_Data.jpeg',dt_str), False, 'Not Enough Data')


"""
encodeImage() Takes Image as input and encodes into Base64
  this is used to render the HTML page dynamically by decoding base64 back to image
"""


def encodeImage(img_name,dt_str):
    images = ['No_Data.jpeg', 'Missing_Data.jpeg', 'Invalid_Search.jpeg', 'Invalid_Search2.jpeg']
    if img_name in images:
        newNm = dt_str + img_name
        print("New image name is "+ newNm)
        shutil.copy(img_name, newNm)
    else:
        newNm = img_name

    print("Encoding Image : " + newNm)

    im = Image.open(newNm)
    data = io.BytesIO()
    if im.mode in ("RGBA", "P"): im = im.convert("RGB")
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    im.close()
    os.remove(newNm)
    return encoded_img_data


"""
Main method
"""
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)