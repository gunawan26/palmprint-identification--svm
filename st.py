import requests
import urllib.request
from bs4 import BeautifulSoup
import time

# 16 0 555 1 xx
nim = 1707531001
url = 'https://salut.unud.ac.id/mainpage/mahasiswa/ucapan_salut/'
response = requests.get(url+"{}".format(nim))



soup = BeautifulSoup(response.text, 'html.parser')

count_err = 0

while True:
    try:
        response = requests.get(url+"{}".format(nim))
        soup = BeautifulSoup(response.text, 'html.parser')

        res = soup.select('.profil-text td')
        nm = res[1].string

        print("nama : {} nim : {}".format(res[1],res[3]))
        nim+=1
        time.sleep(1)
    except:
        count_err +=1

        if count_err == 10:
            break

        pass
    
# for i in range(36,len(soup.findAll('a'))+1): #'a' tags are for links
#     one_a_tag = soup.findAll('a')[i]
#     link = one_a_tag['href']
#     download_url = 'http://web.mta.info/developers/'+ link
#     urllib.request.urlretrieve(download_url,'./'+link[link.find('/turnstile_')+1:]) 

#     time.sleep(1) #pause the code for a sec

