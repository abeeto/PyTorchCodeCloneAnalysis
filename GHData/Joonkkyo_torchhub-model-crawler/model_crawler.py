#tensorflowhub crawler
from selenium import webdriver
from bs4 import BeautifulSoup
import argparse
import pandas as pd 
import time
import wget
import os


def find_model_list(type_link, driver_dir):
    # driver = webdriver.Chrome('/Users/mzc01-jkseo/Downloads/chromedriver')
    driver = webdriver.Chrome(driver_dir)
    type_find = type_link.replace('-',' ').capitalize()
    driver.implicitly_wait(1)
    driver.get('https://tfhub.dev/s?module-type=' + type_link)
    time.sleep(8)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    soup_final = soup.find('mat-sidenav-content').find('div').find('ng-component').find('div').find('div', 'content').find('product-list').find_all('list-item-wrapper')

    titles = []
    dates = []
    architectures = []
    datasets = []
    publishers = []
    links_1, links_2 = [], [] 

    for s in list(soup_final):
        if type_find in str(s):
            title = str(s).split('<!-- --></div>')[1].split('">')[2].replace('</a>','').replace(' ','')
            if 'Publisher:' in str(s): 
                publisher = str(s).split('Publisher:')[1].split('">')[1].split('</a')[0].replace(' ','')
                if publisher == "Rishit Dagli":
                    publisher = "rishit-dagli"
                elif publisher == "Sayak Paul.":
                    publisher = "sayakpaul"
            else: 
                publisher = ''
            if 'Updated: ' in str(s):
                date = str(s).split('Updated:')[1].split('">')[1].split('</a')[0].split('</span>')[0].replace(' ','')

            if 'Architecture:' in str(s): 
                architecture = str(s).split('Architecture:')[1].split('">')[1].split('</a')[0].replace(' ','')
            else: 
                architecture = ''
            if 'Dataset:' in str(s): 
                dataset = str(s).split('Dataset:')[1].split('">')[1].split('</a')[0].replace(' ','')
            else: 
                dataset = ''

            titles.append(title)
            dates.append(date)
            publishers.append(publisher)
            architectures.append(architecture)
            datasets.append(dataset)
            link_1 = "https://storage.googleapis.com/tfhub-modules/" + publisher.lower() + '/' + title + "/1.tar.gz"
            link_2 = "https://storage.googleapis.com/tfhub-modules/" + publisher.lower() + '/' + title + "/2.tar.gz"

            links_1.append(link_1)
            links_2.append(link_2)

    res = pd.DataFrame({'Title': titles, 'Dates': dates, 'Publisher': publishers, 'Architecture': architectures, 'Dataset':datasets, 'Link_1': links_1, 'Link_2': links_2}).drop_duplicates()
    res.to_csv("./model_crawling_list/" + type_link + '.csv', index=False)
    return titles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract all images from excel files.')
    parser.add_argument('--tag', required=True, help='Directory of the files')
    parser.add_argument('--chromedriver', required=True, help='Directory of ChromeDriver')
    args = parser.parse_args()
    model_tag = args.tag
    
    title_list = find_model_list(model_tag, args.chromedriver)
    csv_file = pd.read_csv("./model_crawling_list/" + model_tag + ".csv")
    df_1 = pd.DataFrame(csv_file, columns=['Link_1'])
    df_2 = pd.DataFrame(csv_file, columns=['Link_2'])

    link_list_1 = df_1.values.tolist()
    link_list_2 = df_2.values.tolist()

    link_arr = []
    for link in link_list_1:
        link_arr.append(link[0])

    for link in link_list_2:
        link_arr.append(link[0])

    if not os.path.exists(model_tag):
        os.mkdir(model_tag)

    count = 0
    save_dir = "/Users/mzc01-jkseo/Documents/" + model_tag
    
    for i in range(len(link_arr)):
        try:
            output_name = title_list[i * 2] + ".tar.gz"
            os.system(f"wget -P {save_dir} -O {output_name} {link_arr[i]}")
            count += 1
        except:
            print(f"{link_arr[i]} is not valid.")
            continue
        
    print("total :", count)
