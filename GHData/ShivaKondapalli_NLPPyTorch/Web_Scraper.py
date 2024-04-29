import requests
from bs4 import BeautifulSoup


def get_data(url):
    """request url and prase data appropriately

    :param url: the url of the site to get data from
    :type string

    :return list of names from the website
    """

    request = requests.get(url)

    soup = BeautifulSoup(request.text, "lxml")

    ol_tags = soup.find_all('ol')

    names_list = []

    for li_tags in ol_tags:
        for names in li_tags:
            names_list.append(names.text)

    return names_list


# Following is to check if everything works all right
# call method and get the data
def main():
    girls_list = get_data('https://www.verywellfamily.com/top-1000-baby-girl-names-2757832')
    boys_list = get_data('https://www.verywellfamily.com/top-1000-baby-boy-names-2757618')

    # sex to dictionary mapping
    sex_to_name = dict()

    sex_to_name['Girl'] = girls_list
    sex_to_name['Boy'] = boys_list

    # check if lists, length and dict is okay
    print(girls_list)
    print(len(girls_list))
    print(boys_list)
    print(len(boys_list))

    print(sex_to_name)
    print(len(sex_to_name))

    with open('data/Gender/Boys.txt', 'w') as f:
        for name in boys_list:
            f.write('%s\n' % name)

    with open('data/Gender/Girls.txt', 'w') as f:
        for name in girls_list:
            f.write('%s\n' % name)


if __name__ == '__main__':
    main()
