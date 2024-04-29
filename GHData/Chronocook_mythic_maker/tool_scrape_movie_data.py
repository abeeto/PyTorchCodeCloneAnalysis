"""
This script scrapes The Movie Database
for movie data to use as a training set for a neural net.
(https://developers.themoviedb.org/3)


API keys:
You'll need to apply for keys (it's easy) and put them in
API_keys.txt like:
[Keys]
API_V3 = 0oYKFlAbGuBe35QI7hO2AT7IDj8tTEtMOvVBh2rS
API_V4 = gH0VcY3hv8U1wCtJhSWa15ainzgNI2XJBttY69HV1YxcSBl8G2qw3mRU41QUD5PzfxK0aYx0GIcamKC

External Dependencies:
tmdbsimple (pip install tmdbsimple) https://github.com/celiao/tmdbsimple/

Usage:
scrape_movie_data.py --output ./

@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""


from uplog import log
import os
import configparser
import tmdbsimple as tmdb
import json
import time
from tqdm import tqdm


API_FILE = "./API_keys.txt"
API_V3_KEY = None
API_V4_KEY = None


def read_api_keys(key_filename):
    """
    :param key_filename: (str) name of the file with your API keys
    :return config_dict: (dict) a dictionary with keys (api_v3 and api_v4)
    """
    if os.path.isfile(key_filename):
        log.out.debug('API file ' + key_filename + ' exists!')
    else:
        log.out.warn('API file ' + key_filename + ' does not exist.')
        return
    config = configparser.ConfigParser(defaults=os.environ,
                                       interpolation=configparser.ExtendedInterpolation())
    log.out.info('Reading API file: ' + key_filename)
    config.read(key_filename)
    section_list = config.sections()
    config_dict = {}
    for section in section_list:
        config_dict[section] = {}
        options = config.options(section)
        for option in options:
            try:
                config_dict[option] = config.get(section, option)
                if config_dict[option] == 'None':
                    config_dict[option] = None
                if config_dict[option] == -1:
                    log.out.debug('Skip: %s' % option)
            except SyntaxError:
                log.out.warn('Exception on %s!' % option)
                config_dict[option] = None
    global API_V3_KEY
    global API_V4_KEY
    API_V3_KEY = config_dict["api_v3"]
    API_V4_KEY = config_dict["api_v4"]


def request_v3_by_year(year, language="en", keys=None):
    possible_keys = ["id", "genre_ids", "title", "original_title",
                     "video", "original_language", "release_date",
                     "vote_count", "vote_average", "popularity",
                     "adult", "poster_path", "backdrop_path", "overview"]
    retry_limit = 100
    year_of_movies = []

    if keys is None:
        requested_keys = possible_keys
    else:
        requested_keys = keys
    # Need id key (it's used as master key)
    if "id" not in requested_keys:
        requested_keys.append("id")

    discover = tmdb.Discover()
    # Get the number of pages
    for db_request in range(0, retry_limit):
        while True:
            try:
                movie_page = discover.movie(language="en", primary_release_year=year)
            except:
                log.out.debug("HTTP error, waiting and retrying.")
                time.sleep(db_request/retry_limit)
                continue
            break

    log.out.info("Adding year: " + str(year) + " (" +
                 str(movie_page["total_pages"]) + " pages)")
    for page_num in tqdm(range(1, movie_page["total_pages"] + 1)):
        for db_request in range(0, retry_limit):
            while True:
                try:
                    movie_page = discover.movie(page=page_num,
                                                language=language,
                                                primary_release_year=year)
                except:
                    log.out.debug("HTTP error, waiting and retrying.")
                    time.sleep(db_request / retry_limit)
                    continue
                break

        result_list = movie_page["results"]
        for movie_dict in result_list:
            requested_dict = {}
            for key in requested_keys:
                requested_dict[key] = movie_dict[key]
            year_of_movies.append(requested_dict)
    return year_of_movies


def get_movie_image(image_string):
    image_base_url = "https://image.tmdb.org/t/p/w500/"
    image_url = image_base_url + image_string
    print("foo: " + image_url)
    # TODO Finish this up


def write_json_from_dict(input_dict, filepath=".", filename="out.json", overwrite=True):
    """
    :param input_dict: dictionary to write a file from
    :param filepath: (str) the path to output to
    :param filename: (str) name of the file
    :param overwrite: (bool) overwrite existing files
    :return: bool of success
    """
    full_filepath = os.path.expanduser(filepath)     # Fully resolve ~'s in path
    full_filepath = os.path.realpath(full_filepath)  # Fully resolve symbolic links in path
    full_filepath = os.path.abspath(full_filepath)   # Fully resolve relative paths (../../etc)
    full_filename = os.path.join(full_filepath, filename)  # Merge the filename and path for OS
    # Check path existence and access permission
    if os.access(full_filepath, os.W_OK):
        if os.path.isfile(full_filename):
            if overwrite:
                log.out.warning("File exists! Overwriting.")
            else:
                log.out.warning("File exists! Not overwriting.")
                return False
        with open(full_filename, 'w') as file_handle:
            json.dump(input_dict, file_handle)
        return True
    else:
        log.out.error("Can not write to file with name: " + full_filename)
        return False


def movie_data_to_json(output_json, year_start=1950, year_end=2017):
    keys_wanted = ["genre_ids", "title", "release_date", "original_language",
                   "vote_count", "vote_average", "popularity",
                   "poster_path", "backdrop_path", "overview"]
    master_dict = {}

    for this_year in range(year_start, year_end+1):
        log.out.info("Requesting year: " + str(this_year))
        this_years_data = request_v3_by_year(this_year, keys=keys_wanted)
        for movie_dict in this_years_data:
            master_dict[movie_dict["id"]] = {}
            master_dict[movie_dict["id"]]["year"] = this_year
            for key in keys_wanted:
                master_dict[movie_dict["id"]][key] = movie_dict[key]
        # Overwrite temp file with new data
        pathname, filename = os.path.split(output_json)
        write_json_from_dict(master_dict, filepath=pathname,
                             filename=filename+'.tmp', overwrite=True)
    # Write final data
    pathname, filename = os.path.split(output_json)
    write_json_from_dict(master_dict, filepath=pathname,
                         filename=filename, overwrite=True)

if __name__ == '__main__':
    log.setLevel("INFO")
    read_api_keys(API_FILE)
    tmdb.API_KEY = API_V3_KEY
    movie_data_to_json("./movie_info.json", year_start=1950, year_end=2017)
