"""
This script takes a json and extracts a text file from it

Usage:
extract_text.py --output ./foo.txt

External Dependencies:
mailbox (pip install mailbox) (for mbox type)

@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""


from uplog import log
import random
import mythic_common as common
import re

defined_extraction_types = ["json", "mbox"]


def extract_json_to_text(input_filename, output_filename,
                         key=None, samples=None, clean=True):
    """
    Assumes JSON structure like:
    {
        "1234": {
            "genre_ids": [10751, 14, 16, 10749],
            "title": "Cinderella",
        },
        "5678": {
            "genre_ids": [18],
            "title": "Sunset Boulevard",
        },...
    }
    :param input_filename: the input json filename
    :param output_filename: the output text filename
    :param key: inner key you want to extract
    :param samples: number of samples to grab (all if left None)
    :param clean: boolean to specify data cleaning
    :return:
    """
    import json
    log.out.info("Opening: " + input_filename)
    json_file_handle = open(input_filename)
    json_data = json.load(json_file_handle)
    log.out.info("Found: " + str(len(json_data)) + " records.")
    data_ids = list(json_data.keys())
    log.out.info("Parsing records.")
    data_array = []
    test_id = data_ids[0]
    available_keys = list(json_data[test_id].keys())
    log.out.info("Found these keys in data: " + ', '.join(available_keys))
    if key is None:
        key = available_keys[0]
        log.out.warning("No key specified, using the first one i found: " + key)

    if samples is not None:
        # Get a random subset
        log.out.info("Outputting " + str(samples) + " samples from raw data.")
        for i in range(samples):
            rando = data_ids[random.randint(0, len(json_data)-1)]
            data_array.append(json_data[rando][key])
    else:
        # Grab em all!
        log.out.info("Outputting all text from raw data.")
        for data_id in data_ids:
            if len(json_data[data_id][key]) > 0:
                data_array.append(json_data[data_id][key])
    # Write out the text file
    log.out.info("Writing to: " + output_filename)
    outfile = open(output_filename, 'w')
    for line in data_array:
        outfile.write("%s\n" % line)

# def prepare_nonsense(movie_overview_array, pattern_size=None):
#     movie_info_raw = munge_huge_clump(movie_overview_array)
#     # Create mapping of printable chars to integers
#     sorted_chars = sorted(list(set(printable.lower())))
#     log.out.info("Cleaning data of weird characters.")
#     movie_info_clean = clean_string_to_printable(movie_info_raw)
#     median_overview_length = get_median_length(movie_overview_array)
#     log.out.info("Median overview length: " + str(median_overview_length))
#     char_to_int = dict((c, i) for i, c in enumerate(sorted_chars))
#     num_chars = len(movie_info_clean)
#     num_vocab = len(sorted_chars)
#     log.out.info("Total characters: " + str(num_chars) +
#                  " Total vocabulary: " + str(num_vocab))


def extract_mbox_to_text(input_filename, output_filename, clean=True):
    import mailbox
    log.out.info("Opening: " + input_filename)
    mbox = mailbox.mbox(input_filename)
    body_array = []
    for message in mbox:
        body = None
        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    for subpart in part.walk():
                        if subpart.get_content_type() == 'text/plain':
                            body = subpart.get_payload(decode=True)
                elif part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True)
        elif message.get_content_type() == 'text/plain':
            body = message.get_payload(decode=True)

        body_string = body.decode('ISO-8859-1')

        # If you see this the rest is some dumb tailer that you don't want
        tailer_marker = "This is the body.\r\n>From (should be escaped).\r\nThere are 3 lines."
        if tailer_marker in body_string:
            break
        if clean:
            """
            - Remove all non-printables
            - Remove *s
            - Search bodies for the following pattern:
                 This is the body.
                 >From (should be escaped).
                 There are 3 lines.
              Remove this and everything after it
            - Remove 
            """
            # Regex to remove multiple carriage returns and multiple spaces
            body_string = re.sub(r'\n\s*\n', '\n\n', body_string)
            body_string = re.sub(r'[\r\n][\r\n]{2,}', '\n\n', body_string)
            body_string = re.sub(' +', ' ', body_string)
            # These are weird tab and apostrophe things
            body_string = body_string.replace("\xe2\x80\x93", "")
            body_string = body_string.replace("\xe2\x80\x99", "'")
            body_string = body_string.replace("\xe2\x80\x9c", "")
            body_string = body_string.replace("\xe2\x80\x9d", "")
            body_string = body_string.replace("\xc2\xb7", "")
            body_string = body_string.replace("\xcb\x88", "")
            body_string = body_string.replace("*", "")
            while "<https:" in body_string:
                # print(body_string)
                http_start_index = body_string.find("<https:")
                http_stop_index = http_start_index
                while body_string[http_stop_index] != ">":
                    http_stop_index += 1
                body_string = body_string[:http_start_index] + body_string[http_stop_index+1:]

        # Go through the string and remove all non-printables
        for character in body_string:
            if character not in common.trainable_characters:
                body_string = body_string.replace(character, "")
        body_array.append(body_string)

    # Write out the text file
    log.out.info("Writing to: " + output_filename)
    outfile = open(output_filename, 'w')
    for line in body_array:
        outfile.write("%s\n" % line)


if __name__ == '__main__':
    # Get settings from command line
    extract_settings = common.ExtractorSettings()
    if extract_settings.debug:
        log.out.setLevel('DEBUG')
    else:
        log.out.setLevel('INFO')

    # Parse command line arguments
    log.out.info("Extraction type: " + extract_settings.type)
    log.out.info("Loading from file: " + extract_settings.data_file)

    if extract_settings.type not in defined_extraction_types:
        log.out.error("Extraction type requested: " + extract_settings.type + " not defined.")
    elif extract_settings.type == 'json':
        extract_json_to_text(extract_settings.data_file,
                             extract_settings.out_file,
                             key=extract_settings.key,
                             samples=extract_settings.samples,
                             clean=extract_settings.clean)
    elif extract_settings.type == 'mbox':
        extract_mbox_to_text(extract_settings.data_file,
                             extract_settings.out_file,
                             clean=extract_settings.clean)
