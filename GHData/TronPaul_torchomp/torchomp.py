import argparse
import time
import datetime
import urllib
import urlparse
import sqlite3
import feedparser
import deluge_client


def convert_time(time_struct):
    return datetime.datetime.fromtimestamp(time.mktime(time_struct))


def max_date(entries):
    return max([convert_time(e['published_parsed']) for e in entries])


def get_password(auth_file):
    with open(auth_file) as fp:
        for line in fp:
            if line.startswith('localclient'):
                return line.split(':')[1]


def ensure_database(db_conn):
    c = db_conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS feeds (id INTEGER PRIMARY KEY, query text UNIQUE NOT NULL, max_entry_date timestamp)')
    db_conn.commit()


def add_torrent_to_deluge(deluge_password, torrent_url):
    client = deluge_client.DelugeRPCClient('localhost', 58846, 'localclient', deluge_password)
    client.connect()
    client.call('core.add_torrent_url', torrent_url, {})


def search_query(query):
    qs = urllib.urlencode({'term': query,'page': 'rss'})
    url = urlparse.urlunparse(('http', 'www.nyaa.se', '', '', qs, ''))
    d = feedparser.parse(url)
    return d['entries']


def add_feed(db_conn, deluge_password, query, add_all=True):
    c = db_conn.cursor()
    entries = search_query(query)
    if add_all:
        for e in entries:
            add_torrent_to_deluge(deluge_password, e['link'])
    max_entry_date = max_date(entries)
    c.execute('INSERT INTO feeds (query, max_entry_date) VALUES (?, ?)', (query, max_entry_date))
    db_conn.commit()


def add_new_torrents_from_feeds(db_conn, deluge_password):
    c = db_conn.cursor()
    c.execute('SELECT id, query, max_entry_date as "ts [timestamp]" from feeds')
    for (id_, query, max_entry_date) in c.fetchall():
        entries = [e for e in search_query(query) if not max_entry_date or convert_time(e['published_parsed']) > max_entry_date]
        if entries:
            for e in entries:
                add_torrent_to_deluge(deluge_password, e['link'])
            c.execute('UPDATE feeds SET max_entry_date=? where id=?', (max_date(entries), id_))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', default='/var/lib/deluge/torchomp.sqlite')
    parser.add_argument('--auth_file', default='/etc/deluge/auth')
    subparsers = parser.add_subparsers(dest='subparser_name')
    add_feed_p = subparsers.add_parser('add_feed')
    add_feed_p.add_argument('query')
    add_feed_p.add_argument('--add_all', type=bool, default=True)

    add_new_torrents_p = subparsers.add_parser('add_new_torrents')

    args = parser.parse_args()

    conn = sqlite3.connect(args.database_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    ensure_database(conn)
    password = get_password(args.auth_file)

    if args.subparser_name == 'add_feed':
        add_feed(conn, password, args.query, args.add_all)
    elif args.subparser_name == 'add_new_torrents':
        add_new_torrents_from_feeds(conn, password)
    conn.close()

if __name__ == '__main__':
    main()