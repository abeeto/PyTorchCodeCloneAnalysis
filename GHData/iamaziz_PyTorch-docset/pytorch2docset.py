#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Aziz Alto

"""
Dependency:
- requests
- BeautifulSoup4
- httrack (to download html documentation. Well, website mirroring)
"""

import sqlite3
import os
import plistlib
import string
import requests
from bs4 import BeautifulSoup


class Docset(object):
    def __init__(self, docset_name, index_page, pages, icon_url, html_url, download_html=False):
        self.name = docset_name
        self.index_page = index_page
        self.pages = pages
        self.docs_output = None
        self.docset_name = '{}.docset'.format(self.name)
        self.setup_docset(html_url, download_html)
        self.add_infoplist()
        self.cur, self.db = self.connect_db()
        self.scrape_urls()
        self.get_icon(icon_url)
        self.report()

    def setup_docset(self, url, download_html):

        self.docs_output = self.docset_name + '/Contents/Resources/Documents/'
        if not os.path.exists(self.docs_output):
            os.makedirs(self.docs_output)
        cmd = """
            cd {0} &&
            httrack -%v2 -T60 -R99 --sockets=7 -%c1000 -c10 -A999999999 -%N0 --disable-security-limits --keep-links=K4 -F 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.19 (KHTML, like Gecko) Ubuntu/11.10 Chromium/18.0.1025.168' --mirror --keep-alive --robots=0 "{1}" -n -* +*.css +*css.php +*.ico +*/fonts/* +*.svg +*.ttf +fonts.googleapis.com* +*.woff +*.eot +*.png +*.jpg +*.gif +*.jpeg +*.js +{1}* -github.com* +raw.github.com* &&
            rm -rf hts-* &&
            mkdir -p Contents/Resources/Documents &&
            mv -f *.* Contents/Resources/Documents/
            """.format(self.docset_name, url)
        if download_html:
            os.system(cmd)

    def connect_db(self):
        db = sqlite3.connect(self.docset_name + '/Contents/Resources/docSet.dsidx')
        cursor = db.cursor()
        try:
            cursor.execute('DROP TABLE searchIndex;')
        except sqlite3.OperationalError:
            cursor.execute('CREATE TABLE searchIndex(id INTEGER PRIMARY KEY, name TEXT, type TEXT, path TEXT);')
            cursor.execute('CREATE UNIQUE INDEX anchor ON searchIndex (name, type, path);')
        return cursor, db

    def scrape_urls(self):

        idx = (i + j for i in string.ascii_lowercase for j in string.ascii_lowercase)
        pages = self.pages
        for entry_type in pages:
            # base path of current page
            base_path = pages[entry_type].split("//")[1]
            # soup each page
            html = requests.get(pages[entry_type]).text
            soup = BeautifulSoup(html, 'html.parser')
            # find href and populate entries to db
            for a in soup.findAll('a', class_='reference internal'):
                entry_name = a.text.strip()
                path = a.get('href')
                if entry_type == 'Guide':
                    entry_name = '{}: {}'.format(idx.next(), entry_name.encode('ascii', 'ignore'))
                path = base_path + path
                entry_name = entry_name.encode('ascii', 'ignore')
                self.update_db(entry_name, entry_type, path)

    def update_db(self, entry_name, typ, path):

        self.cur.execute("SELECT rowid FROM searchIndex WHERE path = ?", (path,))
        dbpath = self.cur.fetchone()
        self.cur.execute("SELECT rowid FROM searchIndex WHERE name = ?", (entry_name,))
        dbname = self.cur.fetchone()
        if dbpath is None and dbname is None:
            self.cur.execute('INSERT OR IGNORE INTO searchIndex(name, type, path) VALUES (?,?,?)',
                             (entry_name, typ, path))
            print('DB add >> name: {0} | type: {1} | path: {2}'.format(entry_name, typ, path))
        else:
            print("record exists")

    def add_infoplist(self):

        index_file = self.index_page.split("//")[1]
        plist_path = os.path.join(self.docset_name, "Contents", "Info.plist")
        plist_cfg = {
            'CFBundleIdentifier': self.name,
            'CFBundleName': self.name,
            'DocSetPlatformFamily': self.name.lower(),
            'DashDocSetFamily': 'python',
            'isDashDocset': True,
            'isJavaScriptEnabled': True,
            'dashIndexFilePath': index_file
        }
        plistlib.writePlist(plist_cfg, plist_path)

    def report(self):

        self.cur.execute('SELECT count(*) FROM searchIndex;')
        entry = self.cur.fetchone()
        # commit and close db
        self.db.commit()
        self.db.close()
        self.compress_docset()
        print("{} entry.".format(entry))

    def get_icon(self, png_url):
        """grab icon and resize to 32x32 and 16x16 pixel"""
        cmd = """
        wget -O icon.png {} &&
        cp icon.png {}/icon.png &&
        cp icon.png icon@2x.png &&
        sips -z 32 32 icon@2x.png &&
        sips -z 16 16 icon.png
        """.format(png_url, self.docset_name)
        os.system(cmd)

    def compress_docset(self):
        """compress the docset as .tgz file"""
        cmd = """
        tar --exclude='.DS_Store' -cvzf {0}.tgz {0}.docset
        """.format(self.docset_name.replace('.docset', ''))
        os.system(cmd)


if __name__ == '__main__':
    name = 'PyTorch'
    download_url = 'http://pytorch.org/'
    index_page = 'http://pytorch.org/docs/index.html'
    entry_pages = {
        'func': 'http://pytorch.org/docs/master/',
        'Guide': 'http://pytorch.org/tutorials/'
    }
    icon = 'https://avatars2.githubusercontent.com/u/21003710'

    Docset(name, index_page, entry_pages, icon, download_url, download_html=True)
