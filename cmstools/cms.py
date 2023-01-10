'''
Created on Jan 8, 2023

@author: janis
'''

import requests

import os
import logging
import urllib
from .constants import _xpaths, _strings, _urls

from lxml import etree


logging.basicConfig()
log = logging.getLogger('cmstools')

class CMSSession:
    _xpaths = _xpaths
    _urls = _urls
    _strings = _strings
    def __init__(self, sid, cache=None, verify_login=True):
        self.sid = sid
        self.cache = cache
        self.verify_login = verify_login

    @classmethod
    def parse_dom_from_html(cls, html):
        from io import StringIO

        p = etree.HTMLParser()
        tree = etree.parse(StringIO(html), p)
        return tree.getroot()
    
    def _download_url(self, url, data=None):
        cookies = {'CakeCMS': self.sid}
        full_url = f"{self._urls['base']}{url}"
        if data is None:
            resp = requests.get(full_url,cookies=cookies)
        else:
            resp = requests.post(full_url,cookies=cookies, data=data)
        from http import HTTPStatus
        if resp.status_code != HTTPStatus.OK:
            raise requests.HTTPError(f'While downloading url {resp.url} a non-ok status of {resp.status_code} was received.')
        return resp.text
    
    def request_url(self, url, data=None):
        '''Request a url without caching.
        If data is provided, a POST method is used, otherwise a GET one.
        '''
        html = self._download_url(url=url, data=data)
        return html, self.parse_dom_from_html(html)
        
    def _retrieve_url(self, url, cache=None):
        if cache is None: cache = self.cache
        text = None
        if cache is not None:
            url_safe = urllib.parse.quote(url, safe='').lower()
            cache_path = os.path.join(cache, url_safe)
            try:
                with open(cache_path,'r') as fid:
                    text = fid.read()
                    log.info(f'Retrieved contents of url "{url}" from file "{cache_path}".')
            except OSError as e:
                if not isinstance(e, FileNotFoundError):
                    log.info(f'While trying to load file {url} from cache: {e}')
        if text is None:
            text = self._download_url(url)
            if cache is not None:
                with open(cache_path,'w') as fid:
                    fid.write(text)
                    log.info(f'Stored contents of url "{url}" as file "{cache_path}".')
        return text

    def __call__(self, url='landing', verify_login=None, cache=None):
        if cache is None: cache = self.cache
        text = self._retrieve_url(url, cache=cache)
        root = self.parse_dom_from_html(text)
        
        if verify_login is None: verify_login = self.verify_login
        if verify_login:
            e = root.xpath(self._xpaths['login_action'])
            assert len(e)==1 and e[0] == self._strings['login_response'], f'While downloading url "{url}" could not find logout button. This probably means your session is invalid.'
            msg = root.xpath('//div[@id="authMessage"]/text()')
            if len(msg):
                raise requests.HTTPError(msg[0])
        return text, root

    def fetch_team(self, index, cache=None, verify_login=None):
        url = self._urls['teams'].format(index=index)
        return self(url, cache=cache, verify_login=verify_login)
    
    def fetch_sub_item(self, index, cache=None, verify_login=None):
        url = self._urls['submission_items'].format(index=index)
        return self(url, cache=cache, verify_login=verify_login)
    
    def fetch_submissions(self, cache=None, verify_login=None):
        url = self._urls['submissions']
        return self(url, cache=cache, verify_login=verify_login)





