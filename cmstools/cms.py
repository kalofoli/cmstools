'''
Created on Jan 8, 2023

@author: janis
'''

import requests

import os,re
import logging
import urllib
from .constants import _xpaths, _strings, _urls
from .common import CMSError, ParseError, LoginError

from lxml import etree
from types import SimpleNamespace



logging.basicConfig()
log = logging.getLogger('cmstools')



class CMSSession:
    _xpaths = _xpaths
    _urls = _urls
    _strings = _strings
    def __init__(self, sid, cache=False, verify_login=True, store=False):
        self.sid = sid
        self.cache = cache
        self.verify_login = verify_login
        self.store = store
        self._sess = requests.session()

    @classmethod
    def parse_dom_from_html(cls, html):
        from io import StringIO

        p = etree.HTMLParser()
        tree = etree.parse(StringIO(html), p)
        return tree.getroot()
    
    def _download_url(self, url, data=None, files=None, extra=False, dry=False):
        cookies = {'CakeCMS': self.sid}
        full_url = f"{self._urls['base']}{url}" if not url.startswith('http') else url
        headers = {'Origin':self._urls['origin']}
        method = 'GET' if data is None and files is None else 'POST' 
        req = requests.Request(method=method, url=full_url, files=files, data=data, cookies=cookies, headers=headers).prepare()
        if not dry:
            resp = self._sess.send(req)
            text = resp.text
            from http import HTTPStatus
            if resp.status_code != HTTPStatus.OK:
                raise CMSError(f'While downloading url {full_url} with data "{data}" a non-ok status of {resp.status_code} was received.')
        else:
            resp,text = None,None
        return SimpleNamespace(text=text,request=req,response=resp) if extra else resp.text
    
    def request_url(self, url, data=None):
        '''Request a url without caching.
        If data is provided, a POST method is used, otherwise a GET one.
        '''
        html = self._download_url(url=url, data=data)
        return html, self.parse_dom_from_html(html)
        
    def _retrieve_url(self, url, cache=None, cache_key=None, verify_login=True, data=None, files=None):
        if cache is None:
            cache = self.cache if (data is None or cache_key is not None) else False
        text = None
        if cache is not False:
            url_safe = urllib.parse.quote(url if cache_key is None else cache_key, safe='').lower()
            cache_path = os.path.join(cache, url_safe)
            try:
                with open(cache_path,'r') as fid:
                    text = fid.read()
                    log.info(f'Retrieved contents of url "{url}" from file "{cache_path}".')
                root = self.parse_dom_from_html(text)
            except OSError as e:
                if not isinstance(e, FileNotFoundError):
                    log.info(f'While trying to load file {url} from cache: {e}')
        if text is None:
            text = self._download_url(url, data=data, files=files)
        
            root = self.parse_dom_from_html(text)
            
            if verify_login is None: verify_login = self.verify_login
            if verify_login:
                e = root.xpath(self._xpaths['login_action'])
                if not (len(e)==1 and e[0] == self._strings['login_response']):
                    raise LoginError(f'While downloading url "{url}" could not find logout button. This probably means your session is invalid.')
                msg = root.xpath('//div[@id="authMessage"]/text()')
                if len(msg):
                    raise LoginError(msg[0])
            if cache is not False and self.store:
                with open(cache_path,'w') as fid:
                    fid.write(text)
                    log.info(f'Stored contents of url "{url}" as file "{cache_path}".')
        return text, root

    def __call__(self, url='landing', verify_login=None, cache=None, cache_key=None, data=None, files=None):
        return self._retrieve_url(url, cache=cache, cache_key=cache_key, verify_login=verify_login, data=data, files=files)

    def fetch_team(self, index, cache=None, verify_login=None):
        url = self._urls['teams'].format(index=index)
        return self(url, cache=cache, verify_login=verify_login, cache_key=f'keyed_teams_{index}')
    
    def fetch_sub_item(self, index, cache=None, verify_login=None):
        url = self._urls['submission_items'].format(index=index)
        return self(url, cache=cache, verify_login=verify_login, cache_key=f'keyed_sub_item_{index}')
    
    def fetch_submissions(self, cache=None, verify_login=None):
        url = self._urls['submissions']
        return self(url, cache=cache, verify_login=verify_login, cache_key='keyed_submissions')

    def fetch_students(self, cache=None, verify_login=None):
        url = self._urls['students']
        return self(url, cache=cache, verify_login=verify_login, cache_key='keyed_students')

    def fetch_testings(self, tutorial, indices, cache=None, verify_login=None):
        url = self._urls['testings']
        tutorial = str(tutorial)
        assert tutorial in ["","1","2"],f'Specified {tutorial} which is not in the list of allowed tutorials: "", "1", "2".'
        dct_testings = {f"data[Testings][{tid}]":str(tid) for tid in indices}
        data = {"data[Search][tutorial]": str(tutorial), **dct_testings}
        sids = ','.join(map(str, sorted(indices)))
        cache_key = f"keyed_testing_{sids}?tutorial={tutorial}"
        return self(url, cache=cache, verify_login=verify_login, cache_key=cache_key, data=data)
    
    def fetch_testings_csv(self, ids):
        scols = '~'.join(f'T{t}' for t in ids)
        url = self._urls['student_cols'].format(cols=scols)
        return self(url, verify_login=False, cache_key=f'keyed-testing-{scols}')[0]

    rex_import = re.compile('^File imported, (?P<num>[0-9]+) entries have been saved.$')
    def submit_grades(self, index, mns, points, dry=False, cache=None, verify_login=None, verify_response=True):
        assert len(mns)==len(points),f'The MN and Point sequences must be of same length, but they are of {len(mns)} and {len(points)} sizes, respectively.'
        shdr = 'mtknr;points'
        sbdy = '\r\n'.join(f'{m};{p}' for m,p in zip(mns,points))
        csv = shdr+'\r\n'+sbdy + '\r\n'
        csv
        
        url = self._urls['import_csv'].format(index=index)
        r_data = self(url, cache=cache, verify_login=verify_login, cache_key=f'keyed_import_{index}')[1]
        e_inp = r_data.xpath('//form[@id="ImportImportForm"]//input[@type="hidden"]')
        data = {**dict((e_i.attrib['name'],e_i.attrib['value']) for e_i in e_inp),'data[Import][data]':csv}
        
        if not dry:
            html,r = self._retrieve_url(url, data=data, cache=False, verify_login=verify_login)
            if verify_response:
                e = r.xpath('//body/div[@class="container"]/div[@class="row"]//div[contains(@class,"alert")]')
                if len(e) == 0:
                    raise ParseError(f'While importing grades for testing {index}: no response message found.')
                e_r = e[0]
                stxt = e_r.text.strip()
                e_r_cls = e_r.attrib['class']
                if 'alert-error' in e_r_cls:
                    raise CMSError(f'While importing grades for testing {index}: {stxt}')
                if 'alert-success' in e_r_cls:
                    m = self.rex_import.match(stxt)
                    if m is None:
                        raise ParseError(f'The response returned was "{stxt}" which was not in the expected format.')
                    else:
                        log.info(f'Server responded that {m.group("num")} grades were imported.')
                    rows = int(m.group('num'))
                else:
                    raise ParseError(f'While importing grades for testing {index}: Could not find an error or success tag in the response element.')
            else:
                rows = None
        else:
            html,r = '',None
            rows = None
        return rows, (html,r), csv        

    def update_report(self, content=None, title=None, menu_visible=None, verify_login=None, verify_response=True):
        url = self._urls['upload_report']
        
        r_data = self(url, cache=False, verify_login=verify_login, cache_key=f'keyed_upload_report')[1]
        e_inp = r_data.xpath('//form[@id="ContentEditForm"]//input[@name and @value]')
        e_ta = r_data.xpath('//form[@id="ContentEditForm"]//textarea[@id="ContentContent"]')[0]
        dct_orig = dict((e_i.attrib['name'],e_i.attrib['value']) for e_i in e_inp)
        dct_orig[e_ta.attrib['name']] = e_ta.text
        dct_args = dict(content=content, title=title, menu_visible=menu_visible)
        dct_updates = {f'data[Content][{k}]':str(v) for k,v in dct_args.items() if v is not None}
        data = {**dct_orig,**dct_updates}
    
        html,r = self._retrieve_url(url, data=data, cache=False, verify_login=verify_login)
        if verify_response:
            e = r.xpath('//body/div[@class="container"]/div[@class="row"]//div[contains(@class,"alert")]')
            prefix = f'While updating {", ".join(dct_args.keys())} of report at "{url}": '
            if len(e) == 0:
                raise ParseError(prefix+'no response message found.')
            e_r = e[0]
            stxt = e_r.text.strip()
            e_r_cls = e_r.attrib['class']
            if 'alert-error' in e_r_cls:
                raise CMSError(prefix+stxt)
            if 'alert-success' in e_r_cls:
                if stxt != 'The content has been saved':
                    raise ParseError(prefix+f'The response returned was "{stxt}" which was not in the expected format.')
            else:
                raise ParseError(prefix + f'Could not find an error or success tag in the response element.')
        return (html,r), dct_updates

    @classmethod
    def _parse_form(cls, e_f, hidden=True, types=[]):
        types = (['hidden'] if hidden else []) + list(types)
        stype = " or ".join(f'@type="{t}"' for t in types)
        e_inp = e_f.xpath(f'.//input[{stype}]')
        data = {**dict((e_i.attrib['name'],e_i.attrib.get('value')) for e_i in e_inp)}
        return data
        
    def logout(self):
        url = self._urls['logout']
        r = self()[1]
        e_f = r.xpath('/html/body//form[@id="logoutForm"]')[0]
        data = self._parse_form(e_f)
        log.info(f'Requested logout.')
        import http
        max_reqs = http.client._MAXHEADERS
        http.client._MAXHEADERS = 5000 
        try:
            self(url, data=data)
        except LoginError:
            return
        finally:
            http.client._MAXHEADERS = max_reqs
        raise ParseError('Failed to logout')
        
    def _verify_response(self, root, rex=None):
        ''' Verify a response in a given element.
        @return If rex is provided, the named groups starting with resp_ are returned (without theyr prefix). Otherwise the raw text in the success alert.
        Upon error it raises a CMSError.
        '''
        e = root.xpath('.//div[contains(@class,"alert")]')
        if len(e) == 0:
            raise ParseError(f'No response message found.')
        e_r = e[0]
        stxt = e_r.text.strip()
        e_r_cls = e_r.attrib['class']
        if 'alert-error' in e_r_cls:
            raise CMSError(stxt)
        if 'alert-success' in e_r_cls:
            if rex is not None:
                m = rex.match(stxt)
                if m is None:
                    raise ParseError(f'The response returned was "{stxt}" which was not in the expected format.')
                else:
                    return {k[5:]:v for k,v in m.groupdict().items() if k.startswith('resp_')}
            else:
                return stxt
        else:
            raise ParseError(f'Could not find an error or success tag in the response element.')
        
    _rex_submit = re.compile('Your solution has been submitted: (?P<resp_bytes>[0-9]+) bytes, SHA256=(?P<resp_sha256>[a-z0-9]+)$')
    def upload_submission(self, sub_id, mn, file,file_name="upload.pdf", content_type="application/pdf", verify_result=True, cache=None):
        url = _urls['student_view'].format(mn=mn)
        root = self(url,cache_key=f'keyed_student_{mn}', cache=cache)[1]
        
        els = root.xpath(f'//body/div[@class="container"]/div[@class="row"]//div[@id="content"]/div[@id="submissions"]/table//div[@id="SubmissionUpload{sub_id}"]')
        if len(els) == 0:
            raise ParseError(f'Could not find upload information for submission {sub_id}.')
        el = els[0]
        data = self._parse_form(el)
        files = {'data[SubmissionItem][upload]':(file_name, file, content_type)}
        root = self(_urls['upload_submission'],data=data, files=files,cache=False)[1]
        if verify_result:
            try:
                res_dct = self._verify_response(root.xpath('//div[@id="submissions"]')[0], rex=self._rex_submit)
                res = SimpleNamespace(**res_dct)
                log.info(f'Submitted id:{sub_id} for MN: {mn} file of {res.bytes} bytes and SHA356: {res.sha256}.')
            except ParseError as e:
                raise ParseError(f'Failed to submit id: {sub_id} for MN: {mn}.') from e
        else:
            res = None
        return res
