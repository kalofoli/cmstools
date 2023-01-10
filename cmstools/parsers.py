'''
Created on Jan 8, 2023

@author: janis
'''

import typing
import itertools
from types import SimpleNamespace
import re

import pandas as pd

class ElementParser():
    def __call__(self, e): return e
class TextParser(ElementParser):
    def __init__(self, strip=True, lower=False):
        self.strip = strip
        self.lower = lower
    def _text(self, txt):
        if txt is not None:
            if self.lower: txt = txt.lower()
            if self.strip: txt = txt.strip()
        return txt
    def __call__(self, e):
        return self._text(e.text)
class XpathParser(TextParser):
    def __init__(self, xpath, indices=slice(None,None), sep='', **kwargs):
        super().__init__(**kwargs)
        self.sep = sep
        self.xpath = xpath
        self.indices = indices
    def _xpath(self, e):
        els = e.xpath(self.xpath)
        indices = self.indices.indices(len(els)) if isinstance(self.indices, slice) else self.indices
        els_sel = [e for i,e in enumerate(els) if i in indices]
        return els_sel
    def __call__(self, e):
        els_sel = self._xpath(e)
        txt = map(self._text, els_sel)
        return self.sep.join(els_sel) if self.sep is not None else list(txt)

class ChildrenTextParser(XpathParser):
    def __init__(self, **kwargs):
        super().__init__(xpath='*/text()', **kwargs)
class DateParser(XpathParser):
    def __init__(self, **kwargs):
        super().__init__(xpath='span/text()', **kwargs)

default_parsers = {
    None:'_text',
    '_text': TextParser(),
    '_date': XpathParser(r'span/@title'),
    '_a_href': XpathParser('a/@href'),
    '_a_text': XpathParser('a/text()'),
    '_a' : XpathParser('a/@href | a/text()',sep=None)
}

class HTMLParser():
    def __init__(self, root):
        self.data = self._parse_data_from_root(root)
    @classmethod
    def _expand_tuple(cls,df,column,names,append_names=False,drop=False):
        if append_names:
            names = [f'{column}{name}' for name in names]
        data = dict(zip(names,itertools.zip_longest(*df[column])))
        df = pd.concat((df, pd.DataFrame(data)), axis=1)
        if drop:
            df = df.drop(column, axis=1)
        return df
    @classmethod
    def _expand_a(cls,df,column, text_var='Name', url_var='Url', append_names=True):
        return cls._expand_tuple(df, column, [url_var,text_var], append_names=append_names, drop=True)
    @classmethod
    def _strip_urls(cls, df, cols=None, inplace=False):
        if not inplace:
            df = df.copy()
        if cols is None:
            cols = df.columns[df.columns.str.lower().str.contains('url')]
        df[cols] = df[cols].apply(lambda x:x.str.replace('^/eml22','',regex=True))
        return df

    def _parse_table(self, tbl, only_sortable=False, parsers={}):
        '''
        @param parsers: a dictionary specifying the parser for each element.
            Each key is either the name of a columnt or a column index.
            Each value is either a callable that will be used as a parser with
            a single argument the etree Element to parse, or a key that will be
            further looked up in the parsers dictionary.
            A key of None is used as a fallback.
            This value is expanded by the values of the default_parsers dict.
        '''
        parsers = {**default_parsers, **parsers}
        def parse_head(e):
            c = e.getchildren()
            return (c[0].text,True) if c else (e.text,False)
        
        e_trs = tbl.xpath('tr')
        names,sortable = zip(*map(parse_head, e_trs[0].xpath('th')))
        def parse_data(i,n,e):
            parser = parsers.get(i)
            if parser is None:
                parser = parsers.get(n)
            if parser is None:
                parser = parsers.get(None)
            if not isinstance(parser, typing.Callable):
                parser = parsers.get(parser)
            if parser is None:
                raise ValueError(f'Could not find a valid parser for entry with name {n} at index {i}.')
            return parser(e)
        is_ok = sortable.__getitem__ if only_sortable else lambda _: True
        rows = []
        for e_tr in e_trs[1:]:
            e_tds = e_tr.xpath('td')
            row = [parse_data(i,n,e) for i,(n,e) in enumerate(zip(names, e_tds)) if is_ok(i)]
            rows.append(row)
        names_s = [n for i,n in enumerate(names) if is_ok(i)]
        return pd.DataFrame(rows, columns=names_s)
    
    def __len__(self): return self.data.shape[0]
    def __repr__(self): return f'<{type(self).__name__} with {len(self)} entries>'

class HTMLParserWithMeta(HTMLParser):
    def __init__(self, root):
        super().__init__(root)
        self.meta = self._parse_meta_from_root(root)
    def __repr__(self): return f'<{type(self).__name__}[{self.meta.Name}] with {len(self)} entries>'
    
class SubmissionListParser(HTMLParser):
    def _parse_data_from_root(self, root):
        tbl = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="submissions index"]/h2[text()="Submissions"]/following-sibling::table')[0]
        href_parser = default_parsers['_a_href']
        text_parser = default_parsers['_text']
        df = self._parse_table(tbl, only_sortable=True, parsers={
            'Start': '_date', 'End': '_date', 'Points': '_a', 'Teams': '_a',
            '# Solutions': lambda x:(href_parser(x),text_parser(x))
        })
        df = self._expand_a(df,'# Solutions',text_var='Number',url_var='SolutionsUrl', append_names=False)
        df = self._expand_a(df,'Points')
        df = self._expand_a(df,'Teams')
        df = df.drop(['Admission?'],axis=1)
        df = self._strip_urls(df)
        return df

class SubmissionItemParser(HTMLParser):
    def _parse_data_from_root(self, root):
        tbl = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]/table[@class="table table-striped"]')[0]
        df = self._parse_table(tbl, only_sortable=False, parsers={
            'Student': '_a', 'Date':'_date', 'Mat.Nr':'_a_text',
            'Team':'_a',
            9:XpathParser('a/@href',sep=None)
        })
        df.columns.values[9] = '_actions_'
        df = self._expand_tuple(df,'_actions_',['FileUrl','FeedbackUrl'], append_names=False, drop=True)
        df = self._expand_a(df,'Student')
        df = self._expand_a(df,'Team')
        df = self._strip_urls(df)#, ['FileUrl','FeedbackUrl','StudentUrl','TeamUrl'])
        return df



class TeamParser(HTMLParserWithMeta):
    rex_mn = re.compile('.*/(?P<mn>[0-9]+)')
    def __init__(self, root):
        self.data = self._parse_data_from_root(root)
        self.meta = self._parse_meta_from_root(root)
    
    @classmethod
    def _parse_meta_from_root(cls, r):
        trnsl = {'Id':'Index'}
        trnsf = {'Id':int,'Capacity':int}
        keys = r.xpath('/html//div[@class="teamGroupings view"]/dl/dt/text()')
        vals = r.xpath('/html//div[@class="teamGroupings view"]/dl/dd/text()')
        meta = SimpleNamespace(**{trnsl.get(k,k):trnsf.get(k, lambda x:x)(v.strip()) for k,v in zip(keys,vals)})
        return meta

    _invite_parser = XpathParser('code/text()')
    @classmethod
    def _members_parser(cls, e):
        e_as = e.xpath('a')
        if len(e_as) == 2:
            e_a = e_as[1]
            return e_a.attrib['href'],e_a.text
        else:
            return None,None
        return e
    
    def _parse_data_from_root(self, root):
        tbl = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]//h3[text()="Related Teams"]/following-sibling::table[@class="table table-striped"]')[0]
        df = self._parse_table(tbl, only_sortable=False, parsers={
            'Created':'_date', 'Modified':'_date', 'Invitecode':self._invite_parser,
            'Founder Id':'_a',
            'Members':self._members_parser
        })
        df = self._expand_a(df,'Founder Id',url_var='FounderUrl',text_var='FounderName',append_names=False)
        df = self._expand_a(df,'Members',url_var='PartnerUrl',text_var='PartnerName',append_names=False)
        df = self._strip_urls(df)
        return df
