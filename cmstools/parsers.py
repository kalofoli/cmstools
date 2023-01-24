'''
Created on Jan 8, 2023

@author: janis
'''

import typing
import itertools
from types import SimpleNamespace
import re

import numpy as np
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
    '_date': '_span_tt',
    '_span_tt':XpathParser(r'span/@title'),
    '_div_tt':XpathParser(r'div/@title'),
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
        data = itertools.zip_longest(*df[column])
        dct = dict(itertools.zip_longest(names,data,fillvalue=[]))
        df = pd.concat((df, pd.DataFrame(dct)), axis=1)
        if drop:
            df = df.drop(column, axis=1)
        return df
        
    @classmethod
    def _extract_id(cls, df, column, var_name='Id', append_name=False, dtype=pd.Int64Dtype(), inplace=False, drop=False):
        if not inplace:
            df = df.copy()
        out_var = column+var_name if append_name else var_name
        idx = df[column].astype(object).replace(np.nan,None).str.replace('.*/([0-9]+)$',r'\1',regex=True)
        if dtype is not None:
            idx = idx.astype(dtype)
        df[out_var] = idx
        if drop:
            del df[column]
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
        df[cols] = df[cols].apply(lambda x:x.astype(object).replace(np.nan,None).str.replace('^/eml22','',regex=True))
        return df

    
    @classmethod
    def _parse_table_row(cls, e_rows, parsers={}, names=[], only_columns=None):
        
        def parse_data(i,n,e):
            parser = parsers.get(i)
            if parser is None and n is not None:
                parser = parsers.get(n)
            if parser is None:
                parser = parsers.get(None)
            while parser is not None and not isinstance(parser, typing.Callable):
                parser = parsers.get(parser)
            if parser is None:
                raise ValueError(f'Could not find a valid parser for entry with name {n} at index {i}.')
            return parser(e)
        parsers = {**default_parsers, **parsers}
        is_ok = only_columns.__getitem__ if only_columns is not None else lambda _: True
        rows = []
        for _,e_row in enumerate(e_rows):
            row = [parse_data(i,n,e) for i,(n,e) in enumerate(itertools.zip_longest(names, e_row)) if is_ok(i)]
            rows.append(row)
        return rows
    
    def _parse_table(self, tbl, only_sortable=False, parsers={}, skip_rows=0):
        '''
        @param parsers: a dictionary specifying the parser for each element.
            Each key is either the name of a columnt or a column index.
            Each value is either a callable that will be used as a parser with
            a single argument the etree Element to parse, or a key that will be
            further looked up in the parsers dictionary.
            A key of None is used as a fallback.
            This value is expanded by the values of the default_parsers dict.
        '''
        def parse_head(e):
            c = e.getchildren()
            return (c[0].text,True) if c else (e.text,False)
        
        e_trs = tbl.xpath('tr | thead/tr | tbody/tr')
        names,sortable = zip(*map(parse_head, e_trs[skip_rows].xpath('th')))
        if only_sortable:
            only_columns = sortable
            names_s = [n for i,n in enumerate(names) if sortable[i]]
        else:
            only_columns = None
            names_s = names
            
        elems = (e_tr.xpath('td') for e_tr in e_trs[skip_rows+1:])
        rows = self._parse_table_row(elems, names=names, parsers=parsers, only_columns=only_columns)
        
        return pd.DataFrame(rows, columns=names_s)
    
    def __len__(self): return self.data.shape[0]
    def __repr__(self): return f'<{type(self).__name__} with {len(self)} entries>'

class HTMLParserWithMeta(HTMLParser):
    def __init__(self, root):
        super().__init__(root)
        self.meta = self._parse_meta_from_root(root)
    def __repr__(self): return f'<{type(self).__name__}[{self.meta.Name}] with {len(self)} entries>'
    
class SubmissionListParser(HTMLParser):
    rex_ass = re.compile('(?i)^.*Assignment\s*(?P<match>[0-9]+)(\s.*)?$')
    rex_prb = re.compile('(?i)^.*Problem\s*(?P<match>[0-9]+)(\s.*)?$')
    rex_bon = re.compile('(?i)^.*Bonus.*$')
    rex_coc = re.compile('(?i)^.*code\s*of\s*conduct\s*.*$')
    @classmethod
    def _parse_name(cls, df):
        s_name = df.Name
        idl_m = s_name.str.match(cls.rex_ass)
        assert all(idl_m),f'Could not match some names in the listing. Please correct them: <{">, <".join(s_name.loc[~idl_m])}>.'
        s_ass = s_name.str.replace(cls.rex_ass, r'\g<match>').astype(int)
        idl_m = s_name.str.match(cls.rex_prb)
        
        s_prb = pd.Series('',index=s_name.index)
        s_prb[idl_m] = s_name[idl_m].str.replace(cls.rex_prb,r'\g<match>')
        s_prb[s_name.str.match(cls.rex_bon)] = 'bonus'
        s_prb[s_name.str.match(cls.rex_coc)] = 'coc'
        return df.assign(Assignment=s_ass, Problem=s_prb)
        
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
        df = self._extract_id(df, column='PointsUrl', var_name='TestingId')

        df = self._parse_name(df)
        return df

class SubmissionItemParser(HTMLParser):
    def _parse_data_from_root(self, root):
        tbl = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]/table[@class="table table-striped"]')[0]
        df = self._parse_table(tbl, only_sortable=False, parsers={
            'Student': '_a', 'Date':'_date', 'Mat.Nr':'_a_text',
            'Team':'_a',
            9:XpathParser('a/@href',sep=None)
        })
        df = df.rename({df.columns[9]:'_actions_'},axis=1)
        df = self._expand_tuple(df,'_actions_',['FileUrl','FeedbackUrl'], append_names=False, drop=True)
        df = self._expand_a(df,'Student')
        df = self._expand_a(df,'Team')
        df = self._strip_urls(df)#, ['FileUrl','FeedbackUrl','StudentUrl','TeamUrl'])
        try:
            df = self._extract_id(df, column='TeamUrl', var_name='TeamId')
        except Exception as e:
            raise e
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
        df = self._extract_id(df, column='FounderUrl', var_name='FounderMN', dtype=object)
        df = self._extract_id(df, column='PartnerUrl', var_name='PartnerMN', dtype=object)
        return df

class TestingParser(HTMLParser):
    def __init__(self, root, named_columns=False,name_with_testings=True,only_grades=False):
        self.named_columns = named_columns
        self.name_with_testings = name_with_testings
        self.only_grades = only_grades
        super().__init__(root)
    rex_input = re.compile(r'^data\[(?P<idx>[0-9]+)\]\[Testingresult\]\[(?P<testing>[0-9]+)\]\[(?P<kind>[^]]+)\]$')
    def _parse_grade_value(self, e):
        assert e.attrib['class']=='testingresults-input',f'Tried to parse grade value from an element with class he wrong class "{e.attrib["class"]}" which is not "testingresults-input" as expected.'
        fn = lambda x:self.rex_input.match(x).group('kind')
        return {fn(e.attrib['name']):e.attrib['value'] if 'value' in e.attrib else 'None' for e in e.xpath('.//input')}
    def _parse_grade_info(self, e):
        assert e.attrib['class']=='testingresults-button',f'Tried to parse grade info from an element with class he wrong class "{e.attrib["class"]}" which is not "testingresults-button" as expected.'
        return {e.attrib['name']:e.attrib['value'] if 'value' in e.attrib else 'None' for e in e.xpath('.//input')}
        
    def _parse_data_from_root(self, root):
        e_ths = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]//div[@id="student-entries"]/table/tr/th')
        
        n = len(e_ths)
        rex = re.compile('[()\n]')
        parse_testing_header = lambda e: rex.sub('',e.xpath('.//div/@title')[0])
        parsers = {0:'_text', 1:'_text',n-1:'_text',None:parse_testing_header}
        theads = self._parse_table_row([e_ths], parsers=parsers)[0]
        assert theads[:2]+[theads[n-1]] == ['Mat.','Name','Result'], f'Mismatched testings header. Expected Mat. Name ... Results but found {theads}'
 
        e_trs = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]//div[@id="student-entries"]/form/table/tr')
        e_tds = (e_tr.getchildren() for e_tr in e_trs)
        parsers = {0:'_text', 1:'_text',n-1:'_text',n:self._parse_grade_info, None:self._parse_grade_value}
        
        columns = [True]*n+[False]*2
        trows = self._parse_table_row(e_tds, parsers=parsers, only_columns=columns)
        entries = []
        for trow in trows:
            dct_info = dict(mn=trow[0])
            if not self.only_grades:
                dct_info = {**dct_info, **dict(name=trow[1], student_id=trow[2]['student_id'],result=trow[-1])}
            dct_pts = {self._format_column(d["testing_id"]):d["points"] for d in trow[2:-1]}
            entries.append({**dct_info,**dct_pts})
        df = pd.DataFrame(entries).replace('None','')
        if self.named_columns:
            dct_ren = {self._format_column(d["testing_id"]):self._format_column(d["testing_id"],name) for d,name in zip(trow[2:-1], theads[2:-1])}
            df = df.rename(dct_ren,axis=1)
            grade_columns = dct_ren.values()
        else:
            grade_columns = dct_pts.keys()
        self.grade_columns = list(grade_columns)
        return df
    rex_bonus = re.compile('^Bonus\s+(?P<idx>[0-9]+)\sBonuses$')
    rex_ass = re.compile('^Problem\s+(?P<pid>[0-9]+)\sAssignment\s+(?P<aid>[0-9]+)$')
    def _format_column(self, tid, name=None):
        if name is None:
            return f'testing_{tid}'
        prefix = f'testing_{tid}_' if self.name_with_testings else ''
        m = self.rex_bonus.match(name)
        if m is not None:
            return f'{prefix}a{m.group("idx")}_pb'
        m = self.rex_ass.match(name)
        if m is not None:
            return f'{prefix}a{m.group("aid")}_p{m.group("pid")}'
        raise ValueError(f'Could not parse name {name} for test index {tid}')

class StudentParser(HTMLParser):
    rex_mn = re.compile('.*/(?P<mn>[0-9]+)')
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
        tbl = root.xpath('/html/body/div[@class="container"]/div[@class="row"]//div[@class="content"]//table[@id="studentstable"]')[0]
        df = self._parse_table(tbl, only_sortable=True, parsers={
            'Nr.':'_div_tt',
            'Matr.#':'_a','Joined at':'_date', 'Tutorial':'_a','Email':'_a_text'
        }, skip_rows=1)
        
        df = df.rename({'Nr.':'Id','Matr.#':'Mtr.Nr','Lastname':'Name','Joined at':'Joined',None:'Semester'},axis=1)
        
        df = self._expand_a(df,'Mtr.Nr',url_var='StudentUrl',text_var='MN',append_names=False)
        df = self._expand_a(df,'Tutorial',url_var='TutorialUrl',text_var='TutotialName',append_names=False)
        df = self._strip_urls(df)
        df = self._extract_id(df, column='TutorialUrl', var_name='Tutorial', dtype=object)
        df = df.assign(Id = lambda x:x.Id.str.replace('DB-Id: ','').astype(int))
        return df

    
    