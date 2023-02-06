import re
import typing
import functools

import gspread



def dict_merge(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            dict_merge(node, value)
        else:
            destination[key] = value
    return destination

class BorderSpecification:
    '''
    Border specification
    
    The format is loc:style
    
    where loc is the border location specification (or a preset that contains a tuple of such strings).
    style is either a dict, or a string of a preset or a preset function, or a tuple of the above.
    All resolved dicts are merged.
    Examples:
    'all:none'
    'all:style(solid)+red(.5)+green(.4)'
    ('all:red','left:green')
    
    
    '''
    
    _border_presets = {
        'all': ('top','bottom','left','right','innerHorizontal','innerVertical'),
        'inner':('innerVertical','innerHorizontal'),
        'outer':('top','bottom','left','right'),
        'none':'style(NONE)',
        'dotted':'style(DOTTED)',
        'dashed':'style(DASHED)',
        'solid': 'style(SOLID)',
        'solid_medium':'style(SOLID_MEDIUM)',
        'solid_thick':'style(SOLID_THICK)',
        'none':'style(NONE)',
        'double':'style(DOUBLE)',
        'style':lambda x='NONE':dict(style=x),
        'red':lambda x=1:dict(colorStyle=dict(rgbColor=dict(red=x))),
        'green':lambda x=1:dict(colorStyle=dict(rgbColor=dict(green=x))),
        'blue':lambda x=1:dict(colorStyle=dict(rgbColor=dict(blue=x))),
    }
    def __init__(self, presets={}):
        self.presets = {**self._border_presets, **presets}
        self._border_specs = {}

    rex_body_fn=re.compile('^(?P<name>[^()]+)\((?P<arg>[^[()]+)\)$')
    @classmethod
    def parse_body(cls, spec, presets):
        if isinstance(spec, dict):
            return spec
        elif spec in presets:
            r = presets[spec]
        elif isinstance(spec, str):
            if spec in presets:
                r = presets[spec]
            elif '+' in spec:
                r = tuple(spec.split('+'))
            else:
                m = cls.rex_body_fn.match(spec)
                if m is not None:
                    fn = presets.get(m.group('name'))
                    if fn is None:
                        raise KeyError(f'Requested function {m.group("name")} which was not found in the presets.')
                    elif not isinstance(fn, typing.Callable):
                        raise TypeError(f'Requested function {m.group("name")} which yielded {fn!r} of non-function type {type(fn).__name__}.')
                    r = fn(m.group('arg'))
                else:
                    raise KeyError(f'Requested specification "{spec}" which was not found in the presets.')
        elif isinstance(spec, typing.Callable):
            r = spec()
        elif isinstance(spec, tuple):
            parts = (cls.parse_body(s, presets) for s in spec)
            r = functools.reduce(dict_merge, parts, {})
        else:
            raise TypeError(f'Could not parse border body from specification "{spec!r}" of type {type(spec).__name__}.')
        return cls.parse_body(r, presets)

    def parse_border(self, spec):
        presets = self.presets
        if isinstance(spec, dict):
            return spec
        if isinstance(spec, tuple):
            parts = map(self.parse_border,spec)
            r = functools.reduce(dict_merge, parts, {})
        elif isinstance(spec, str):
            if spec in self._border_specs:
                return self._border_specs[spec]
            if ':' in spec:
                parts = spec.split(':')
                assert len(parts) == 2,f'The correct format is loc:body, but received "{spec}".'
                loc,body = parts
                body = self.parse_body(body, self.presets)
                loc = presets.get(loc,(loc,))
                assert all(isinstance(l,str) for l in loc),f'The location part must be a string or a preset of strings, but {loc} was found.'
                r = functools.reduce(dict_merge, ({l:body} for l in loc), {})
            else:
                r = presets.get(spec)
                if r is None:
                    raise KeyError(f'No preset "{spec}" known. Try one of: {",".join(presets)}.')
            r = self.parse_border(r)
            self._border_specs[spec] = r
            return r
        else:
            raise TypeError(f'Could not parse border from specification{spec!r} of type {type(spec).__name__}. Must be either a string, dict, or tuple of the above.')
        return self.parse_border(r)
    __call__ = parse_border
    
    
def update_borders(self, borders, presets={}):
        """Update the content of the notes located at `cell`.
        @param notes A sequence of (cell, content) pairs, where cell is a string with cell coordinates in A1 notation,
            e.g. 'D7', and content is a string with the text note to insert.
        """
        bs = BorderSpecification(presets)
        
        requests = []
        for cell, content in borders:
            grid_range = gspread.utils.a1_range_to_grid_range(cell, self.id)
            request = dict(updateBorders={
                "range": grid_range,
                **bs(content)
            })
            requests.append(request)
        body = dict(requests=requests)
        self.spreadsheet.batch_update(body)
gspread.Worksheet.update_borders = update_borders

def update_notes(self, notes):
        """Update the content of the notes located at `cell`.
        @param notes A sequence of (cell, content) pairs, where cell is a string with cell coordinates in A1 notation,
            e.g. 'D7', and content is a string with the text note to insert.
        """

        requests = []
        for cell, content in notes:
            grid_range = gspread.utils.a1_range_to_grid_range(cell, self.id)
            request = dict(updateCells={
                "range": grid_range,
                "rows": [{"values": [{"note": content}]}],
                "fields": "note",
            })
            requests.append(request)
        body = dict(requests=requests)
        self.spreadsheet.batch_update(body)
gspread.Worksheet.update_notes = update_notes
