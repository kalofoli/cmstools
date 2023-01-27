'''
Created on Jan 20, 2023

@author: janis
'''
import unittest
from cmstools.controllers import SheetController, CMSController
from cmstools.cms import CMSSession


class Test(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sc = SheetController.from_credentials(filename='../notebooks/credentials.json', tag='devel')
        cms = CMSSession('hcjr8gh98k2t255tso3382mknk', cache='../notebooks/cache', store=True)
        self.cmc = CMSController(cms) 
        
    def testAnnotate(self):
        self.sc.annotate_from_cms(self.cmc, [4], ['bonus'], notes=False, highlight=False)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()