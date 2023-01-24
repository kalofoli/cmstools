'''
Created on Jan 20, 2023

@author: janis
'''
import unittest
from cmstools.controllers import SheetController, CMSController
from cmstools.cms import CMSSession


class TestReport(unittest.TestCase):


    def setUp(self):
        self.sc = SheetController.from_credentials(filename='../credentials.json', tag='devel')
        cms = CMSSession('mhuuvusauma7nv2ojfsrkph0f5', cache='../cache', store=True)
        self.cmc = CMSController(cms)

    def testSubmissionItem(self):
        self.cmc.get_submission_item(37)

    def testReport(self):
        html = self.cmc.report(self.sc, [1], dry=True)
        print(html)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()