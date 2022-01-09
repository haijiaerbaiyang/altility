import unittest
import sys
sys.path.append('/altility/src')
import altility

class TestSomeFunctions(unittest.TestCase):

    """ Tests functions defined in data.py
    """

    @classmethod
    def setUpClass(cls):
        
        """ Runs once before the first test.
        """

        pass


    @classmethod
    def tearDownClass(cls):
        
        """ Runs once after the last test.
        """

        pass


    def setUp(self):
        
        """ Runs before every test.
        """
        
        pass
        

    def tearDown(self):

        """ Runs after every test.
        """
        
        pass
        
        
    def test_somefirstfunction(self):
    
        print('the file test/unit/test_somefunctions.py is successfully executed through docker.')

if __name__ == '__main__':

    unittest.main()

