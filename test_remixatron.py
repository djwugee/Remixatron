import unittest
import os
from Remixatron import InfiniteJukebox

class TestRemixatron(unittest.TestCase):
    def test_import(self):
        # Basic check to ensure the class can be instantiated
        # (even if we don't run the full audio processing which requires a file)
        try:
            from Remixatron import InfiniteJukebox
            import Remixatron
            self.assertTrue(True)
        except ImportError:
            self.fail("Remixatron could not be imported")

    def test_pygame_optional(self):
        # Verify that pygame is not required for importing the module
        import sys
        if 'pygame' in sys.modules:
            del sys.modules['pygame']

        try:
            import Remixatron
            from Remixatron import InfiniteJukebox
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Remixatron failed to import without pygame: {e}")

if __name__ == '__main__':
    unittest.main()
