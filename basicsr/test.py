"""Main test module for the PADUM project."""
import unittest

# Import test cases from submodules
from basicsr.data.tests import test_dataset_paired_image
from basicsr.metrics.tests import test_psnr_ssim
from basicsr.models.tests import test_padm_architecture

# Create test suite
def suite():
    """Create a test suite containing all test cases."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(test_dataset_paired_image.TestPairedImageDataset))
    test_suite.addTest(unittest.makeSuite(test_psnr_ssim.TestPSNRSIM))
    test_suite.addTest(unittest.makeSuite(test_padm_architecture.TestPADMArchitecture))
    return test_suite

if __name__ == '__main__':
    """Run all tests when executed directly."""
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())