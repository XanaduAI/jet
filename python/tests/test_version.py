import re

import jet


class TestVersion:
    def test_attribute(self):
        """Tests that the version attribute has the correct form."""
        semver_pattern = re.compile(r"^\d+\.\d+\.\d+$")
        assert semver_pattern.match(jet.__version__)

    def test_function(self):
        """Tests that the version attribute matches the version function."""
        assert jet.__version__ == jet.version()
