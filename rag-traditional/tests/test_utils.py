from src.utils import clean_title


def test_clean_title():
    """Test the clean_title function with various title formats."""
    # Test normal title without brackets
    assert clean_title("Normal Title") == "Normal Title"

    # Test title with brackets
    assert clean_title("[Bracketed Title]") == "Bracketed Title"

    # Test title with whitespace around brackets
    assert clean_title("  [Bracketed Title]  ") == "Bracketed Title"

    # Test title with whitespace inside brackets
    assert (
        clean_title("[  Bracketed Title with spaces  ]")
        == "Bracketed Title with spaces"
    )

    # Test empty title
    assert clean_title("") == ""

    # Test None title
    assert clean_title(None) == None

    # Test title with only the opening bracket
    assert clean_title("[Incomplete bracket") == "[Incomplete bracket"

    # Test title with only the closing bracket
    assert clean_title("Incomplete bracket]") == "Incomplete bracket]"

    # Test title with nested brackets
    assert clean_title("[Outer [Inner] bracket]") == "Outer [Inner] bracket"

    # Test title with brackets followed by period
    assert clean_title("[Bracketed Title].") == "Bracketed Title"

    # Test title with brackets followed by period and space
    assert clean_title("[Bracketed Title]. Some text") == "[Bracketed Title]. Some text"
