import unittest

import tokenizer


def sut():
    return tokenizer.IndexPairTokenizer()


def test_single_char():
    tok = sut()

    tok.train("a", 1)
    assert tok.vocab_size == 1

    tok.train("a", 2)
    assert tok.vocab_size == 1

    tok.train("aa", 2)
    assert tok.vocab_size == 1

    assert tok.encode("aaaa") == [0, 0, 0, 0]
    assert tok.decode([0, 0, 0, 0]) == "aaaa"


def test_two_chars():
    tok = sut()

    tok.train("ab", 2)
    assert tok.vocab_size == 2
    assert tok.encode("ababab") == [0, 1, 0, 1, 0, 1]
    assert tok.decode([1, 0]) == "ba"

    tok.train("ababab", 3)
    assert tok.encode("ab") == [2]
    assert tok.encode("abaa") == [2, 0, 0]
    assert tok.encode("aab") == [0, 2]
    assert tok.encode("aaba") == [0, 2, 0]
    assert tok.encode("aabb") == [0, 2, 1]


def test_three_chars():
    tok = sut()

    tok.train("abcabcabab", 5)
    assert tok.encode("abc") == [4]
    assert tok.encode("ababcababb") == [3, 4, 3, 3, 1]
    assert tok.encode("aaaa") == [0, 0, 0, 0]


if __name__ == '__main__':
    unittest.main()
