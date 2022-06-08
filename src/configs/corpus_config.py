import string

words_spacing_gap = 10
letters = string.ascii_lowercase
language = letters + " "  # out_of_vocab_separator
words = ["abc", "omn"]  # , "can"]
words_capture_window_size = words_spacing_gap + max(map(len, words))
UNK = "UNK"
