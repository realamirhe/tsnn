import string

words_spacing_gap = 1
letters = string.ascii_lowercase
language = letters + " "  # out_of_vocab_separator
words = ["abc", "omn"]
words_capture_window_size = words_spacing_gap + max(map(len, words))
UNK = "UNK"