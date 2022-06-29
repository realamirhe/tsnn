import string

words_spacing_gap = 7
letters = string.ascii_lowercase
language = letters + " "  # out_of_vocab_separator
words = ["abc", "mno"]
words_capture_window_size = words_spacing_gap + max(map(len, words))
words_average_size_occupation = words_spacing_gap + sum(map(len, words)) / len(words)
UNK = "UNK"
