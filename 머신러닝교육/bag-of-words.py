import re

special_chars_remover = re.compile("[^\w'|_]")

def main():
    sentence = input()
    bow = create_BOW(sentence)

    print(bow)


def create_BOW(sentence):
    bow = {}
    sentence = sentence.lower()
    sentence = remove_special_characters(sentence)
    splitted_sentence = sentence.split()
    splitted_sentence_filtered = [
        token
        for token in splitted_sentence
        if len(token) >= 1
    ]
    
    for token in splitted_sentence_filtered:
        bow.setdefault(token,0)
        bow[token] += 1
    return bow


def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)


if __name__ == "__main__":
    main()