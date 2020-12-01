from nltk.util import ngrams


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def load_data(infile):
    sent_list = []
    with open(infile, 'r') as infile:
        for line in infile:
            # print(line)
            line = line.rstrip('\n')
            line = line.split()
            # print(line)
            sent_list.append(line)
    return sent_list


def main():
    # without_sent_list = load_data('output_task_4/model_translations_without.out')
    # with_sent_list = load_data('output_task_4/model_translations.out')
    # without_sent_list = [without_sent_list[i:i + 3] for i in range(0, len(without_sent_list), 3)]
    # with_sent_list = [with_sent_list[i:i + 3] for i in range(0, len(with_sent_list), 3)]
    # without_score = []
    # with_score = []
    # for i, j in zip(without_sent_list, with_sent_list):
    #     without_score.append(distinct_n_corpus_level(i, 2))
    #     with_score.append(distinct_n_corpus_level(j, 2))
    #     # print(i)
    #     # print(j)
    # print(without_score[-2:])
    # print(with_score[-2:])
    # print(f'Without avg: {sum(without_score)/len(without_score)}')
    # print(f'With avg: {sum(with_score)/len(with_score)}')

    without_sent_list = load_data('output_task_4/model_translations_without.out')
    with_sent_list = load_data('output_task_4/model_translations.out')
    gamma_10_sent_list = load_data('output_task_4/model_translations_gamma_10.out')
    gamma_100_sent_list = load_data('output_task_4/model_translations_gamma_100.out')
    for i in range(1, 4):
        print(f'distinct {i}:')
        print(f'without: {distinct_n_corpus_level(without_sent_list, i)}')
        print(f'with: {distinct_n_corpus_level(with_sent_list, i)}')
        print(f'gamma 10: {distinct_n_corpus_level(gamma_10_sent_list, i)}')
        print(f'gamma 100: {distinct_n_corpus_level(gamma_100_sent_list, i)}')
        print('='*30)


if __name__ == '__main__':
    main()
