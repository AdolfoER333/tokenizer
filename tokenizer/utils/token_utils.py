import regex


def get_int_rep(text: str) -> list[list[int]]:
    """
    Convert the input string into utf-8 standard which is, then, converted to its
    integer representation. Given that the string will be broken in some parts to
    avoid forming tokens with certain pairs by using the regex_pattern, the result
    is a list of said integer representations.

    The regex_pattern was taken from the gpt-4 tokenizer written in the tiktoken repository, broken into parts for
    easier understanding, merged back and can be found at:
    https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

    """
    # Parts of the regex with descriptive names
    contractions = r"'(?i:[sdmt]|ll|ve|re)"
    letters_with_optional_preceding_char = r"[^\s\r\n\p{L}\p{N}]?+\p{L}+"  # Modified to not include whitespace
    digits_1_to_3 = r"\p{N}{1,3}"
    non_alphanumeric_with_optional_space = r" ?[^\s\p{L}\p{N}]++[\r\n]*"
    whitespace_followed_by_newline = r"\s*[\r\n]"
    whitespace_not_followed_by_non_whitespace = r"\s+(?!\S)"
    whitespace = r"\s+"

    # Final regex as one string
    regex_pattern = (
        fr"{contractions}|"
        fr"{letters_with_optional_preceding_char}|"
        fr"{digits_1_to_3}|"
        fr"{non_alphanumeric_with_optional_space}|"
        fr"{whitespace_followed_by_newline}|"
        fr"{whitespace_not_followed_by_non_whitespace}|"
        fr"{whitespace}|"
    )
    groups = regex.findall(regex_pattern, text)
    binaries = [list(map(int, group.encode('utf-8'))) for group in groups]
    return binaries


def get_pair_counts(list_of_int_reps: list[int]) -> dict[tuple[int, int], int]:
    """
    Outputs a dictionary with the keys being pairs found in list_of_int_reps and
    values are that pair's number of sightings.

    """
    counts = {}
    # Iterate over a group and the same group shifted by 1 element
    for pair in zip(list_of_int_reps, list_of_int_reps[1:]):
        # Start with 1 occurrence or sum 1 occurrence
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def aggregate_pair_counts(
        list_of_pair_counts: list[dict[tuple[int, int], int]]
) -> dict[tuple[int, int], int]:
    """
    Simply put, iterates one or more outputs of the counting function returning a
    unified count, having the same format as said function's output but grouping
    counts that, before, were separate.

    """
    all_counts = {}
    for count_dct in list_of_pair_counts:
        for key in count_dct:
            all_counts[key] = all_counts.get(key, count_dct[key]) + 1
    return all_counts


def collect_counts(list_of_int_reps: list[list[int]]) -> dict[tuple[int, int], int]:
    pair_counts = [get_pair_counts(pair) for pair in list_of_int_reps]
    agg_counts = aggregate_pair_counts(pair_counts)
    return agg_counts


def replace_with_new_token(
    list_int_reps: list[list[int]],
    pair_to_replace: tuple[int, int],
    new_id: int,
    verbose: bool = False
) -> list[list[int]]:
    """
    Starting for the current state of tokens, generates a new_lst that contains
    the same structure as list_int_reps, but with merged new tokens.

    """
    new_lst_agg = []

    if verbose:
        print(f'Merging the pair {pair_to_replace} into token number {new_id}')

    # Iterate through all sub-texts
    for lst in list_int_reps:
        if len(lst) < 2:
            continue
        new_lst = []
        idx = 0
        # On the original list, for every element, check if corresponding pair
        while idx < len(lst):
            if idx == len(lst) - 1:
                new_lst.append(lst[-1])
                break
            # If the pair was found, skip next element because it will be merged
            if (lst[idx], lst[idx+1]) == pair_to_replace:
                new_lst.append(new_id)
                idx += 2
            # If not the sought out pair, skip to next element and keep it
            else:
                new_lst.append(lst[idx])
                idx += 1
        # Last element of the list (wasn't past by in the loop)
        new_lst_agg.append(new_lst)
    return new_lst_agg


def merge_new_tokens(list_int_reps: list[list[int]], num_to_merge: int
                     ) -> (list[list[int]], dict[tuple[int, int], int]):
    """
    Using the pair-by-pair merger function, turns the original list_int_reps into a modifier version that underwent the
    number of merges specified on the argument num_to_merge.

    The objetive is recalculating the counts for every iteration, so that new merged tokens can be counted and, if
    necessary, be part of the next merged token.

    """
    num_of_merged = 0
    reps = list(list_int_reps)
    next_id = 256
    new_ids = {}

    while num_of_merged < num_to_merge:
        counts = collect_counts(reps)
        pair_to_merge = max(counts, key=counts.get)
        new_ids[next_id] = pair_to_merge

        # New token list of lists after merging
        reps = replace_with_new_token(reps, pair_to_merge, next_id, verbose=True)

        # Modify loop control variables
        num_of_merged += 1
        next_id += 1
    return reps, new_ids


def text_to_merges(text: str, vocab_size: int) -> dict[int, tuple[int, int]]:
    int_reps = get_int_rep(text)
    new_int_reps, merges = merge_new_tokens(int_reps, num_to_merge=vocab_size-256)
    return merges
