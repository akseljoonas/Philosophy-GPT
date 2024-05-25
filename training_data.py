# getting the original txt file
def get_txt_file():
    nietzsche_txt = open('/Users/mikaumana/Documents/PycharmProjects/NietzscheGPT/nietzsche.txt', 'r')
    pre_nietzsche_data = nietzsche_txt.read()
    return pre_nietzsche_data
def remove_char(pre_nietzsche_data):
    n_nietzsche_data = pre_nietzsche_data.replace("_", "")
    nietzsche_data = n_nietzsche_data.replace("$", "")
    return nietzsche_data


def get_info(nietzsche_data):
    print(len(nietzsche_data))
    tokens = []
    lettercount = []
    for x in nietzsche_data:
        if x not in tokens:
            tokens.append(x)
    print(len(tokens))
    print("\n")
    print(tokens)

    char_counts = {}

    for char in nietzsche_data:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1

    for char, count in char_counts.items():
        print(f"'{char}' occurs {count} times.")


def save_data(nietzsche_data):
    with open('new_nietzsche.txt', 'w') as output:
        output.write(nietzsche_data)

if __name__ == "__main__":
    x = get_txt_file()
    y = remove_char(x)
    get_info(y)
    save_data(y)


