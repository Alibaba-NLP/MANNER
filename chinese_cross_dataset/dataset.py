


def read_examples_from_file(file_path):

    guid_index = 0
    examples = []
    type_sent_map = defaultdict(set)

    num_sent = 0
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if words:
                    examples.append(InputExample(guid="{}".format(guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
                    num_sent += 1
            else:
                splits = line.split()
                # append words 
                words.append(splits[0])
                if len(splits) > 1:
                    typ = splits[-1].replace("\n", "").replace("B-", "").replace("I-", "")
                    labels.append(splits[-1].replace("\n", ""))
                    if typ != "O":
                        type_sent_map[typ].add(num_sent)
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}".format(guid_index), words=words, labels=labels))
    
    final_type_sent_map = {k: list(v) for k, v in type_sent_map.items()}

    for k in type_sent_map.keys():
        np.random.shuffle(final_type_sent_map[k])

    return examples, final_type_sent_map



def read_and_load_examples(args, tokenizer, mode):

    # data_dir = "data/cross_domain_ccks2020"
    data_dir = os.path.join(args.data_dir, "cross_domain_ccks2020")

    examples, type_sent_map = read_examples_from_file(os.path.join(data_dir, "{}.txt".format(mode)))
    features = [convert_example_to_features(args, tokenizer, example) for example in examples]

    return features, type_sent_map

