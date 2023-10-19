import pickle

pronouns_tag_conversion = {
    "he": "M",
    "she": "F",
    "him": "M",
    "her": "F",
    "his": "M",
    "hers": "F",
}

if __name__ == "__main__":
    pro_stereotyped = open("pro_stereotyped_type2.txt.test", "r").readlines()
    anti_stereotyped = open("anti_stereotyped_type2.txt.test", "r").readlines()

    prompts = []
    for i, (pro_line, anti_line) in enumerate(zip(pro_stereotyped, anti_stereotyped)):
        # stereotypical answer is second word in square brackets in pro_line
        # anti-stereotypical answer is second word in square brackets in anti_line
        pro_answer = pro_line.split("[")[2].split("]")[0]
        #anti_answer = " " + anti_line.split("[")[2].split("]")[0]
        occupation = " " + pro_line.split("[")[1].split("]")[0].split(" ")[1]

        # prompt is everything before the second `[`, replacing the first `[` with `` and the first `]` with ``
        #prompt = "".join(pro_line[pro_line.index(" ")+1:].split("[")[:2]).replace("[", "").replace("]", "")[:-1]

        # there are now spaces in front of apostrophes, so remove them
        #prompt = prompt.replace(" '", "'")

        prompts.append({
            "gender": pronouns_tag_conversion[pro_answer],
            "occupation": occupation,
        })
    
    with open("stereotyped_prompts.pkl", "wb") as f:
        pickle.dump(prompts, f)