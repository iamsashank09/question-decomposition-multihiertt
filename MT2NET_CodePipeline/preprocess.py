import json
import copy
import argparse


# Split the data into long and short questions
def splitdata(file_path):

    sep = "*" * 10
    print(sep)
    print("Splitting Data")
    print(sep)

    f = open(file_path)
    data = json.load(f)
    keys = data[0].keys()

    long_qstns = []
    short_qstns = []


    for i in range(len(data)):
        qstn = data[i]['qa']['question']

        if len(qstn.split(' ')) > 13:
            long_qstns.append(data[i])
        else:
            short_qstns.append(data[i])

    files = ['./converted_data/long_qstns.json', './converted_data/short_qstns.json']
    combined = [long_qstns, short_qstns]

    for i in range(len(files)):
        json.dump(combined[i], open(files[i], "w"), indent=4)



# Process Data Decomposition
def data_input_to_decomposition(file_path):
    f = open(file_path)
    data = json.load(f)

    input_to_decomp = {
        "data":[]
    }

    for i in range(len(data)):
        long_qstns_data = {}
        long_qstns_data ['title'] = ""
        long_qstns_data['paragraphs'] = [
            {
                "context": data[i]['paragraphs'],
                "qas": [
                        {
                        "question": data[i]['qa']['question'],
                            "id": data[i]['uid']
                        }
                    ]

            }
        ]

        input_to_decomp['data'].append(long_qstns_data)
        

    json.dump(input_to_decomp, open("../QuestionDecomposition_CodePipeline/data/dev.json", "w"), indent=4)


# Re convert decomposed questions

def reconvert_decomposed_questions_to_mhert():

    f = open("../QuestionDecomposition_CodePipeline/decomposed_questions/decomposed_questions.json")
    decomp_data = json.load(f)

    f = open("./converted_data/long_qstns.json")
    original_data = json.load(f)

    combine_qstns_1 = []
    combine_qstns_2 = []

    for i in decomp_data.keys():
        for j in original_data:
            if i == j['uid']:

                if len(decomp_data[i]['decomposed_questions']) == 1:
                    j['uid'] = i
                    j['qa']['question'] = decomp_data[i]['decomposed_questions'][0]
                    combine_qstns_2.append(j)
                else:
                    for k in range(len(decomp_data[i]['decomposed_questions'])):

                        tmp = copy.deepcopy(j)
                        tmp['uid'] = i
                        tmp['qa']['question'] = decomp_data[i]['decomposed_questions'][k]

                        if k == 0:
                            combine_qstns_1.append(tmp)
                        else:
                            combine_qstns_2.append(tmp)


                break


    json.dump(combine_qstns_1, open("./converted_data/sub_questions1_converted_to_mhert.json", "w"), indent=4)
    json.dump(combine_qstns_1, open("./input_data/MT2NET_input.json", "w"), indent=4)


    json.dump(combine_qstns_2, open("./converted_data/sub_questions2_converted_to_mhert.json", "w"), indent=4)


# Insert answers and combine with the short questions
def insert_answers_to_decomposed_questions():
    f = open("./outputs_intermediate/final_predictions/MT2NET_output_predictions.json")
    predicted_ansrs = json.load(f)
    json.dump(predicted_ansrs, open("./outputs_intermediate/final_predictions/sub_questions_predictions.json", "w"), indent=4)


    f = open("./converted_data/sub_questions2_converted_to_mhert.json")
    fit_ansrs = json.load(f)

    for i in range(len(fit_ansrs)):
        for j in range(len(predicted_ansrs)):
            if str(fit_ansrs[i]["uid"]) == str(predicted_ansrs[j]["uid"]):
                answer = predicted_ansrs[j]['predicted_ans']
                question =  fit_ansrs[i]['qa']['question'] 
                question = question.split(' ')
                print(question)
                k = question.index('[ANSWER]')
  
                question = question[:k]+[str(answer)]+question[k+1:]
                question = ' '.join(question)

                fit_ansrs[i]['qa']['question'] = question
                print(fit_ansrs[i]['qa']['question'])

                break

    f = open("./converted_data/short_qstns.json")
    short_qstns = json.load(f)
    for i in fit_ansrs:
        short_qstns.append(i)
    json.dump(short_qstns, open("./input_data/MT2NET_input.json", "w"), indent=4)




parser = argparse.ArgumentParser()
parser.add_argument("--split_data", default = False)
parser.add_argument("--preprocess_data_to_decomposition", default = False)
parser.add_argument("--reconvert_decomposed_question_to_mhert", default = False)
parser.add_argument("--insert_answers_to_decomposed_questions",  default = False)

 
args = parser.parse_args()
 
if args.split_data:
    splitdata("./input_data/dev.json")


if args.preprocess_data_to_decomposition:
    data_input_to_decomposition("./converted_data/long_qstns.json")

if args.reconvert_decomposed_question_to_mhert:
    reconvert_decomposed_questions_to_mhert()

if args.insert_answers_to_decomposed_questions:
    insert_answers_to_decomposed_questions()



        

