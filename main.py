import subprocess

sep = '*' * 10


MT2NET_dir = "MT2NET_CodePipeline"
QD_dir = "QuestionDecomposition_CodePipeline"

# *********************************  Step - 1 ******************************************************************

subprocess.run("python preprocess.py --split_data True", cwd=f"{MT2NET_dir}", shell=True )
subprocess.run("python preprocess.py --preprocess_data_to_decomposition True", cwd=f"{MT2NET_dir}", shell=True )

# Decompose questions
subprocess.run("python3 main.py --do_predict --model span-predictor --output_dir decomposed_questions --init_checkpoint model/model.pt --predict_file data/dev.json --max_seq_length 100 --max_n_answers 1 --prefix dev_ --with_key", cwd=f"{QD_dir}", shell=True)
subprocess.run("python preprocess.py --reconvert_decomposed_question_to_mhert True", cwd=f"{MT2NET_dir}", shell=True )

# Retriever Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/retriever_model.ckpt --config MT2NET_configs/retriever_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Question Classification Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/question_classification_model.ckpt --config MT2NET_configs/question_classification_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Combine Retriever Results
subprocess.run("python combine_retriever_results_pipeline.py", cwd=f"{MT2NET_dir}", shell=True)

# Span selection Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/span_selection_model.ckpt --config MT2NET_configs/span_selection_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Program generation Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/program_generation_model.ckpt --config MT2NET_configs/program_generation_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

#Evaluation
subprocess.run("python evaluate_pipeline.py", cwd=f"{MT2NET_dir}", shell=True)

# *********************************  Step - 2  ******************************************************************

subprocess.run("python preprocess.py --insert_answers_to_decomposed_questions True", cwd=f"{MT2NET_dir}", shell=True )

# Retriever Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/retriever_model.ckpt --config MT2NET_configs/retriever_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Question Classification Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/question_classification_model.ckpt --config MT2NET_configs/question_classification_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Combine Retriever Results
subprocess.run("python combine_retriever_results_pipeline.py", cwd=f"{MT2NET_dir}", shell=True)

# Span Selection Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/span_selection_model.ckpt --config MT2NET_configs/span_selection_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Program generation Module
subprocess.run("python trainer.py predict --ckpt_path MT2NET_models/program_generation_model.ckpt --config MT2NET_configs/program_generation_inference.yaml", cwd=f"{MT2NET_dir}", shell=True)

# Evaluation
subprocess.run("python evaluate_pipeline.py", cwd=f"{MT2NET_dir}", shell=True)
