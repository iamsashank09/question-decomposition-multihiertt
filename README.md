# Applying Question Decomposition to Numerical Reasoning over Multi Hierarchical Data

Research Report: [Applying Question Decomposition to Numerical Reasoning over Multi-hierarchical Data](https://github.com/iamsashank09/question-decomposition-multihiertt/blob/main/Question%20Decompositon%20for%20Numerical%20Reasoning%20QA.pdf)

**Please follow the below instructions to run our code:**

1) Download checkpoints from https://drive.google.com/drive/folders/1LPVGoWm1Tsm4Asjc6soeFI3sMTfsAz1_?usp=sharing and place in: 

		Checkpoint_3/MT2NET_CodePipeline/MT2NET_models/

2) Download BERT checkpoints from https://drive.google.com/file/d/1XaMX-u5ZkWGH3f0gPrDtrBK1lKDU-QFk/view?usp=sharing and place it in 

	    Checkpoint_3/QuestionDecomposition_CodePipeline/model/

3) Run *main.py* which is in the project root folder,  we've built a single touch pipeline that runs both Decomposition and the QA pipeline:

	    python main.py

4) Decomposed questions can be found at:

	    Checkpoint_3/QuestionDecomposition_CodePipeline/decomposed_questions

5) Outputs at each step of MT2NET (QA Pipeline) can be found in the *outputs_intermediate* folder of the MT2NET_CodePipeline folder at:

	    Checkpoint_3/MT2NET_CodePipeline/outputs_intermediate
