# RAG_ITT
1. alibaba_gte-large-en-v1.5_with_mistral.py
2. bert_embeddings_with_mistral.py
3. alibaba_gte-base-en-v1.5_with_mistral.py
4. roberta_embeddings_with_mistral.py
5. requirements.txt

Above 4 files contain 4 opensource embedding models along with mistralai/Mixtral-8x7B-Instruct-v0.1 as open source LLM

requirements.txt contains all the libraries which are needed to run the application.

I have used 4 embedding models which are compatible with CPU and give acceptable answers. The best performing Embedding model is alibaba_gte-base-en-v1.5(alibaba_gte-base-en-v1.5_with_mistral.py) in time-accuracy tradeoff, as it gives way better answer than Bert and Roberta embedding models and faster than alibaba_gte-large-en-v1.5 embedding model.

How to run above files:

1. Clone the above repository.
2. Download required libraries from requirements.txt file(for Windows OS)
3. Use "streamlit run [filename]" command to run any file
4. You will see below screen(shown in picture below) image
![image](https://github.com/user-attachments/assets/0772c8af-d37b-4f32-a894-8bd27179764c)


In the above image as shown first you have to upload a pdf file and wait for sometime to process it, after that Enter your query in textbox and press Get Answer button. In Response you will get the required Answer along with Context.

