# RAG-LLM
Data-X assignment
### Conversational Document Summarizer

#### Description:

This application leverages Large Language Models (LLMs) to provide a conversational document summarization experience. Users can input questions related to a given document, and the system responds by generating coherent answers based on the document's content. The application utilizes advanced prompt engineering and the Retrieval-Augmented Generation (RAG) model for dynamic knowledge integration.

#### Key Features:

1. **Conversational Summarization:**
   - Users can interactively ask questions about a document, and the application responds with coherent answers, leveraging LLMs.

2. **Dynamic Knowledge Integration (RAG):**
   - The application incorporates RAG to dynamically retrieve relevant information from external knowledge sources, enhancing the accuracy and richness of responses.

3. **Document Embeddings and Similarity Search:**
   - The system extracts text from PDF documents, processes it into chunks, and generates embeddings using Google's Generative AI. Faiss, a similarity search tool, is employed to find relevant document sections.

4. **User-Friendly Interface:**
   - The interface is designed with Streamlit, providing an intuitive user experience. Users input questions, and the application displays both the ChatGPT and RAG responses.

#### Instructions:

1. **API Key Setup:**
   - Run the application and provide your Google API Key when prompted. This key is used for accessing external services.

2. **Document Processing:**
   - The application processes PDF documents located in the specified folder (`pdf_folder`). Ensure the folder contains relevant PDF files.

3. **User Interaction:**
   - After processing, the interface prompts users to input questions related to the document.

4. **Response Display:**
   - The system displays responses from both ChatGPT and RAG, allowing users to compare and analyze the information.

#### Technical Innovations:

- **Prompt Engineering:**
   - The application utilizes a sophisticated prompt template to guide user interactions, enhancing the context provided to the LLMs.

- **RAG Integration:**
   - RAG is employed for dynamic knowledge retrieval, making the application more informative and context-aware.

- **Document Embeddings and Faiss:**
   - The use of document embeddings and Faiss enables efficient similarity search, enhancing the user's experience.

#### Acknowledgments:

- The application uses the ChatGoogleGenerativeAI model for conversational responses.
- RAG components are based on the Facebook RAG model for dynamic knowledge integration.
- Google's API Key is used for document processing.

Feel free to explore and interact with the application to experience conversational document summarization powered by state-of-the-art language models and dynamic knowledge retrieval. If you encounter any issues or have suggestions for improvement, please provide feedback. Thank you for using the application!
