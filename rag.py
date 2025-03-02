import os
import argparse
import yaml
import numpy as np

from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase
from bs4 import BeautifulSoup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())


class SimpleRAG:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if self.config.get("llm_model").startswith("claude"):
            self.llm = ChatAnthropic(
                model_name=self.config.get("llm_model", "claude-3-sonnet-20240229"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            self.embedder = HuggingFaceEmbeddings(
                model_name=self.config.get("embed_model", "BAAI/bge-small-en-v1.5")
            )
        elif self.config.get("llm_model").startswith("gpt"):
            self.llm = ChatOpenAI(
                model_name=self.config.get("llm_model", "gpt-3.5-turbo")
            )
            self.embedder = OpenAIEmbeddings(
                model=self.config.get("embed_model", "text-embedding-ada-002")
            )
        else:
            raise ValueError(f"Unsupported LLM model: {self.config.get('llm_model')}")

        self.neo4j_url = self.config.get("neo4j_url", "bolt://localhost:7687")
        self.neo4j_user = self.config.get("neo4j_user", "neo4j")
        self.neo4j_password = self.config.get("neo4j_password", "password")

        self.prompt_template = (
            "Use the following context to answer the question.\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

    def extract_html(self, directory_path):
        """Extract text from HTML files in a directory"""
        documents = []

        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist")
            return documents

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            html_content = f.read()

                        soup = BeautifulSoup(html_content, "html.parser")

                        # Remove script and style elements that won't contribute to content
                        for script in soup(["script", "style"]):
                            script.extract()

                        # Extract title for metadata
                        title = (
                            soup.title.string
                            if soup.title
                            else os.path.basename(file_path)
                        )

                        # Get the page text while maintaining some structure
                        text = soup.get_text(separator="\n", strip=True)

                        documents.append(
                            {"file_name": file_path, "title": title, "text": text}
                        )
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

        if not documents:
            print(f"Warning: No HTML files found in '{directory_path}'")

        return documents

    def chunk_documents(self, documents):
        """Chunk documents using recursive character text splitter"""
        # For HTML, we'll first use BeautifulSoup to extract text with structure preserved,
        # then use the configured separators for chunking
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1024),
            chunk_overlap=self.config.get("chunk_overlap", 0),
            separators=self.config.get(
                "separators",
                ["</p>", "</div>", "</section>", "<br>", ". ", "\n", " ", ""],
            ),
        )

        chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            for chunk in chunker.split_text(doc["text"]):
                chunks.append(
                    {
                        "doc_id": doc["file_name"],
                        "title": doc.get("title", ""),
                        "chunk": chunk,
                    }
                )

        return chunks

    def embed_chunks(self, chunks):
        """Generate embeddings for text chunks"""
        texts = [c["chunk"] for c in chunks]
        metadata = [
            {"doc_id": c["doc_id"], "title": c.get("title", ""), "content": c["chunk"]}
            for c in chunks
        ]

        embeddings = []
        batch_size = self.config.get("batch_size", 32)

        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self.embedder.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

        return embeddings, metadata

    def load_to_neo4j(self, embeddings, metadata, clear_db=False):
        """Load embeddings and metadata into Neo4j"""
        vector_dimension = len(embeddings[0])

        driver = GraphDatabase.driver(
            self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password)
        )

        with driver.session() as session:
            # Clear existing data and drop indexes only if clear_db flag is True
            if clear_db:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                print("Database cleared.")

                # Drop existing indexes
                session.run("DROP INDEX vector_index IF EXISTS")
                print("Existing indexes dropped.")

            # Create vector index if it doesn't exist
            index_query = f"""
            CREATE VECTOR INDEX vector_index IF NOT EXISTS FOR (v:Vector)
            ON (v.vector)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {vector_dimension},
                `vector.similarity_function`: 'COSINE'
              }}
            }}
            """
            session.run(index_query)
            print("Vector index created or verified.")

            # Insert vector nodes
            for vector_id, (vector, meta) in tqdm(
                enumerate(zip(embeddings, metadata)),
                desc="Inserting vectors",
                total=len(embeddings),
            ):
                # Convert vector to list if it's a NumPy array
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()

                # Merge vector node
                query = """
                MERGE (v:Vector { vector_id: $vector_id })
                SET
                    v.vector = $vector,
                    v.doc_id = $doc_id,
                    v.title = $title,
                    v.original_content = $content
                """

                params = {
                    "vector_id": vector_id,
                    "vector": vector,
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "content": meta.get("content", ""),
                }

                session.run(query, **params)

        driver.close()
        print("Data successfully inserted into Neo4j.")

    def run_ingestion_pipeline(self, html_dir, clear_db=False):
        """Run the complete ingestion pipeline"""
        print(f"Extracting HTML from {html_dir}...")
        docs = self.extract_html(html_dir)

        if not docs:
            print("Error: No documents to process. Ingestion pipeline aborted.")
            return False

        print(f"Extracted {len(docs)} documents")

        print("Chunking documents...")
        chunks = self.chunk_documents(docs)

        if not chunks:
            print("Error: No chunks created. Ingestion pipeline aborted.")
            return False

        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings...")
        embeddings, metadata = self.embed_chunks(chunks)

        if not embeddings or len(embeddings) == 0:
            print("Error: No embeddings generated. Ingestion pipeline aborted.")
            return False

        print("Loading data to Neo4j...")
        self.load_to_neo4j(embeddings, metadata, clear_db)

        print("Ingestion pipeline completed successfully.")
        return True

    def setup_qa(self):
        """Initialize the QA system"""
        vectorstore = Neo4jVector.from_existing_index(
            embedding=self.embedder,
            index_name="vector_index",
            url=self.neo4j_url,
            username=self.neo4j_user,
            password=self.neo4j_password,
            text_node_property="original_content",
            # metadata_node_properties=["title", "doc_id"],
        )

        prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 5)}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=False,
        )

        return qa_chain

    def answer_question(self, query, qa_chain=None):
        """Answer a question using the QA system"""
        if qa_chain is None:
            qa_chain = self.setup_qa()

        try:
            result = qa_chain.invoke({"query": query})
            return result
        except Exception as e:
            print(f"Error answering query '{query}': {e}")
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RAG Application")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ingest", "query"],
        required=True,
        help="Operation mode: ingest or query",
    )
    parser.add_argument(
        "--html_dir", type=str, help="Directory containing HTML files (for ingest mode)"
    )
    parser.add_argument("--query", type=str, help="Query to answer (for query mode)")
    parser.add_argument(
        "--clear_db",
        action="store_true",
        help="Clear database and drop indexes before ingestion",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    rag = SimpleRAG(args.config)

    if args.mode == "ingest":
        if not args.html_dir:
            print("Error: --html_dir is required for ingest mode")
            return

        success = rag.run_ingestion_pipeline(args.html_dir, args.clear_db)
        if not success:
            print("Ingestion pipeline failed. Please check the logs for details.")

    elif args.mode == "query":
        if not args.query:
            print("Error: --query is required for query mode")
            return

        qa_chain = rag.setup_qa()
        result = rag.answer_question(args.query, qa_chain)

        if result:
            print("\n--- Answer ---")
            print(result.get("result", "No answer found."))

            # Comment this in to see the source documents, ommitted for focus on the answer
            # print("\n--- Source Documents ---")
            # for idx, doc in enumerate(result.get("source_documents", []), start=1):
            #     title = doc.metadata.get("title", "Untitled")
            #     doc_id = doc.metadata.get("doc_id", "Unknown source")
            #     print(f"Document {idx}: {title} ({doc_id})")
            #     print(f"{doc.page_content[:200]}...\n")
        else:
            print("No answer could be generated.")


if __name__ == "__main__":
    main()
