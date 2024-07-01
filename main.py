from sentence_transformers import SentenceTransformer, util, models
from sentence_transformers.quantization import quantize_embeddings
import torch
import chromadb
import random
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Constants
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "TaskEmbeddings"

# Embedding function for ChromaDB using a pre-defined model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")

# Step 1: Generate Text Embeddings
def generate_embedding(task_description):
    embedder = SentenceTransformer(EMBED_MODEL)
    return embedder.encode(task_description, convert_to_tensor=True)

# Step 2: Connect to ChromaDB
# Containerized Approach
chroma_client = chromadb.HttpClient(
    host="chroma",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
    embedding_function=sentence_transformer_ef
)

# Step 3: Store Embeddings
def save_embedding_to_chromadb(task):
    collection.add(
        documents=[task["taskdescription"]],
        metadatas=[{"EmployeeId": task["empid"]}],
        ids=[task["taskid"]]
    )

# Task descriptions
taskdescs = [
    "Schedule and organize a project kickoff meeting to introduce team members and discuss project goals.",
    "Conduct interviews with key stakeholders to gather requirements and understand their expectations.",
    "Develop a detailed project timeline outlining key milestones and deadlines.",
    "Identify potential risks and uncertainties associated with the project and develop a risk mitigation plan.",
    "Clearly define the scope of the project, including deliverables, features, and exclusions.",
    "Allocate resources effectively, considering team members' skills, availability, and project requirements.",
    "Evaluate and choose the appropriate technology stack for the project, considering scalability and compatibility.",
    "Create a prototype or wireframe to visualize the project's user interface and functionality.",
    "Set up the project's version control system, development environment, and coding standards.",
    "Design the database schema and establish data relationships for efficient data management.",
    "Initiate frontend development, implementing the user interface and ensuring a responsive design.",
    "Begin backend development, focusing on server-side logic, database integration, and API development.",
    "Plan and conduct UAT sessions with stakeholders to validate that the system meets their requirements.",
    "Implement a comprehensive QA plan, including testing protocols, to ensure the software's reliability and functionality.",
    "Generate project documentation, including user manuals, technical documentation, and API documentation.",
    "Develop a deployment plan outlining the steps to release the project to production.",
    "Conduct training sessions for end-users and stakeholders on how to use the new system effectively.",
    "Address and resolve any bugs or issues identified during testing, and optimize system performance.",
    "Hold a project review meeting to assess the overall success of the project, gather feedback, and discuss lessons learned.",
]

# Ensure the collection is created
collection_status = False
while not collection_status:
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=sentence_transformer_ef
        )
        collection_status = True
    except Exception as e:
        pass

# Generate task data and save to ChromaDB
tasks = []
for i in range(len(taskdescs)):
    task = {
        "taskdescription": taskdescs[i],  # task description
        "empid": f"emp{random.randrange(1, 19)}",  # Assign employee ID Randomly
        "taskid": f"task{i+1}"  # Assign task ID (e.g., task1, task2, ...)
    }
    tasks.append(task)

print("_____________________________________________________________________________________________________")

# Save tasks to ChromaDB
for task in tasks:
    save_embedding_to_chromadb(task)

# Querying the collection
task_to_be_done = "Assign specific tasks to team members based on their expertise and the project requirements."
results = collection.query(query_texts=[task_to_be_done], n_results=4)
print(results)
