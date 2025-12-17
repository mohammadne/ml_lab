from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import numpy as np

# Connect to Milvus server (19530 is gRPC port)
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
# fields are like columns that will be stored
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto ID for primary key
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=100),  # 100-dimensional vectors
]

schema = CollectionSchema(fields, description="Vector database for AI applications")

# Create a collection in Milvus
collection = Collection(name="vectors_collection", schema=schema)

# Generate random vectors (100 vectors, each of 100 dimensions)
num_vectors = 100
dimension = 100
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Insert the vectors into the collection
data = [vectors.tolist()]

# Insert the vectors into the collection
collection.insert(data)
print(f"Inserted {num_vectors} vectors into Milvus!")

# Create an index for the 'vector' field
index_params = {
    "index_type": "IVF_FLAT",  # Index type, inverted file index for fast search
    "metric_type": "L2",  # Metric type, L2 for Euclidean distance
    "params": { # Additional parameters like nlist
        "nlist": 128, # controls how many partitions are used in the index. Larger values may result in faster search but require more memory
    }
}

# Create index on the 'vector' field
collection.create_index(field_name="vector", index_params=index_params)
print("Index created for the collection!")

# Load the collection into memory before searching
# This makes the collection ready for searching and retrieval.
collection.load()
print("Collection loaded into memory!")

# Create a query vector (Query with the first vector)
query_vector = vectors[0].reshape(1, -1)

# Perform a similarity search (find 3 nearest neighbors)
search_params = {"nprobe": 10}  # nprobe is a parameter controlling how many partitions to search in the index, higher values give more accurate results but take longer
results = collection.search(query_vector, "vector", search_params, limit=3)

# Print the search results
print("\nSearch Results:")
for result in results[0]:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Optionally, drop the collection when done (clean up)
collection.drop()
