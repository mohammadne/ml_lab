# Vector Database

- https://www.dailydoseofds.com/a-beginner-friendly-and-comprehensive-deep-dive-on-vector-databases/
- https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications
- [deeplerning ai course for usages of vector-database](https://learn.deeplearning.ai/courses/building-applications-vector-databases)

## Local Milvus

Local Milvus + MinIO setup for vector similarity search:

```bash
# milvus examples
docker compose up

# mnist-server
py train.py
uvicorn app.main:app --reload
http://127.0.0.1:8000/docs
```
