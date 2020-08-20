Dependencies:

- Python 3
- pip install transformers torch transformers requests beautifulsoup4 nltk Flask

To run, from this directory:

```
python server.py
```

This will open a server on localhost:5000.

The entire application is plain-vanilla Flask and should be easy to customize for any deployment environment.

The inferences run decently well on a CPU, but hosting on a GPU will be lead to faster inferences and is also conducive to batch processing many documents at a time.
