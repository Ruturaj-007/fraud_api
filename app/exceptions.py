'''
uvicorn app.main:app --reload
streamlit run streamlit_app.py
'''

class ModelNotLoadedError(Exception):
    def __init__(self, trace_id: str):
        self.trace_id = trace_id


class InvalidTransactionError(Exception):
    def __init__(self, trace_id: str, reason: str):
        self.trace_id = trace_id
        self.reason = reason

class LLMProviderError(Exception):
    def __init__(self, trace_id: str, reason: str):
        self.trace_id = trace_id
        self.reason = reason