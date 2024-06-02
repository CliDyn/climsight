from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    """
    Handles streaming output from LLM. Can work with both Streamlit and prompt-based applications.
    """

    def __init__(self, container=None, display_method="markdown"):
        self.container = container
        self.display_method = display_method
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container:
            display_function = getattr(self.container, self.display_method, None)
            if display_function is not None:
                display_function(self.text)
            else:
                raise ValueError(f"Invalid display_method: {self.display_method}")

    def get_text(self):
        return self.text
