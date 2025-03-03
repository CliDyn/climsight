from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    """
    Handles streaming output from LLM and agent workflow progress.
    """

    def __init__(self, container=None, display_method="markdown"):
        self.container = container
        self.display_method = display_method
        self.text = ""
        self.progress_text = ""  # New field for workflow progress

    def send_text(self, text: str) -> None:
        self.text += text
        self._display_text()
        
    def update_progress(self, progress: str) -> None:
        """Update UI with progress information from the agent workflow"""
        self.progress_text = progress
        self._display_progress()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self._display_text()

    def _display_text(self):
        if self.container:
            display_function = getattr(self.container, self.display_method, None)
            if display_function is not None:
                display_function(self.text)
            else:
                raise ValueError(f"Invalid display_method: {self.display_method}")
                
    def _display_progress(self):
        """Display progress information"""
        if self.container:
            # For Streamlit, we can display progress above the main text area
            # You might need to adjust based on your UI structure
            display_function = getattr(self.container, "info", None)
            if display_function is not None:
                display_function(self.progress_text)

    def get_text(self):
        return self.text