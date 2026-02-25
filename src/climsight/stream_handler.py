from langchain_core.callbacks.base import BaseCallbackHandler
import logging

logger = logging.getLogger(__name__)

class StreamHandler(BaseCallbackHandler):
    """
    Handles streaming output from LLM and agent workflow progress.
    Works with any container that has a display method (or None for headless mode).
    """

    def __init__(self, container=None, container2=None, display_method="markdown"):
        self.container = container
        self.container2 = container2
        self.display_method = display_method
        self.text = ""
        self.reference_text = ""
        self.progress_text = ""

    def send_text(self, text: str) -> None:
        self.text += text
        self._display_text()
        
    def send_reference_text(self, text: str) -> None:
        """Send separate text to the second container"""
        self.reference_text += text
        self._display_reference_text()

    def update_progress(self, progress: str) -> None:
        self.progress_text = progress
        self._display_progress()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self._display_text()

    def _display_text(self):
        if self.container:
            display_function = getattr(self.container, self.display_method, None)
            if display_function is not None:
                try:
                    display_function(self.text)
                except Exception as e:
                    logger.debug(f"Display unavailable, skipping: {self.text[:100]}")
            else:
                raise ValueError(f"Invalid display_method: {self.display_method}")
                
    def _display_reference_text(self):
        if self.container2:
            display_function = getattr(self.container2, self.display_method, None)
            if display_function is not None:
                try:
                    display_function(self.reference_text)
                except Exception as e:
                    logger.debug(f"Reference display unavailable, skipping")
            else:
                raise ValueError(f"Invalid display_method: {self.display_method}")

    def _display_progress(self):
        if self.container:
            display_function = getattr(self.container, "info", None)
            if display_function is not None:
                try:
                    display_function(self.progress_text)
                except Exception as e:
                    logger.debug(f"Progress update (no UI context): {self.progress_text}")

    def get_text(self):
        return self.text

    def get_reference_text(self):
        return self.reference_text