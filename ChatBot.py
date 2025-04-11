from PyPDF2 import PdfReader
from pydantic import  BaseModel
import os


class ChatBotModel :

    def __init__(self , pdf_path):
        self.pdf_path = "agronomy-book.pdf"

    @property
    def llm_self(self):
        return
