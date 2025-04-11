from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import io
from dotenv import load_dotenv


class LeafDiseaseDetection:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.processor = AutoImageProcessor.from_pretrained("aashituli/promblemo")
        self.model =  AutoModelForImageClassification.from_pretrained("aashituli/promblemo")
        self.prompt_template = PromptTemplate(
            input_variables=["disease", "language"],
            template="""
            You are a helpful medical assistant.
            Give detailed symptoms and remedies for the disease: {disease}.
            Respond in {language}.
            """
        )
        self.converter_template = PromptTemplate(
            input_variable = ["finding" , "language"] ,
            template = """
            You are the best language converter or translator
            Convert the given finding : {finding} into the language : {language}
            Give only the converted text . No other languages should be included in
            the response . 
            """
        )
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            gemini_api_key = self.gemini_api_key ,
            temperature=0.7
        )
        self.chain = self.prompt_template | self.llm
        self.converter_chain = self.converter_template | self.llm


    def predict_leaf_disease(self ,image_bytes): 
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_class_idx]
        return predicted_class
    
    def find_information(self , disease , lang):
        information = self.chain.invoke({"disease": disease, "language": lang})
        return information

    def convert_text(self , finding , language):
        result = self.converter_chain.invoke({"finding" : finding , "language" : language})
        return result




if __name__ == "__main__":
    leafDiseaseDetection = LeafDiseaseDetection()
  # response = leafDiseaseDetection.find_information("light-Blight" , "Hindi")
    result = leafDiseaseDetection.convert_text("what is my name" , "punjabi")
    print(result)