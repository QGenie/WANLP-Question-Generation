from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("MIIB-NLP/Arabic-question-generation")
tokenizer = AutoTokenizer.from_pretrained("MIIB-NLP/Arabic-question-generation")

# Define FastAPI app
app = FastAPI()

# Define request and response models
class QuestionRequest(BaseModel):
    context: str
    answer: str

class QuestionResponse(BaseModel):
    question: str

# Define route for question generation
@app.post("/generate_question/")
async def generate_question(request: QuestionRequest) -> QuestionResponse:
    context = request.context
    answer = request.answer

    # Generate question
    text = "context: " + context + " " + "answer: " + answer + " </s>"
    text_encoding = tokenizer.encode_plus(text, return_tensors="pt")

    model.eval()
    generated_ids = model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=64,
        num_beams=5,
        num_return_sequences=1
    )

    question = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).replace('question: ', '')

    return QuestionResponse(question=question)

# Run the FastAPI app with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)