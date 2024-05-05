from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("MIIB-NLP/Arabic-question-generation")
tokenizer = AutoTokenizer.from_pretrained("MIIB-NLP/Arabic-question-generation")

tokenizer_english = T5Tokenizer.from_pretrained('ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation')
model_english = T5ForConditionalGeneration.from_pretrained('ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation')

# Define FastAPI app
app = FastAPI()

# Define request and response models
class QuestionRequest(BaseModel):
    context: str
    answer: str
    lang: str = 'arabic'

class QuestionResponse(BaseModel):
    question: str

# Define route for question generation
@app.post("/generate_question/")
async def generate_question(request: QuestionRequest) -> QuestionResponse:
    context = request.context
    answer = request.answer
    lang = request.lang

    if lang == 'arabic':
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

        question = tokenizer.decode(generated_ids[0], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True).replace('question: ', '')

        return QuestionResponse(question=question)
    elif lang == 'english':
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = tokenizer_english.encode_plus(
            input_text,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = model_english.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        question = tokenizer_english.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return QuestionResponse(question=question)
    # else:
    #     return 'Language not provided'



# Run the FastAPI app with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)