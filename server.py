import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from google import genai

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

client_groq = Groq(api_key=GROQ_KEY)
client_gemini = genai.Client(api_key=GEMINI_KEY)

app = FastAPI(title="OmniVoice AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPTS = {
    "email": "Transforma esta transcripción en un correo profesional estructurado con asunto.",
    "tasks": "Extrae las tareas y compromisos pendientes en una lista de viñetas Markdown.",
    "summary": "Crea un resumen ejecutivo conciso con los puntos clave del audio.",
    "whatsapp": "Convierte este audio en un mensaje informal, claro y directo listo para enviar por WhatsApp.",
    "translate_en": "Traduce el mensaje transcrito directamente al inglés manteniendo un tono profesional.",
    "meeting_notes": "Genera una minuta de reunión estructurada: Asistentes mencionados, Temas tratados y Acuerdos."
}

@app.get("/")
def read_root():
    return {"status": "ok", "service": "OmniVoice AI API Engine"}

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("summary"),
    custom_prompt: str = Form(None)
):
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        with open(temp_filename, "rb") as audio_file:
            transcription = client_groq.audio.transcriptions.create(
                file=(temp_filename, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="es"
            )

        prompt_instruction = custom_prompt if custom_prompt else PROMPTS.get(mode, PROMPTS["summary"])

        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{prompt_instruction}\n\nTexto transcrito:\n\"\"\"{transcription}\"\"\""
        )

        return {
            "transcription": transcription,
            "result": response.text,
            "mode_used": mode
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)