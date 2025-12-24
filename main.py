import asyncio
import os
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob, ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from dotenv import load_dotenv
from rag_utils import RAGPipeline

pre_download_model()

load_dotenv()
VIDEOSDK_AUTH_TOKEN = os.getenv('VIDEOSDK_AUTH_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
dag_folder = os.path.join(os.path.dirname(__file__), 'docs')
rag_pipeline = RAGPipeline(docs_folder=dag_folder, openai_api_key=OPENAI_API_KEY)

class RAGVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful AI assistant with access to private knowledge base.")
    async def on_enter(self):
        await self.session.say("Hello! How can I help using my knowledge base?")
    async def on_message(self, user_message):
        question = user_message.text
        docs, scores = rag_pipeline.retrieve(question)
        rag_context = "\n".join(docs) if docs else None
        if rag_context:
            context_str = f"Use ONLY the following info to answer. If not sure, say so.\n{rag_context}"
            response = await self.session.complete(prompt=question, system=context_str)
        else:
            response = await self.session.complete(prompt=question)
        await self.session.say(response)
    async def on_exit(self):
        await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    agent = RAGVoiceAgent()
    conversation_flow = ConversationFlow(agent)
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )
    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )
    try:
        await context.connect()
        await session.start()
        await asyncio.Event().wait()
    finally:
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(
        name="RAG Voice Agent",
        playground=True
    )
    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()

