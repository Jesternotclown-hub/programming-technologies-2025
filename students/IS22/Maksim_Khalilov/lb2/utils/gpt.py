from openai import AsyncOpenAI
from config import OPENAI_API_KEY
import logging
from utils.database import get_messages, add_message

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
SYSTEM_PROMPT = "Ты заикающийся ассистент."

async def get_response(user_id: int, user_message: str, client: AsyncOpenAI) -> str:
    try:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

        # добавляем последние сообщения из базы
        for role, content in get_messages(user_id):
            history.append({"role": role, "content": content})

        # добавляем новое сообщение пользователя
        history.append({"role": "user", "content": user_message})

        response = await client.responses.create(
            model="gpt-4o-mini",
            input=history
        )

        answer = response.output_text

        # сохраняем в БД
        add_message(user_id, "user", user_message)
        add_message(user_id, "assistant", answer)

        return answer

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return "Произошла ошибка при получении ответа"
